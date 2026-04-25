"""
Fine-tune EfficientNet-B4 on FF++ face images for deepfake detection.

Two-phase training
------------------
Phase 1 — warm-up (EPOCHS_WARMUP epochs, lr=1e-3):
    EfficientNet base is frozen. Only the added Dense head is trained.

Phase 2 — fine-tune (EPOCHS_FT epochs, lr=1e-5):
    Top UNFREEZE_N layers of EfficientNet are unfrozen.
    Very low LR preserves ImageNet knowledge while adapting to deepfake artefacts.

Video-level split
-----------------
All frames of one video go to the same split (train or val).

Class balancing
---------------
Fake frames are undersampled to match the real frame count (1:1 ratio)
before the TF Dataset is built. This is more reliable than class_weight
at 1:3 imbalance — class_weight causes gradient scale instability and
still produces biased predictions (TN=144, FP=336 in the previous run).

Augmentation
------------
- Random JPEG quality (40-95): the most important augmentation for
  cross-dataset generalisation. FF++ uses c23 compression; Celeb-DF uses
  a different encoder. If the model sees only one quality level it learns
  the compression artefacts as the fake signal, not the actual manipulation.
- Hue / saturation jitter: colour-domain robustness.
- Flip, brightness, contrast: standard geometric/photometric augmentation.

Threshold optimisation
----------------------
After training, the validation set is used to find the threshold that
maximises F1. The result is printed so you can set THRESHOLD in
test_celebdf_finetuned.py.

Saved artefacts
---------------
efficientnet_finetuned.keras  — best checkpoint (by val_accuracy)
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# ── Config ─────────────────────────────────────────────────────────────────────
FACES_DIR     = "faces_dataset"
MODEL_PATH    = "efficientnet_finetuned.keras"
IMG_SIZE      = 380
BATCH_SIZE    = 16
UNFREEZE_N    = 50       # top N EfficientNet layers to unfreeze in Phase 2
LR_WARMUP     = 1e-3
LR_FINETUNE   = 1e-5
EPOCHS_WARMUP = 10
EPOCHS_FT     = 30


# ── Video-level split ──────────────────────────────────────────────────────────
def get_video_id(filepath: str) -> str:
    """Drop the frame-index suffix to get the video identifier."""
    return "_".join(os.path.basename(filepath).split("_")[:-1])


real_frames = sorted(glob.glob(os.path.join(FACES_DIR, "real", "*.jpg")))
fake_frames = sorted(glob.glob(os.path.join(FACES_DIR, "fake", "*.jpg")))

print(f"Frames on disk  - real: {len(real_frames)}, fake: {len(fake_frames)}")

real_vids = sorted(set(get_video_id(f) for f in real_frames))
fake_vids = sorted(set(get_video_id(f) for f in fake_frames))

print(f"Unique videos   - real: {len(real_vids)}, fake: {len(fake_vids)}")

# Split by VIDEO ID so no frame leaks across the boundary
train_rv, val_rv = train_test_split(real_vids, test_size=0.2, random_state=42)
train_fv, val_fv = train_test_split(fake_vids, test_size=0.2, random_state=42)

train_rv_set, val_rv_set = set(train_rv), set(val_rv)
train_fv_set, val_fv_set = set(train_fv), set(val_fv)

train_pairs  = [(f, 0) for f in real_frames if get_video_id(f) in train_rv_set]
train_pairs += [(f, 1) for f in fake_frames if get_video_id(f) in train_fv_set]
val_pairs    = [(f, 0) for f in real_frames if get_video_id(f) in val_rv_set]
val_pairs   += [(f, 1) for f in fake_frames if get_video_id(f) in val_fv_set]

# Undersample fake training frames to 1:1 ratio.
# Handles imbalance in the data itself — more reliable than class_weight at 1:3.
real_train_pairs = [(p, l) for p, l in train_pairs if l == 0]
fake_train_pairs = [(p, l) for p, l in train_pairs if l == 1]
rng_bal = np.random.default_rng(0)
sampled_idx = rng_bal.choice(len(fake_train_pairs),
                              size=len(real_train_pairs), replace=False)
train_pairs = real_train_pairs + [fake_train_pairs[i] for i in sampled_idx]
np.random.default_rng(42).shuffle(train_pairs)

train_paths  = [p for p, _ in train_pairs]
train_labels = [l for _, l in train_pairs]
val_paths    = [p for p, _ in val_pairs]
val_labels   = [l for _, l in val_pairs]

print(f"Train frames (balanced): {len(train_paths)}  "
      f"(real={train_labels.count(0)}, fake={train_labels.count(1)})")
print(f"Val   frames: {len(val_paths)}  "
      f"(real={val_labels.count(0)}, fake={val_labels.count(1)})")


# ── TF Dataset ─────────────────────────────────────────────────────────────────
def load_image(path: tf.Tensor, label: tf.Tensor):
    """Load and resize; keep uint8 so random_jpeg_quality can be applied."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)          # uint8 RGB
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.uint8)                         # resize returns float — cast back
    return img, tf.cast(label, tf.float32)


def augment(img: tf.Tensor, label: tf.Tensor):
    """
    Augmentation pipeline for training frames.

    Stage 1 (uint8)  — JPEG quality simulation.
    Stage 2 ([0,1])  — hue / saturation jitter (these ops require [0,1] range).
    Stage 3 ([-1,1]) — EfficientNet scaling, then flip / brightness / contrast.

    JPEG quality (40-95) is the most important augmentation here: it forces
    the model to detect actual face manipulation rather than compression
    level, which is the single biggest driver of cross-dataset degradation.
    """
    # --- uint8: codec diversity ---
    img = tf.image.random_jpeg_quality(img, min_jpeg_quality=40, max_jpeg_quality=95)

    # --- [0, 1] float: colour augmentation ---
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_hue(img, max_delta=0.05)
    img = tf.image.random_saturation(img, lower=0.7, upper=1.3)

    # --- EfficientNet scale: [0,1] -> [-1,1] (same as preprocess_input) ---
    img = img * 2.0 - 1.0

    # --- any float range: geometric / intensity ---
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, label


def preprocess_val(img: tf.Tensor, label: tf.Tensor):
    """Deterministic preprocessing for validation — no augmentation."""
    img = tf.cast(img, tf.float32) / 127.5 - 1.0        # matches preprocess_input
    return img, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(load_image,   num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment,      num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(2048, seed=42)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(load_image,     num_parallel_calls=tf.data.AUTOTUNE)
    .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ── Model ──────────────────────────────────────────────────────────────────────
base = EfficientNetB4(
    weights="imagenet",
    include_top=False,
    pooling=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False      # frozen for warm-up

x   = base.output
x   = layers.GlobalAveragePooling2D(name="gap")(x)
x   = layers.Dropout(0.3, name="dropout")(x)
out = layers.Dense(1, activation="sigmoid", name="output")(x)

model = Model(base.input, out)
print(f"\nTotal params: {model.count_params():,}")
print(f"Trainable params (warm-up): "
      f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}")


# ── Phase 1: warm-up ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Phase 1: Warm-up - training Dense head only")
print("="*60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# No class_weight — imbalance is handled by undersampling in the dataset.
model.fit(
    train_ds,
    epochs=EPOCHS_WARMUP,
    validation_data=val_ds,
    callbacks=[
        EarlyStopping(patience=3, monitor="val_accuracy",
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, min_lr=1e-6, verbose=1),
    ],
)


# ── Phase 2: fine-tune ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"Phase 2: Fine-tune - unfreeze top {UNFREEZE_N} EfficientNet layers")
print("="*60)

base.trainable = True
for layer in base.layers[:-UNFREEZE_N]:
    layer.trainable = False

n_trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"Trainable params (fine-tune): {n_trainable:,}")

# Recompile with a much lower LR to avoid destroying ImageNet features
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=EPOCHS_FT,
    validation_data=val_ds,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_accuracy",
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ],
)


# ── Final validation evaluation ────────────────────────────────────────────────
print("\n" + "="*60)
print("Final validation evaluation (best checkpoint)")
print("="*60)

best = tf.keras.models.load_model(MODEL_PATH)

all_probs, all_true = [], []
for img_batch, lbl_batch in val_ds:
    probs = best(img_batch, training=False).numpy().flatten()
    all_probs.extend(probs.tolist())
    all_true.extend(lbl_batch.numpy().tolist())

all_probs = np.array(all_probs)
all_true  = np.array(all_true)

# Find the threshold that maximises F1 on the validation set.
# 0.5 is rarely optimal after fine-tuning — the sigmoid output distribution
# shifts depending on training data composition.
best_thr, best_f1_thr = 0.5, 0.0
for thr in np.arange(0.25, 0.76, 0.01):
    _pred = (all_probs >= thr).astype(int)
    _f1   = f1_score(all_true, _pred, zero_division=0)
    if _f1 > best_f1_thr:
        best_f1_thr = _f1
        best_thr    = thr

print(f"Optimal threshold: {best_thr:.2f}  (val F1={best_f1_thr:.4f})")
print(f"Set THRESHOLD = {best_thr:.2f} in test_celebdf_finetuned.py\n")

y_pred = (all_probs >= best_thr).astype(int)

acc_v  = accuracy_score(all_true, y_pred)
prec_v = precision_score(all_true, y_pred, zero_division=0)
rec_v  = recall_score(all_true, y_pred, zero_division=0)
f1_v   = f1_score(all_true, y_pred, zero_division=0)
cm_v   = confusion_matrix(all_true, y_pred)

print(f"Accuracy:  {acc_v:.4f} ({acc_v*100:.2f}%)")
print(f"Precision: {prec_v:.4f} ({prec_v*100:.2f}%)")
print(f"Recall:    {rec_v:.4f} ({rec_v*100:.2f}%)")
print(f"F1-score:  {f1_v:.4f} ({f1_v*100:.2f}%)")
print("Confusion Matrix:\n", cm_v)
print(f"\nModel saved to: {MODEL_PATH}")
