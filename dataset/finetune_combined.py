"""
finetune_combined.py
====================
Fine-tune EfficientNet-B0 on combined FF++ + Celeb-DF face images.

Key design decisions
--------------------
- **EfficientNetB0 (224×224)** instead of B4 (380×380):
  ~5× faster on CPU, minimal accuracy loss for in-distribution binary
  classification.  B4's extra capacity only matters for cross-dataset
  generalisation, which we explicitly dropped.

- **Symmetric augmentation** for both classes:
  Asymmetric augmentation (heavy for real, light for fake) teaches the model
  to detect *augmentation artefacts* rather than *deepfake artefacts*.
  Result: ~86% train accuracy but 50% val accuracy.

- **Per-sample weights** across (source, label) cells instead of class_weight:
  No data is discarded. Rebalances the four cells FF++/Celeb-DF × real/fake
  equally, so the model doesn't learn "real = celebrity-style" just because
  Celeb-DF contributes ~3× more real frames than FF++.

- **Monitor val_auc** (not val_accuracy):
  AUC is threshold-independent and robust to class imbalance.

- **Video-level threshold optimisation** after training:
  Validation frames are grouped by video, aggregated via median, then the
  threshold that maximises F1 is saved to model_config.json.

- **tf.data.cache()** after JPEG decode:
  First epoch reads from disk; all subsequent epochs read from RAM.

Saved artefacts
---------------
efficientnet_combined.keras   best checkpoint (by val_auc)
test_split.csv                held-out test video IDs + labels
model_config.json             threshold, img_size, backbone, aggregation
"""

import os
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from keras.applications import EfficientNetB0
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
)

# ── Config ─────────────────────────────────────────────────────────────────────
FF_FACES_DIR      = "faces_dataset"
CELEBDF_FACES_DIR = "celebdf_faces"
MODEL_PATH        = "efficientnet_combined.keras"
TEST_SPLIT_PATH   = "test_split.csv"
CONFIG_PATH       = "model_config.json"

IMG_SIZE         = 224        # B0 native resolution (B4 was 380)
BATCH_SIZE       = 32         # Larger batch OK with smaller images
UNFREEZE_N       = 80         # B0 has ~237 layers — unfreeze top ~1/3
LR_WARMUP        = 1e-3
LR_FINETUNE      = 5e-5       # 1e-5 was too low — head can't escape ~0.5 plateau
EPOCHS_WARMUP    = 1          # Frozen ImageNet features can't discriminate deepfakes
EPOCHS_FT        = 30
FRAMES_PER_VIDEO = 10
RANDOM_SEED      = 42


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_video_id(filepath: str) -> str:
    """Strip frame-index suffix to recover the video identifier."""
    return "_".join(os.path.basename(filepath).split("_")[:-1])


def collect_frames(faces_dir: str, prefix: str):
    """Return (real_frames, fake_frames) as lists of (path, video_id)."""
    real_frames, fake_frames = [], []
    for label_name, bucket in [("real", real_frames), ("fake", fake_frames)]:
        folder = os.path.join(faces_dir, label_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
        for path in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            vid_id = prefix + get_video_id(path)
            bucket.append((path, vid_id))
    return real_frames, fake_frames


def load_frame_numpy(path: str):
    """Load a single face image for numpy-based evaluation.

    Returns float32 in [0, 255] — EfficientNetB0's include_preprocessing
    handles ImageNet normalisation internally.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32")


# ── Collect all frames ─────────────────────────────────────────────────────────
print("=" * 60)
print("Collecting frames from FF++ and Celeb-DF")
print("=" * 60)

ff_real, ff_fake = collect_frames(FF_FACES_DIR, prefix="ff__")
cdf_real, cdf_fake = collect_frames(CELEBDF_FACES_DIR, prefix="cdf__")

all_real = ff_real + cdf_real
all_fake = ff_fake + cdf_fake

print(f"FF++     — real: {len(ff_real):>5}, fake: {len(ff_fake):>5}")
print(f"Celeb-DF — real: {len(cdf_real):>5}, fake: {len(cdf_fake):>5}")
print(f"Combined — real: {len(all_real):>5}, fake: {len(all_fake):>5}")


# ── Label verification ────────────────────────────────────────────────────────
print("\n── Label verification ──")
for faces_dir, prefix, desc in [
    (FF_FACES_DIR, "ff__", "FF++"),
    (CELEBDF_FACES_DIR, "cdf__", "Celeb-DF"),
]:
    for label_name, expected in [("real", 0), ("fake", 1)]:
        folder = os.path.join(faces_dir, label_name)
        n = len(glob.glob(os.path.join(folder, "*.jpg")))
        sample = os.path.basename(glob.glob(os.path.join(folder, "*.jpg"))[0])
        print(f"  {desc:>8}/{label_name:<4} → label={expected}  "
              f"({n:>5} frames)  sample: {sample}")
print("  Convention: real=0, fake=1  ✓")


# ── Video-level 3-way split (70 / 15 / 15) ────────────────────────────────────
print("\n── Video-level split (70% train / 15% val / 15% test) ──")

real_vids = sorted(set(vid for _, vid in all_real))
fake_vids = sorted(set(vid for _, vid in all_fake))
print(f"Unique videos — real: {len(real_vids)}, fake: {len(fake_vids)}")

real_train_v, real_temp_v = train_test_split(
    real_vids, test_size=0.30, random_state=RANDOM_SEED)
fake_train_v, fake_temp_v = train_test_split(
    fake_vids, test_size=0.30, random_state=RANDOM_SEED)
real_val_v, real_test_v = train_test_split(
    real_temp_v, test_size=0.50, random_state=RANDOM_SEED)
fake_val_v, fake_test_v = train_test_split(
    fake_temp_v, test_size=0.50, random_state=RANDOM_SEED)

train_set = set(real_train_v + fake_train_v)
val_set   = set(real_val_v   + fake_val_v)
test_set  = set(real_test_v  + fake_test_v)


def partition(frames, label):
    tr, va, te = [], [], []
    for path, vid in frames:
        if   vid in train_set: tr.append((path, vid, label))
        elif vid in val_set:   va.append((path, vid, label))
        elif vid in test_set:  te.append((path, vid, label))
    return tr, va, te


r_tr, r_va, r_te = partition(all_real, 0)
f_tr, f_va, f_te = partition(all_fake, 1)

train_all = r_tr + f_tr
val_all   = r_va + f_va

print(f"Train — real: {len(r_tr):>5}, fake: {len(f_tr):>5}, total: {len(train_all)}")
print(f"Val   — real: {len(r_va):>5}, fake: {len(f_va):>5}, total: {len(val_all)}")
print(f"Test  — real: {len(r_te):>5}, fake: {len(f_te):>5}, total: {len(r_te)+len(f_te)}")


# ── Save test split ───────────────────────────────────────────────────────────
test_records = []
for vid in sorted(real_test_v):
    test_records.append({"video_id": vid, "label": 0})
for vid in sorted(fake_test_v):
    test_records.append({"video_id": vid, "label": 1})
pd.DataFrame(test_records).to_csv(TEST_SPLIT_PATH, index=False)
print(f"\nTest split → {TEST_SPLIT_PATH} "
      f"({len(real_test_v)} real + {len(fake_test_v)} fake videos)")


# ── Compute per-sample weights (balances real/fake AND source) ────────────────
# class_weight balances only real vs fake. Our dataset has a second imbalance:
# within the real class, Celeb-DF contributes ~3× more frames than FF++
# (and within fake the ratio is reversed). If left uncorrected, the model
# learns "real = celebrity-style" and misclassifies YouTube-style real videos.
# Per-sample weights rebalance all four (source, label) cells equally; this
# subsumes class_weight.
train_paths   = [p for p, _, _ in train_all]
train_labels  = [l for _, _, l in train_all]
train_sources = ["ff" if v.startswith("ff__") else "cdf"
                 for _, v, _ in train_all]

cell_counts = defaultdict(int)
for src, lbl in zip(train_sources, train_labels):
    cell_counts[(src, lbl)] += 1

n_total = len(train_all)
n_cells = len(cell_counts)   # expected: 4
sample_weight_map = {
    cell: n_total / (n_cells * count)
    for cell, count in cell_counts.items()
}

train_weights = [
    sample_weight_map[(src, lbl)]
    for src, lbl in zip(train_sources, train_labels)
]

print(f"\n── Per-sample weights — (source, label) cells ──")
for (src, lbl), w in sorted(sample_weight_map.items()):
    name = f"{src.upper():<4} {'real' if lbl==0 else 'fake'}"
    print(f"  {name}: weight = {w:.4f}  ({cell_counts[(src, lbl)]} frames)")
print(f"  (replaces class_weight — subsumes both real/fake and source balance)")


# ── Group validation frames by video (for post-training threshold tuning) ─────
val_videos = defaultdict(lambda: {"paths": [], "label": None})
for path, vid, label in val_all:
    val_videos[vid]["paths"].append(path)
    val_videos[vid]["label"] = label

print(f"\nVal videos for threshold tuning: {len(val_videos)} "
      f"(real: {sum(1 for v in val_videos.values() if v['label']==0)}, "
      f"fake: {sum(1 for v in val_videos.values() if v['label']==1)})")


# ── TF Dataset ─────────────────────────────────────────────────────────────────
# Pre-shuffle paths so classes are interleaved before .cache()
rng = np.random.default_rng(RANDOM_SEED)
perm = rng.permutation(len(train_paths))
train_paths   = [train_paths[i]   for i in perm]
train_labels  = [train_labels[i]  for i in perm]
train_weights = [train_weights[i] for i in perm]

val_paths  = [p for p, _, _ in val_all]
val_labels = [l for _, _, l in val_all]


def load_image_train(path: tf.Tensor, label: tf.Tensor, weight: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.uint8)
    return img, tf.cast(label, tf.float32), tf.cast(weight, tf.float32)


def load_image_val(path: tf.Tensor, label: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.uint8)
    return img, tf.cast(label, tf.float32)


def augment(img: tf.Tensor, label: tf.Tensor, weight: tf.Tensor):
    """Symmetric augmentation — identical for real and fake.

    Outputs uint8-range float32 in [0, 255]. EfficientNetB0 has
    include_preprocessing=True by default — it does ImageNet normalisation
    internally. Pre-normalising here would feed garbage to the frozen BN
    layers and collapse the model to a constant.
    """
    img = tf.image.random_jpeg_quality(img, min_jpeg_quality=70, max_jpeg_quality=100)
    img = tf.cast(img, tf.float32)                      # stays in [0, 255]
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=25.0)        # ±10% of 255
    img = tf.image.random_contrast(img, lower=0.90, upper=1.10)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label, weight


def preprocess_val(img: tf.Tensor, label: tf.Tensor):
    img = tf.cast(img, tf.float32)                      # stays in [0, 255]
    return img, label


# .cache() after decode: first epoch reads disk, rest read RAM (~2 GB).
# Remove .cache() if RAM is tight.
# Dataset yields (img, label, sample_weight); Keras automatically applies
# per-sample weighting from the third tuple element — no class_weight needed.
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels, train_weights))
    .map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .shuffle(4096, seed=RANDOM_SEED, reshuffle_each_iteration=True)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(load_image_val, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print(f"\n── TF Dataset ──")
print(f"  Train: {len(train_paths)} frames, {len(train_paths)//BATCH_SIZE} batches/epoch")
print(f"  Val:   {len(val_paths)} frames, {len(val_paths)//BATCH_SIZE} batches/epoch")


# ── Diagnostic callback ───────────────────────────────────────────────────────
class OutputDistribution(tf.keras.callbacks.Callback):
    """Prints per-class probability distribution after each epoch.

    If real-mean and fake-mean stay close together, the model isn't learning
    discriminative features — symptom of frozen base, too-low LR, or augmentation
    leakage. A healthy run shows separation growing each epoch.
    """
    def __init__(self, val_ds, val_labels):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = np.array(val_labels)

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.val_ds, verbose=0).flatten()
        real_p = probs[self.val_labels == 0]
        fake_p = probs[self.val_labels == 1]
        sep = float(np.mean(fake_p) - np.mean(real_p))
        print(f"  [diag] real μ={np.mean(real_p):.3f} σ={np.std(real_p):.3f} | "
              f"fake μ={np.mean(fake_p):.3f} σ={np.std(fake_p):.3f} | "
              f"separation={sep:+.3f}")


# ── Model ──────────────────────────────────────────────────────────────────────
base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False

x   = base.output
x   = layers.GlobalAveragePooling2D(name="gap")(x)
x   = layers.Dropout(0.3, name="dropout")(x)
out = layers.Dense(1, activation="sigmoid", name="output")(x)

model = Model(base.input, out)

print(f"\n── Model: EfficientNetB0 ({IMG_SIZE}×{IMG_SIZE}) ──")
print(f"  Total params:          {model.count_params():>12,}")
trainable_warmup = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"  Trainable (warm-up):   {trainable_warmup:>12,}")


# ── Phase 1: warm-up ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Phase 1: Warm-up — Dense head only ({EPOCHS_WARMUP} epochs)")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

model.fit(
    train_ds,
    epochs=EPOCHS_WARMUP,
    validation_data=val_ds,
)


# ── Phase 2: fine-tune ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Phase 2: Fine-tune — top {UNFREEZE_N} EfficientNet layers")
print("=" * 60)

base.trainable = True
for layer in base.layers[:-UNFREEZE_N]:
    layer.trainable = False

# CRITICAL: keep all BatchNormalization in inference mode.
# Otherwise BN re-computes statistics on small training batches, which
# destroys ImageNet pretrained features and causes oscillating val_auc.
# This is the documented EfficientNet fine-tuning trap.
n_bn_frozen = 0
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
        n_bn_frozen += 1
print(f"  BN layers kept in inference mode: {n_bn_frozen}")

trainable_ft = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"  Trainable (fine-tune): {trainable_ft:>12,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

model.fit(
    train_ds,
    epochs=EPOCHS_FT,
    validation_data=val_ds,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_auc", mode="max",
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        OutputDistribution(val_ds, val_labels),
    ],
)


# ── Video-level threshold optimisation on validation ──────────────────────────
print("\n" + "=" * 60)
print("Video-level threshold optimisation (validation set)")
print("=" * 60)

best_model = tf.keras.models.load_model(MODEL_PATH)

vid_probs, vid_labels = [], []
for vid_id, info in sorted(val_videos.items()):
    paths = sorted(info["paths"])
    if len(paths) > FRAMES_PER_VIDEO:
        idx = np.linspace(0, len(paths) - 1, FRAMES_PER_VIDEO, dtype=int)
        paths = [paths[i] for i in idx]

    frames = [load_frame_numpy(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        continue

    probs = best_model.predict(
        np.array(frames), batch_size=BATCH_SIZE, verbose=0
    ).flatten()
    vid_probs.append(float(np.median(probs)))
    vid_labels.append(info["label"])

vid_probs  = np.array(vid_probs)
vid_labels = np.array(vid_labels)

print(f"Evaluated {len(vid_labels)} val videos "
      f"(real: {np.sum(vid_labels==0)}, fake: {np.sum(vid_labels==1)})")

# Search threshold that maximises balanced accuracy.
# F1 is degenerate on imbalanced data — it can reward "predict all positive"
# (recall=1, precision=2/3 → F1=0.80) which has zero discriminative power.
# Balanced accuracy = mean(TPR, TNR), so a degenerate model scores 0.5.
best_thr, best_bal = 0.5, 0.0
for thr in np.arange(0.20, 0.80, 0.01):
    preds = (vid_probs >= thr).astype(int)
    bal = balanced_accuracy_score(vid_labels, preds)
    if bal > best_bal:
        best_bal, best_thr = bal, round(float(thr), 2)

print(f"Optimal threshold: {best_thr:.2f}  (val balanced acc = {best_bal:.4f})")


# ── Save config ───────────────────────────────────────────────────────────────
config = {
    "threshold":        best_thr,
    "img_size":         IMG_SIZE,
    "backbone":         "EfficientNetB0",
    "model_path":       MODEL_PATH,
    "aggregation":      "median",
    "frames_per_video": FRAMES_PER_VIDEO,
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved → {CONFIG_PATH}")


# ── Final validation metrics (video-level) ────────────────────────────────────
y_pred = (vid_probs >= best_thr).astype(int)

val_auc     = roc_auc_score(vid_labels, vid_probs)
val_f1      = f1_score(vid_labels, y_pred, zero_division=0)
val_prec    = precision_score(vid_labels, y_pred, zero_division=0)
val_rec     = recall_score(vid_labels, y_pred, zero_division=0)
val_acc     = accuracy_score(vid_labels, y_pred)
val_bal_acc = balanced_accuracy_score(vid_labels, y_pred)

cm = confusion_matrix(vid_labels, y_pred)

print(f"\n{'='*60}")
print("Validation Results (video-level, median aggregation)")
print(f"{'='*60}")
print(f"  Threshold:        {best_thr:.2f}")
print(f"  ROC-AUC:          {val_auc:.4f}")
print(f"  F1-score:         {val_f1:.4f}")
print(f"  Balanced Acc:     {val_bal_acc:.4f}")
print(f"  Precision:        {val_prec:.4f}")
print(f"  Recall:           {val_rec:.4f}")
print(f"  Accuracy:         {val_acc:.4f}")
print(f"\n  Confusion Matrix (real=0, fake=1):")
print(f"                  Predicted Real  Predicted Fake")
print(f"  Actual Real       {cm[0][0]:>5}           {cm[0][1]:>5}")
print(f"  Actual Fake       {cm[1][0]:>5}           {cm[1][1]:>5}")
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

print(f"\n  Model   → {MODEL_PATH}")
print(f"  Config  → {CONFIG_PATH}")
print(f"  Test IDs → {TEST_SPLIT_PATH}")
print(f"\nNext: python test_combined.py")
