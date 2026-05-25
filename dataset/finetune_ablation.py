"""
finetune_ablation.py
====================
Re-train EfficientNet-B0 with one methodology choice disabled, for the
ablation study reported in section 3.3.4 of the diploma document.

Variants
--------
- production       full method (sanity check; should reproduce ~0.898 ROC-AUC)
- no_weighting     class_weight only (no per-sample weighting across cells)
- bn_trainable     BatchNormalization left in training mode during Phase 2
- asymmetric_aug   heavier augmentation on real frames than on fake frames

All artefacts are written under dataset/ablations/{variant}/ so the
production checkpoint, threshold, and test split are not overwritten.
The held-out test split is reused from the existing test_split.csv at
the repository root so all variants are evaluated on the same videos.

Run from dataset/:
    python finetune_ablation.py --variant no_weighting
    python finetune_ablation.py --variant bn_trainable
    python finetune_ablation.py --variant asymmetric_aug
"""

import os
import glob
import json
import argparse
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


# ── Variant selection ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    required=True,
    choices=["production", "no_weighting", "bn_trainable", "asymmetric_aug"],
    help="Which methodology choice to disable",
)
args = parser.parse_args()
VARIANT = args.variant


# ── Config ─────────────────────────────────────────────────────────────────────
FF_FACES_DIR      = "faces_dataset"
CELEBDF_FACES_DIR = "celebdf_faces"
ABLATION_DIR      = os.path.join("ablations", VARIANT)
MODEL_PATH        = os.path.join(ABLATION_DIR, "model.keras")
CONFIG_PATH       = os.path.join(ABLATION_DIR, "model_config.json")
LOG_PATH          = os.path.join(ABLATION_DIR, "train_log.json")

# Reuse the production test split so all variants are evaluated on identical videos.
PRODUCTION_TEST_SPLIT = "test_split.csv"

IMG_SIZE         = 224
BATCH_SIZE       = 32
UNFREEZE_N       = 80
LR_WARMUP        = 1e-3
LR_FINETUNE      = 5e-5
EPOCHS_WARMUP    = 1
EPOCHS_FT        = 15      # production EarlyStopping fired at epoch 15 — match it
FRAMES_PER_VIDEO = 10
RANDOM_SEED      = 42

os.makedirs(ABLATION_DIR, exist_ok=True)

print("=" * 70)
print(f"Ablation variant: {VARIANT}")
print(f"Artefacts → {ABLATION_DIR}/")
print("=" * 70)


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_video_id(filepath: str) -> str:
    return "_".join(os.path.basename(filepath).split("_")[:-1])


def collect_frames(faces_dir: str, prefix: str):
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
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32")


# ── Collect all frames ─────────────────────────────────────────────────────────
ff_real, ff_fake = collect_frames(FF_FACES_DIR, prefix="ff__")
cdf_real, cdf_fake = collect_frames(CELEBDF_FACES_DIR, prefix="cdf__")
all_real = ff_real + cdf_real
all_fake = ff_fake + cdf_fake

print(f"FF++     — real: {len(ff_real):>5}, fake: {len(ff_fake):>5}")
print(f"Celeb-DF — real: {len(cdf_real):>5}, fake: {len(cdf_fake):>5}")


# ── Reuse production test split, regenerate train/val deterministically ───────
# The production run used random_state=42 with the same 70/15/15 split on the
# same real_vids and fake_vids lists. Reproducing that split here gives the
# same train/val partitions, and the test partition is verified against
# test_split.csv to guard against drift.
real_vids = sorted(set(vid for _, vid in all_real))
fake_vids = sorted(set(vid for _, vid in all_fake))

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

# Sanity check: regenerated test set must match the production test_split.csv
prod_test_df = pd.read_csv(PRODUCTION_TEST_SPLIT)
prod_test_ids = set(prod_test_df["video_id"].tolist())
if prod_test_ids != test_set:
    raise RuntimeError(
        f"Regenerated test set differs from {PRODUCTION_TEST_SPLIT}. "
        f"Cannot proceed — ablation evaluation would not be comparable to production."
    )
print(f"\n✓ Regenerated test split matches {PRODUCTION_TEST_SPLIT} "
      f"({len(test_set)} videos)")


def partition(frames, label):
    tr, va = [], []
    for path, vid in frames:
        if   vid in train_set: tr.append((path, vid, label))
        elif vid in val_set:   va.append((path, vid, label))
    return tr, va


r_tr, r_va = partition(all_real, 0)
f_tr, f_va = partition(all_fake, 1)

train_all = r_tr + f_tr
val_all   = r_va + f_va

print(f"Train — real: {len(r_tr):>5}, fake: {len(f_tr):>5}, total: {len(train_all)}")
print(f"Val   — real: {len(r_va):>5}, fake: {len(f_va):>5}, total: {len(val_all)}")


# ── Sample weights / class_weight per variant ─────────────────────────────────
train_paths   = [p for p, _, _ in train_all]
train_labels  = [l for _, _, l in train_all]
train_sources = ["ff" if v.startswith("ff__") else "cdf"
                 for _, v, _ in train_all]

cell_counts = defaultdict(int)
for src, lbl in zip(train_sources, train_labels):
    cell_counts[(src, lbl)] += 1

n_total = len(train_all)
n_cells = len(cell_counts)

class_weight_arg = None

if VARIANT == "no_weighting":
    # Disable per-sample weighting across (source, label) cells.
    # Fall back to plain class_weight (only balances real/fake), which is the
    # baseline this scheme is designed to replace.
    n_real = sum(1 for l in train_labels if l == 0)
    n_fake = sum(1 for l in train_labels if l == 1)
    class_weight_arg = {
        0: n_total / (2 * n_real),
        1: n_total / (2 * n_fake),
    }
    train_weights = [1.0] * len(train_all)
    print(f"\n── Variant: no_weighting (class_weight only) ──")
    print(f"  class_weight: real={class_weight_arg[0]:.4f}, "
          f"fake={class_weight_arg[1]:.4f}")
else:
    # Production scheme: per-sample weights across all four (source, label) cells.
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


# ── Group validation frames by video ──────────────────────────────────────────
val_videos = defaultdict(lambda: {"paths": [], "label": None})
for path, vid, label in val_all:
    val_videos[vid]["paths"].append(path)
    val_videos[vid]["label"] = label


# ── TF Dataset ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)
perm = rng.permutation(len(train_paths))
train_paths   = [train_paths[i]   for i in perm]
train_labels  = [train_labels[i]  for i in perm]
train_weights = [train_weights[i] for i in perm]

val_paths  = [p for p, _, _ in val_all]
val_labels = [l for _, _, l in val_all]


# Dataset is built with or without per-sample weights depending on the variant.
# Keras 3 forbids combining class_weight with sample_weight, so for the
# no_weighting variant we drop the weight tuple element entirely and let
# class_weight do the balancing instead.
USE_SAMPLE_WEIGHTS = (class_weight_arg is None)


def load_image_train_w(path: tf.Tensor, label: tf.Tensor, weight: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.uint8)
    return img, tf.cast(label, tf.float32), tf.cast(weight, tf.float32)


def load_image_train_nw(path: tf.Tensor, label: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.uint8)
    return img, tf.cast(label, tf.float32)


def load_image_val(path: tf.Tensor, label: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.uint8)
    return img, tf.cast(label, tf.float32)


def _augment_core(img, is_real_scalar):
    """Core augmentation logic — applied identically by both signatures."""
    if VARIANT == "asymmetric_aug":
        def heavy():
            x = tf.image.random_jpeg_quality(img, min_jpeg_quality=30, max_jpeg_quality=70)
            x = tf.cast(x, tf.float32)
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, max_delta=60.0)
            x = tf.image.random_contrast(x, lower=0.70, upper=1.30)
            x = tf.image.random_saturation(x, lower=0.7, upper=1.3)
            x = tf.image.random_hue(x, max_delta=0.10)
            x = tf.clip_by_value(x, 0.0, 255.0)
            return x

        def light():
            x = tf.image.random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, 0.0, 255.0)
            return x

        return tf.cond(is_real_scalar, heavy, light)
    else:
        # Symmetric augmentation (production policy).
        x = tf.image.random_jpeg_quality(img, min_jpeg_quality=70, max_jpeg_quality=100)
        x = tf.cast(x, tf.float32)
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=25.0)
        x = tf.image.random_contrast(x, lower=0.90, upper=1.10)
        x = tf.clip_by_value(x, 0.0, 255.0)
        return x


def augment_w(img, label, weight):
    out = _augment_core(img, tf.equal(label, 0.0))
    return out, label, weight


def augment_nw(img, label):
    out = _augment_core(img, tf.equal(label, 0.0))
    return out, label


def preprocess_val(img: tf.Tensor, label: tf.Tensor):
    img = tf.cast(img, tf.float32)
    return img, label


if USE_SAMPLE_WEIGHTS:
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels, train_weights))
        .map(load_image_train_w, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(4096, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .map(augment_w, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
else:
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .map(load_image_train_nw, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(4096, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .map(augment_nw, num_parallel_calls=tf.data.AUTOTUNE)
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


# ── Diagnostic callback ───────────────────────────────────────────────────────
class OutputDistribution(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, val_labels):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = np.array(val_labels)
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.val_ds, verbose=0).flatten()
        real_p = probs[self.val_labels == 0]
        fake_p = probs[self.val_labels == 1]
        sep = float(np.mean(fake_p) - np.mean(real_p))
        entry = {
            "epoch": int(epoch + 1),
            "real_mean": float(np.mean(real_p)),
            "fake_mean": float(np.mean(fake_p)),
            "separation": sep,
            "val_auc": float(logs.get("val_auc", 0.0)) if logs else 0.0,
            "val_loss": float(logs.get("val_loss", 0.0)) if logs else 0.0,
            "val_accuracy": float(logs.get("val_accuracy", 0.0)) if logs else 0.0,
        }
        self.history.append(entry)
        print(f"  [diag] real μ={np.mean(real_p):.3f} | fake μ={np.mean(fake_p):.3f} | "
              f"sep={sep:+.3f}")


diag_callback = OutputDistribution(val_ds, val_labels)


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


# ── Phase 1: warm-up ──────────────────────────────────────────────────────────
print(f"\nPhase 1: warm-up ({EPOCHS_WARMUP} epoch)")
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

fit_kwargs_p1 = dict(epochs=EPOCHS_WARMUP, validation_data=val_ds)
if class_weight_arg is not None:
    # When using class_weight, the dataset still yields 3-tuples (img, lbl, w=1.0).
    # Keras applies class_weight on top of the per-sample weight.
    fit_kwargs_p1["class_weight"] = class_weight_arg

model.fit(train_ds, **fit_kwargs_p1)


# ── Phase 2: fine-tune ────────────────────────────────────────────────────────
print(f"\nPhase 2: fine-tune (up to {EPOCHS_FT} epochs)")
base.trainable = True
for layer in base.layers[:-UNFREEZE_N]:
    layer.trainable = False

# Variant-specific BN handling.
n_bn_frozen = 0
n_bn_trainable = 0
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        if VARIANT == "bn_trainable":
            # Ablation: leave BN trainable, allowing it to re-estimate
            # running statistics from small training batches.
            # Only flip those whose surrounding block is unfrozen.
            if layer.trainable:
                n_bn_trainable += 1
        else:
            layer.trainable = False
            n_bn_frozen += 1

if VARIANT == "bn_trainable":
    print(f"  ── Variant: bn_trainable — BN layers in unfrozen blocks remain "
          f"in training mode ({n_bn_trainable} trainable BN layers) ──")
else:
    print(f"  BN layers kept in inference mode: {n_bn_frozen}")


model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

fit_kwargs_p2 = dict(
    epochs=EPOCHS_FT,
    validation_data=val_ds,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_auc", mode="max",
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        diag_callback,
    ],
)
if class_weight_arg is not None:
    fit_kwargs_p2["class_weight"] = class_weight_arg

model.fit(train_ds, **fit_kwargs_p2)


# ── Video-level threshold tuning on validation ────────────────────────────────
print("\nValidation video-level threshold tuning …")
best_model = tf.keras.models.load_model(MODEL_PATH)

vid_probs, vid_labels_arr = [], []
for vid_id, info in sorted(val_videos.items()):
    paths = sorted(info["paths"])
    if len(paths) > FRAMES_PER_VIDEO:
        idx = np.linspace(0, len(paths) - 1, FRAMES_PER_VIDEO, dtype=int)
        paths = [paths[i] for i in idx]
    frames = [load_frame_numpy(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        continue
    probs = best_model.predict(np.array(frames), batch_size=BATCH_SIZE, verbose=0).flatten()
    vid_probs.append(float(np.median(probs)))
    vid_labels_arr.append(info["label"])

vid_probs  = np.array(vid_probs)
vid_labels_arr = np.array(vid_labels_arr)

best_thr, best_bal = 0.5, 0.0
for thr in np.arange(0.20, 0.80, 0.01):
    preds = (vid_probs >= thr).astype(int)
    bal = balanced_accuracy_score(vid_labels_arr, preds)
    if bal > best_bal:
        best_bal, best_thr = bal, round(float(thr), 2)

print(f"Optimal threshold: {best_thr:.2f}  (val BalAcc = {best_bal:.4f})")


# ── Save config + log ─────────────────────────────────────────────────────────
config = {
    "variant":          VARIANT,
    "threshold":        best_thr,
    "img_size":         IMG_SIZE,
    "backbone":         "EfficientNetB0",
    "model_path":       MODEL_PATH,
    "aggregation":      "median",
    "frames_per_video": FRAMES_PER_VIDEO,
    "val_bal_acc":      best_bal,
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

with open(LOG_PATH, "w") as f:
    json.dump({"variant": VARIANT, "history": diag_callback.history}, f, indent=2)

print(f"\n✓ Done.")
print(f"  Model  → {MODEL_PATH}")
print(f"  Config → {CONFIG_PATH}")
print(f"  Log    → {LOG_PATH}")
print(f"\nNext: python test_ablation.py --variant {VARIANT}")
