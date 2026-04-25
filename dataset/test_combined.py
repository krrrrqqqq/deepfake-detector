"""
test_combined.py
================
Evaluate the fine-tuned model on the held-out test split.

Reads model_config.json (produced by finetune_combined.py) for:
  - threshold   — optimal decision boundary (tuned on validation F1)
  - img_size    — input resolution the model expects
  - aggregation — how per-frame probabilities are combined (default: median)

For each test video the script:
  1. Loads all available face-crop frames.
  2. Uniformly subsamples to FRAMES_PER_VIDEO if there are more.
  3. Runs per-frame inference.
  4. Aggregates per-frame probabilities to a single video-level score.
  5. Applies the threshold to produce REAL / FAKE prediction.

Prints: ROC-AUC, F1, balanced accuracy, precision, recall, accuracy,
confusion matrix, and a comparison table of aggregation methods.

Run from dataset/:
    python test_combined.py
"""

import os
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
    average_precision_score,
)

# ── Config (defaults — overridden by model_config.json if present) ─────────────
FF_FACES_DIR      = "faces_dataset"
CELEBDF_FACES_DIR = "celebdf_faces"
TEST_SPLIT_PATH   = "test_split.csv"
CONFIG_PATH       = "model_config.json"

# Fallback defaults if config is missing
MODEL_PATH       = "efficientnet_combined.keras"
IMG_SIZE         = 224
FRAMES_PER_VIDEO = 10
THRESHOLD        = 0.5
AGGREGATION      = "median"
BATCH_SIZE       = 32


# ── Load config ───────────────────────────────────────────────────────────────
if os.path.isfile(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    MODEL_PATH       = cfg.get("model_path",       MODEL_PATH)
    IMG_SIZE         = cfg.get("img_size",          IMG_SIZE)
    FRAMES_PER_VIDEO = cfg.get("frames_per_video",  FRAMES_PER_VIDEO)
    THRESHOLD        = cfg.get("threshold",         THRESHOLD)
    AGGREGATION      = cfg.get("aggregation",       AGGREGATION)
    print(f"Loaded config from {CONFIG_PATH}")
    print(f"  model:       {MODEL_PATH}")
    print(f"  img_size:    {IMG_SIZE}")
    print(f"  threshold:   {THRESHOLD}")
    print(f"  aggregation: {AGGREGATION}")
    print(f"  backbone:    {cfg.get('backbone', '?')}")
else:
    print(f"WARNING: {CONFIG_PATH} not found — using hardcoded defaults")
    print(f"  Run finetune_combined.py first to generate it.")


# ── Load model ─────────────────────────────────────────────────────────────────
print(f"\nLoading model from {MODEL_PATH} …")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded.  Input shape: {model.input_shape}")


# ── Load test split ────────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_SPLIT_PATH)
test_videos = {row["video_id"]: int(row["label"]) for _, row in test_df.iterrows()}
n_test_real = sum(1 for l in test_videos.values() if l == 0)
n_test_fake = sum(1 for l in test_videos.values() if l == 1)

print(f"\nTest split: {len(test_df)} videos "
      f"(real: {n_test_real}, fake: {n_test_fake})")

# Label audit
print("\n── Label audit (test_split.csv) ──")
print(f"  label=0 (real): {n_test_real} videos")
print(f"  label=1 (fake): {n_test_fake} videos")
print(f"  Convention: real=0, fake=1  ✓")


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_video_id(filepath: str) -> str:
    return "_".join(os.path.basename(filepath).split("_")[:-1])


def load_frame(path: str):
    """Load face image: BGR→RGB, resize, return float32 in [0, 255].

    EfficientNetB0 has include_preprocessing=True (default) — it does ImageNet
    normalisation internally. Pre-normalising here would double-normalise
    and produce wrong predictions.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32")


def sample_frames_uniform(paths: list[str], max_frames: int) -> list[str]:
    """
    Uniformly subsample frame paths if there are more than max_frames.
    Face-extraction scripts already sample 10 uniform frames per video,
    so this is a safety net — not the primary temporal sampling.
    """
    paths = sorted(paths)
    if len(paths) <= max_frames:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
    return [paths[i] for i in indices]


def aggregate_probs(probs: np.ndarray, method: str = "median") -> float:
    """Aggregate per-frame probabilities into a single video score."""
    if method == "median":
        return float(np.median(probs))
    if method == "mean":
        return float(np.mean(probs))
    if method == "top3_mean":
        k = min(3, len(probs))
        return float(np.mean(np.sort(probs)[-k:]))
    if method == "max":
        return float(np.max(probs))
    raise ValueError(f"Unknown aggregation: {method}")


# ── Index face images ──────────────────────────────────────────────────────────
def index_faces(faces_dir: str, prefix: str) -> dict[str, list[str]]:
    result = defaultdict(list)
    for label_name in ["real", "fake"]:
        folder = os.path.join(faces_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fp in sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                         glob.glob(os.path.join(folder, "*.png"))):
            vid_id = prefix + get_video_id(fp)
            result[vid_id].append(fp)
    return result


print("\nIndexing face images …")
all_index = {**index_faces(FF_FACES_DIR, "ff__"),
             **index_faces(CELEBDF_FACES_DIR, "cdf__")}
print(f"Indexed {len(all_index)} videos total")


# ── Run inference on test videos ──────────────────────────────────────────────
# Store per-frame probs so we can compare aggregation methods afterwards
video_frame_probs = {}   # vid_id → np.array of per-frame probs
video_labels_map  = {}   # vid_id → int label
missing = 0

for vid_id, label in sorted(test_videos.items()):
    if vid_id not in all_index:
        missing += 1
        continue

    paths = sample_frames_uniform(all_index[vid_id], FRAMES_PER_VIDEO)
    frames = [load_frame(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        missing += 1
        continue

    arr   = np.array(frames, dtype="float32")
    probs = model.predict(arr, batch_size=BATCH_SIZE, verbose=0).flatten()

    video_frame_probs[vid_id] = probs
    video_labels_map[vid_id]  = label

if missing:
    print(f"WARNING: {missing} test videos had no frames or were not found")

vid_ids    = sorted(video_frame_probs.keys())
vid_labels = np.array([video_labels_map[v] for v in vid_ids])

print(f"\nEvaluated {len(vid_ids)} test videos "
      f"(real: {np.sum(vid_labels==0)}, fake: {np.sum(vid_labels==1)})")
print(f"Sampled frames per video: {FRAMES_PER_VIDEO}")


# ── Aggregation method comparison ─────────────────────────────────────────────
methods = ["mean", "median", "top3_mean", "max"]

print(f"\n{'='*70}")
print("Aggregation method comparison (threshold from config)")
print(f"{'='*70}")
print(f"  {'Method':<12} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} "
      f"{'BalAcc':>8} {'Prec':>8} {'Recall':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for method in methods:
    scores = np.array([aggregate_probs(video_frame_probs[v], method) for v in vid_ids])
    preds  = (scores >= THRESHOLD).astype(int)

    auc_roc = roc_auc_score(vid_labels, scores)
    auc_pr  = average_precision_score(vid_labels, scores)
    f1      = f1_score(vid_labels, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(vid_labels, preds)
    prec    = precision_score(vid_labels, preds, zero_division=0)
    rec     = recall_score(vid_labels, preds, zero_division=0)

    marker = "  ◄" if method == AGGREGATION else ""
    print(f"  {method:<12} {auc_roc:>8.4f} {auc_pr:>8.4f} {f1:>8.4f} "
          f"{bal_acc:>8.4f} {prec:>8.4f} {rec:>8.4f}{marker}")


# ── Primary metrics (using configured aggregation + threshold) ────────────────
vid_scores = np.array(
    [aggregate_probs(video_frame_probs[v], AGGREGATION) for v in vid_ids])
y_pred = (vid_scores >= THRESHOLD).astype(int)

roc_auc = roc_auc_score(vid_labels, vid_scores)
pr_auc  = average_precision_score(vid_labels, vid_scores)
f1_val  = f1_score(vid_labels, y_pred, zero_division=0)
prec_v  = precision_score(vid_labels, y_pred, zero_division=0)
rec_v   = recall_score(vid_labels, y_pred, zero_division=0)
acc_v   = accuracy_score(vid_labels, y_pred)
bal_v   = balanced_accuracy_score(vid_labels, y_pred)

cm = confusion_matrix(vid_labels, y_pred)

print(f"\n{'='*60}")
print(f"Test Results ({AGGREGATION} aggregation, threshold={THRESHOLD:.2f})")
print(f"{'='*60}")
print(f"  ROC-AUC:          {roc_auc:.4f}")
print(f"  PR-AUC:           {pr_auc:.4f}")
print(f"  F1-score:         {f1_val:.4f}")
print(f"  Balanced Acc:     {bal_v:.4f}")
print(f"  Precision:        {prec_v:.4f}")
print(f"  Recall:           {rec_v:.4f}")
print(f"  Accuracy:         {acc_v:.4f}  (secondary — dataset is imbalanced)")

print(f"\n  Confusion Matrix (real=0, fake=1):")
print(f"                  Predicted Real  Predicted Fake")
print(f"  Actual Real       {cm[0][0]:>5}           {cm[0][1]:>5}")
print(f"  Actual Fake       {cm[1][0]:>5}           {cm[1][1]:>5}")
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# ── Per-frame probability distribution ────────────────────────────────────────
all_frame_probs_real = np.concatenate(
    [video_frame_probs[v] for v in vid_ids if video_labels_map[v] == 0])
all_frame_probs_fake = np.concatenate(
    [video_frame_probs[v] for v in vid_ids if video_labels_map[v] == 1])

print(f"\n── Per-frame probability distribution ──")
print(f"  Real frames: mean={np.mean(all_frame_probs_real):.4f}, "
      f"median={np.median(all_frame_probs_real):.4f}, "
      f"std={np.std(all_frame_probs_real):.4f}")
print(f"  Fake frames: mean={np.mean(all_frame_probs_fake):.4f}, "
      f"median={np.median(all_frame_probs_fake):.4f}, "
      f"std={np.std(all_frame_probs_fake):.4f}")
print(f"  Separation:  {np.mean(all_frame_probs_fake) - np.mean(all_frame_probs_real):.4f} "
      f"(fake mean - real mean; >0 = correct direction)")


# ── Per-source breakdown (FF++ vs Celeb-DF) ───────────────────────────────────
# Overall recall hides per-source performance. After fixing real-vs-fake
# imbalance by adding Celeb-DF real videos, a within-real imbalance can
# appear (FF++ real is ~3× smaller than Celeb-DF real). This block splits
# the test set by video-id prefix to expose per-source recall.
print(f"\n{'='*60}")
print("Per-source breakdown")
print(f"{'='*60}")

for prefix, name in [("ff__", "FaceForensics++"), ("cdf__", "Celeb-DF v2")]:
    src_ids = [v for v in vid_ids if v.startswith(prefix)]
    if not src_ids:
        continue

    src_scores = np.array([
        aggregate_probs(video_frame_probs[v], AGGREGATION) for v in src_ids
    ])
    src_labels = np.array([video_labels_map[v] for v in src_ids])
    src_preds  = (src_scores >= THRESHOLD).astype(int)

    n_real = int(np.sum(src_labels == 0))
    n_fake = int(np.sum(src_labels == 1))

    print(f"\n  {name} — {len(src_ids)} videos ({n_real} real, {n_fake} fake)")

    if n_real == 0 or n_fake == 0:
        acc = float(np.mean(src_preds == src_labels))
        print(f"    Accuracy: {acc:.3f}  (single-class subset)")
        continue

    cm_src = confusion_matrix(src_labels, src_preds, labels=[0, 1])
    tn, fp, fn, tp = cm_src.ravel()
    tnr    = tn / max(tn + fp, 1)
    tpr    = tp / max(tp + fn, 1)
    balacc = (tnr + tpr) / 2

    print(f"    TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"    Real recall (TNR): {tnr:.3f}  "
          f"Fake recall (TPR): {tpr:.3f}  "
          f"Balanced Acc: {balacc:.3f}")

print(f"\n── Per-frame real-prob distribution by source ──")
for prefix, name in [("ff__", "FF++"), ("cdf__", "Celeb-DF")]:
    real_src_vids = [v for v in vid_ids
                     if v.startswith(prefix) and video_labels_map[v] == 0]
    if not real_src_vids:
        continue
    probs = np.concatenate([video_frame_probs[v] for v in real_src_vids])
    print(f"  {name:<10} real (N={probs.size:>4}): "
          f"mean={np.mean(probs):.4f}, "
          f"median={np.median(probs):.4f}, "
          f"std={np.std(probs):.4f}")
