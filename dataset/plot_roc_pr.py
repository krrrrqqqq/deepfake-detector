"""
plot_roc_pr.py
==============
Generate ROC and Precision-Recall curves for the production
EfficientNet-B0 detector on the held-out test split.

Reads:
  - efficientnet_combined.keras
  - test_split.csv
  - model_config.json (for img_size, threshold, frames_per_video)
  - faces_dataset/ and celebdf_faces/

Writes:
  - figure_10_roc_pr.png
  - figure_10_roc_pr.svg

Run from dataset/:
    python plot_roc_pr.py
"""

import os
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
)


FF_FACES_DIR      = "faces_dataset"
CELEBDF_FACES_DIR = "celebdf_faces"
TEST_SPLIT_PATH   = "test_split.csv"
CONFIG_PATH       = "model_config.json"
BATCH_SIZE        = 32


# ── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

MODEL_PATH       = cfg["model_path"]
IMG_SIZE         = cfg["img_size"]
FRAMES_PER_VIDEO = cfg["frames_per_video"]
THRESHOLD        = cfg["threshold"]


# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)


# ── Load test split ───────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_SPLIT_PATH)
test_videos = {row["video_id"]: int(row["label"]) for _, row in test_df.iterrows()}
print(f"Test videos: {len(test_videos)}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_video_id(filepath: str) -> str:
    return "_".join(os.path.basename(filepath).split("_")[:-1])


def load_frame(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32")


def sample_uniform(paths, max_n):
    paths = sorted(paths)
    if len(paths) <= max_n:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_n, dtype=int)
    return [paths[i] for i in idx]


def index_faces(faces_dir, prefix):
    result = defaultdict(list)
    for label_name in ["real", "fake"]:
        folder = os.path.join(faces_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fp in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            vid_id = prefix + get_video_id(fp)
            result[vid_id].append(fp)
    return result


print("Indexing face images ...")
all_index = {**index_faces(FF_FACES_DIR, "ff__"),
             **index_faces(CELEBDF_FACES_DIR, "cdf__")}


# ── Inference ─────────────────────────────────────────────────────────────────
print("Running inference on test set ...")
vid_scores = []
vid_labels = []

for vid_id, label in sorted(test_videos.items()):
    if vid_id not in all_index:
        continue
    paths = sample_uniform(all_index[vid_id], FRAMES_PER_VIDEO)
    frames = [load_frame(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        continue
    arr = np.array(frames, dtype="float32")
    probs = model.predict(arr, batch_size=BATCH_SIZE, verbose=0).flatten()
    vid_scores.append(float(np.median(probs)))
    vid_labels.append(label)

vid_scores = np.array(vid_scores)
vid_labels = np.array(vid_labels)
print(f"Evaluated {len(vid_labels)} videos "
      f"(real: {np.sum(vid_labels == 0)}, fake: {np.sum(vid_labels == 1)})")


# ── Compute ROC and PR ────────────────────────────────────────────────────────
fpr, tpr, roc_thresholds = roc_curve(vid_labels, vid_scores)
roc_auc = roc_auc_score(vid_labels, vid_scores)

precision, recall, pr_thresholds = precision_recall_curve(vid_labels, vid_scores)
pr_auc = average_precision_score(vid_labels, vid_scores)

# Operating point at τ from config
y_pred = (vid_scores >= THRESHOLD).astype(int)
tp = int(np.sum((y_pred == 1) & (vid_labels == 1)))
fp = int(np.sum((y_pred == 1) & (vid_labels == 0)))
fn = int(np.sum((y_pred == 0) & (vid_labels == 1)))
tn = int(np.sum((y_pred == 0) & (vid_labels == 0)))
op_tpr = tp / max(tp + fn, 1)
op_fpr = fp / max(fp + tn, 1)
op_prec = tp / max(tp + fp, 1)
op_rec = op_tpr

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print(f"Operating point at tau={THRESHOLD}: TPR={op_tpr:.3f}, "
      f"FPR={op_fpr:.3f}, Precision={op_prec:.3f}")


# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- ROC ---
ax = axes[0]
ax.plot(fpr, tpr, color="#1f4e79", linewidth=2.0,
        label=f"ROC curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.0,
        label="Random classifier")
ax.scatter([op_fpr], [op_tpr], color="#c0392b", s=70, zorder=5,
           label=f"Operating point (τ = {THRESHOLD})")
ax.annotate(f"  TPR = {op_tpr:.3f}\n  FPR = {op_fpr:.3f}",
            xy=(op_fpr, op_tpr), xytext=(op_fpr + 0.05, op_tpr - 0.10),
            fontsize=9, color="#c0392b")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.set_xlabel("False Positive Rate (1 − TNR)")
ax.set_ylabel("True Positive Rate (fake recall)")
ax.set_title("(a) ROC curve")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

# --- PR ---
ax = axes[1]
ax.plot(recall, precision, color="#1f4e79", linewidth=2.0,
        label=f"PR curve (AP = {pr_auc:.3f})")
# Baseline = prevalence
prevalence = float(np.mean(vid_labels))
ax.axhline(prevalence, color="grey", linestyle="--", linewidth=1.0,
           label=f"Random classifier (prevalence = {prevalence:.3f})")
ax.scatter([op_rec], [op_prec], color="#c0392b", s=70, zorder=5,
           label=f"Operating point (τ = {THRESHOLD})")
ax.annotate(f"  P = {op_prec:.3f}\n  R = {op_rec:.3f}",
            xy=(op_rec, op_prec), xytext=(op_rec - 0.35, op_prec - 0.10),
            fontsize=9, color="#c0392b")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.set_xlabel("Recall (fake recall)")
ax.set_ylabel("Precision")
ax.set_title("(b) Precision–Recall curve")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)

fig.suptitle("Figure 10 — Test-set ROC and Precision–Recall curves "
             f"(N = {len(vid_labels)} videos)",
             fontsize=12, y=1.02)
fig.tight_layout()

out_png = "figure_10_roc_pr.png"
out_svg = "figure_10_roc_pr.svg"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
print(f"\nSaved → {out_png}")
print(f"Saved → {out_svg}")
