"""
test_ablation.py
================
Evaluate an ablation checkpoint on the held-out test set, using the same
test split and same evaluation procedure as test_combined.py.

Produces a single-line JSON record per variant for direct insertion into
the §3.3.4 ablation table:
    ROC-AUC, BalAcc, FF++ real recall, Celeb-DF fake recall.

Run from dataset/:
    python test_ablation.py --variant no_weighting
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
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
    f1_score, precision_score, recall_score, average_precision_score,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    required=True,
    choices=["production", "no_weighting", "bn_trainable", "asymmetric_aug"],
)
args = parser.parse_args()
VARIANT = args.variant

ABLATION_DIR = os.path.join("ablations", VARIANT)
CONFIG_PATH  = os.path.join(ABLATION_DIR, "model_config.json")
RESULT_PATH  = os.path.join(ABLATION_DIR, "test_result.json")

FF_FACES_DIR      = "faces_dataset"
CELEBDF_FACES_DIR = "celebdf_faces"
TEST_SPLIT_PATH   = "test_split.csv"
BATCH_SIZE        = 32


# ── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

MODEL_PATH       = cfg["model_path"]
IMG_SIZE         = cfg["img_size"]
FRAMES_PER_VIDEO = cfg["frames_per_video"]
THRESHOLD        = cfg["threshold"]

print(f"Evaluating variant: {VARIANT}")
print(f"  model:     {MODEL_PATH}")
print(f"  threshold: {THRESHOLD}")


# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)


# ── Load test split ───────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_SPLIT_PATH)
test_videos = {row["video_id"]: int(row["label"]) for _, row in test_df.iterrows()}


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


def sample_frames_uniform(paths, max_frames):
    paths = sorted(paths)
    if len(paths) <= max_frames:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
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


all_index = {**index_faces(FF_FACES_DIR, "ff__"),
             **index_faces(CELEBDF_FACES_DIR, "cdf__")}


# ── Inference ─────────────────────────────────────────────────────────────────
video_frame_probs = {}
video_labels_map = {}

for vid_id, label in sorted(test_videos.items()):
    if vid_id not in all_index:
        continue
    paths = sample_frames_uniform(all_index[vid_id], FRAMES_PER_VIDEO)
    frames = [load_frame(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        continue
    arr = np.array(frames, dtype="float32")
    probs = model.predict(arr, batch_size=BATCH_SIZE, verbose=0).flatten()
    video_frame_probs[vid_id] = probs
    video_labels_map[vid_id] = label


vid_ids    = sorted(video_frame_probs.keys())
vid_scores = np.array([float(np.median(video_frame_probs[v])) for v in vid_ids])
vid_labels = np.array([video_labels_map[v] for v in vid_ids])
y_pred     = (vid_scores >= THRESHOLD).astype(int)


# ── Overall metrics ───────────────────────────────────────────────────────────
roc_auc = roc_auc_score(vid_labels, vid_scores)
pr_auc  = average_precision_score(vid_labels, vid_scores)
bal_acc = balanced_accuracy_score(vid_labels, y_pred)
f1_val  = f1_score(vid_labels, y_pred, zero_division=0)
prec_v  = precision_score(vid_labels, y_pred, zero_division=0)
rec_v   = recall_score(vid_labels, y_pred, zero_division=0)
cm      = confusion_matrix(vid_labels, y_pred)
tn, fp, fn, tp = cm.ravel()


# ── Per-source recalls ────────────────────────────────────────────────────────
def source_recalls(prefix):
    src_ids = [v for v in vid_ids if v.startswith(prefix)]
    src_scores = np.array([float(np.median(video_frame_probs[v])) for v in src_ids])
    src_labels = np.array([video_labels_map[v] for v in src_ids])
    src_preds  = (src_scores >= THRESHOLD).astype(int)
    cm_src = confusion_matrix(src_labels, src_preds, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_src.ravel()
    tnr_s = tn_s / max(tn_s + fp_s, 1)
    tpr_s = tp_s / max(tp_s + fn_s, 1)
    return {
        "n_videos":    int(len(src_ids)),
        "n_real":      int(np.sum(src_labels == 0)),
        "n_fake":      int(np.sum(src_labels == 1)),
        "real_recall": float(tnr_s),
        "fake_recall": float(tpr_s),
        "bal_acc":     float((tnr_s + tpr_s) / 2),
    }


ff_metrics  = source_recalls("ff__")
cdf_metrics = source_recalls("cdf__")


# ── Save + print ──────────────────────────────────────────────────────────────
result = {
    "variant":        VARIANT,
    "threshold":      THRESHOLD,
    "n_test_videos":  int(len(vid_ids)),
    "overall": {
        "roc_auc":     float(roc_auc),
        "pr_auc":      float(pr_auc),
        "bal_acc":     float(bal_acc),
        "f1":          float(f1_val),
        "precision":   float(prec_v),
        "fake_recall": float(rec_v),
        "real_recall": float(tn / max(tn + fp, 1)),
        "confusion":   {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    },
    "ff_pp":  ff_metrics,
    "celebdf": cdf_metrics,
}

with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)


print(f"\n{'='*60}")
print(f"Test results — variant: {VARIANT}")
print(f"{'='*60}")
print(f"  Overall  ROC-AUC: {roc_auc:.4f}  BalAcc: {bal_acc:.4f}")
print(f"           F1: {f1_val:.4f}  Prec: {prec_v:.4f}  Rec: {rec_v:.4f}")
print(f"  FF++     real_recall: {ff_metrics['real_recall']:.4f}  "
      f"fake_recall: {ff_metrics['fake_recall']:.4f}  "
      f"BalAcc: {ff_metrics['bal_acc']:.4f}")
print(f"  CelebDF  real_recall: {cdf_metrics['real_recall']:.4f}  "
      f"fake_recall: {cdf_metrics['fake_recall']:.4f}  "
      f"BalAcc: {cdf_metrics['bal_acc']:.4f}")
print(f"\nSaved → {RESULT_PATH}")
