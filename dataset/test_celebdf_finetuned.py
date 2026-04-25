"""
Evaluate the fine-tuned EfficientNet-B4 model on the Celeb-DF v2 test set.

Inference:
  - For each video: run all available frames through the model
  - Average per-frame sigmoid probabilities → video-level score
  - Threshold at 0.5 for FAKE / REAL prediction

Run from the dataset/ directory:
    python test_celebdf_finetuned.py
"""

import os
import glob
from collections import defaultdict

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)

# ── Config ─────────────────────────────────────────────────────────────────────
FACES_DIR        = "celebdf_faces"
IMG_SIZE         = 380
BATCH_SIZE       = 16
FRAMES_PER_VIDEO = 10
# Set this to the optimal threshold printed by finetune_efficientnet.py /
# finetune_combined.py. The scripts search val F1 over [0.25, 0.75] — using
# that value here avoids re-tuning on the test set.
THRESHOLD        = 0.5

# Auto-detect model: prefer the combined model if present, fall back to the
# standard fine-tuned one. Override by setting MODEL_PATH explicitly.
_CANDIDATES = ["efficientnet_combined.keras", "efficientnet_finetuned.keras"]
MODEL_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[-1])


# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading fine-tuned model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.\n")


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_video_id(filepath: str) -> str:
    """Strip frame-index suffix to recover the video identifier."""
    return "_".join(os.path.basename(filepath).split("_")[:-1])


def load_frame(path: str) -> np.ndarray | None:
    """Read image, resize, BGR->RGB, preprocess for EfficientNet."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    img = img / 127.5 - 1.0
    return img


def predict_video(frame_paths: list[str]) -> float:
    """
    Return average sigmoid probability (fake=1) for a set of frame paths.
    Processes frames in one batch if they fit, otherwise in chunks.
    """
    images = []
    for p in sorted(frame_paths)[:FRAMES_PER_VIDEO]:
        img = load_frame(p)
        if img is not None:
            images.append(img)

    if not images:
        return float("nan")

    arr   = np.array(images, dtype="float32")    # (N, 380, 380, 3)
    probs = model.predict(arr, batch_size=BATCH_SIZE, verbose=0).flatten()
    return float(np.mean(probs))


# ── Build video -> [frame_paths] mapping ───────────────────────────────────────
video_probs  = []
video_labels = []
video_ids    = []

for label_name, label in [("real", 0), ("fake", 1)]:
    folder = os.path.join(FACES_DIR, label_name)
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found: {folder}")
        continue

    all_frames = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                        glob.glob(os.path.join(folder, "*.png")))

    # Group frames by video
    videos: dict[str, list[str]] = defaultdict(list)
    for fp in all_frames:
        videos[get_video_id(fp)].append(fp)

    print(f"{label_name}: {len(videos)} videos, {len(all_frames)} frames total")

    for vid_id, paths in sorted(videos.items()):
        prob = predict_video(paths)
        if np.isnan(prob):
            print(f"  Skipped (no readable frames): {vid_id}")
            continue
        video_probs.append(prob)
        video_labels.append(label)
        video_ids.append(vid_id)

print()

# ── Metrics ────────────────────────────────────────────────────────────────────
video_probs  = np.array(video_probs)
video_labels = np.array(video_labels)
y_pred       = (video_probs >= THRESHOLD).astype(int)

print(f"Total videos evaluated: {len(video_labels)}")
print(f"Real: {np.sum(video_labels == 0)}, Fake: {np.sum(video_labels == 1)}")
print()

acc_v  = accuracy_score(video_labels, y_pred)
prec_v = precision_score(video_labels, y_pred, zero_division=0)
rec_v  = recall_score(video_labels, y_pred, zero_division=0)
f1_v   = f1_score(video_labels, y_pred, zero_division=0)

print("=== Celeb-DF v2 Test Results (Fine-tuned EfficientNet-B4) ===\n")
print(f"Accuracy:  {acc_v:.4f} ({acc_v*100:.2f}%)")
print(f"Precision: {prec_v:.4f} ({prec_v*100:.2f}%)")
print(f"Recall:    {rec_v:.4f} ({rec_v*100:.2f}%)")
print(f"F1-score:  {f1_v:.4f} ({f1_v*100:.2f}%)")

try:
    auc = roc_auc_score(video_labels, video_probs)
    print(f"ROC-AUC:   {auc:.4f} ({auc*100:.2f}%)")
except Exception as e:
    print(f"ROC-AUC skipped: {e}")

cm = confusion_matrix(video_labels, y_pred)
print("\nConfusion Matrix:")
print(f"              Predicted Real  Predicted Fake")
print(f"Actual Real       {cm[0][0]:5d}           {cm[0][1]:5d}")
print(f"Actual Fake       {cm[1][0]:5d}           {cm[1][1]:5d}")

if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
