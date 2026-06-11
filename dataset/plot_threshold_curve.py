"""
plot_threshold_curve.py
=======================
Воспроизводит валидационное разбиение (тот же RANDOM_SEED = 42, что и в
finetune_combined.py), прогоняет все валидационные видео через обученную модель
и строит график сбалансированной точности (balanced accuracy) в зависимости от
порога решения на той же сетке, что использовалась при обучении
(np.arange(0.20, 0.80, 0.01)).

Красный маркер показывает порог, сохранённый в model_config.json (τ = 0.79).
ПРИМЕЧАНИЕ: после переобучения 2026-06-11 безусловный оптимум по сетке
переобучился на валидацию (~0.30) и лежит ниже порога по recall настоящих,
поэтому τ выбран как argmax balanced accuracy при условии real recall >= 0.80
и равен 0.79 — маркер не совпадает с пиком кривой.

Читает:
  - efficientnet_combined.keras
  - model_config.json
  - faces_dataset/ и celebdf_faces/

Сохраняет:
  - figure_11_threshold_curve.png
  - figure_11_threshold_curve.svg

Запуск из dataset/:
    python plot_threshold_curve.py
"""

import os
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score


FF_FACES_DIR      = "faces_dataset"
CELEBDF_FACES_DIR = "celebdf_faces"
CONFIG_PATH       = "model_config.json"

RANDOM_SEED       = 42
BATCH_SIZE        = 32


# ── Загрузка конфигурации ───────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

MODEL_PATH       = cfg["model_path"]
IMG_SIZE         = cfg["img_size"]
FRAMES_PER_VIDEO = cfg["frames_per_video"]
THRESHOLD        = cfg["threshold"]


# ── Вспомогательные функции ────────────────────────────────────────────────────
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


def collect_frames(faces_dir, prefix):
    real_frames, fake_frames = [], []
    for label_name, bucket in [("real", real_frames), ("fake", fake_frames)]:
        folder = os.path.join(faces_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for path in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            vid_id = prefix + get_video_id(path)
            bucket.append((path, vid_id))
    return real_frames, fake_frames


# ── Воспроизводим то же разбиение 70/15/15, что в finetune_combined.py ─────────
print("Reconstructing validation split (RANDOM_SEED = 42) ...")
ff_real, ff_fake     = collect_frames(FF_FACES_DIR, "ff__")
cdf_real, cdf_fake   = collect_frames(CELEBDF_FACES_DIR, "cdf__")
all_real = ff_real + cdf_real
all_fake = ff_fake + cdf_fake

real_vids = sorted(set(vid for _, vid in all_real))
fake_vids = sorted(set(vid for _, vid in all_fake))

real_train_v, real_temp_v = train_test_split(
    real_vids, test_size=0.30, random_state=RANDOM_SEED)
fake_train_v, fake_temp_v = train_test_split(
    fake_vids, test_size=0.30, random_state=RANDOM_SEED)
real_val_v, _ = train_test_split(
    real_temp_v, test_size=0.50, random_state=RANDOM_SEED)
fake_val_v, _ = train_test_split(
    fake_temp_v, test_size=0.50, random_state=RANDOM_SEED)

val_set = set(real_val_v + fake_val_v)
val_labels_map = {v: 0 for v in real_val_v}
val_labels_map.update({v: 1 for v in fake_val_v})
print(f"Validation videos: {len(val_set)} "
      f"({len(real_val_v)} real, {len(fake_val_v)} fake)")


# ── Индексация изображений лиц, только для валидационных видео ─────────────────
val_index = defaultdict(list)
for path, vid in all_real + all_fake:
    if vid in val_set:
        val_index[vid].append(path)


# ── Загрузка модели ──────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)


# ── Инференс по валидационным видео ───────────────────────────────────────────
print("Running inference on validation videos ...")
vid_scores = []
vid_labels = []

for vid_id in sorted(val_set):
    paths = sample_uniform(val_index[vid_id], FRAMES_PER_VIDEO)
    frames = [load_frame(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        continue
    arr = np.array(frames, dtype="float32")
    probs = model.predict(arr, batch_size=BATCH_SIZE, verbose=0).flatten()
    vid_scores.append(float(np.median(probs)))
    vid_labels.append(val_labels_map[vid_id])

vid_scores = np.array(vid_scores)
vid_labels = np.array(vid_labels)
print(f"Scored {len(vid_labels)} validation videos")


# ── Перебор порогов по сетке ──────────────────────────────────────────────────
REAL_RECALL_FLOOR = 0.80
grid = np.arange(0.20, 0.80 + 1e-9, 0.01)
balaccs, realrecalls = [], []
for t in grid:
    preds = (vid_scores >= t).astype(int)
    balaccs.append(balanced_accuracy_score(vid_labels, preds))
    real_mask = (vid_labels == 0)
    realrecalls.append(float(np.mean(preds[real_mask] == 0)))
balaccs = np.array(balaccs)
realrecalls = np.array(realrecalls)

best_idx = int(np.argmax(balaccs))
best_t   = float(grid[best_idx])
best_ba  = float(balaccs[best_idx])

# Также считаем BalAcc при сохранённом пороге для подписи на графике
saved_idx = int(np.argmin(np.abs(grid - THRESHOLD)))
saved_ba  = float(balaccs[saved_idx])

print(f"\nGrid optimum: tau = {best_t:.2f}, BalAcc = {best_ba:.4f}")
print(f"Persisted tau = {THRESHOLD}, BalAcc on val = {saved_ba:.4f}")


# ── Построение графика ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(8.5, 5.0))

# Показываем только рабочую область (>= 0.50); безусловный пик BalAcc около 0.30
# лежит ниже порога по recall настоящих и всё равно исключён из выбора.
PLOT_FROM = 0.50
m = grid >= PLOT_FROM

ax.plot(grid[m], balaccs[m], color="#1f4e79", linewidth=2.0,
        label="Validation balanced accuracy")
ax.axvline(THRESHOLD, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.8)
ax.scatter([THRESHOLD], [saved_ba], color="#c0392b", s=90, zorder=5,
           label=f"Selected τ = {THRESHOLD}  (BalAcc = {saved_ba:.4f})")

ax.set_xlim([PLOT_FROM - 0.02, 0.82])
ax.set_ylim([min(balaccs[m]) - 0.01, max(balaccs[m]) + 0.012])
ax.set_xlabel("Decision threshold τ")
ax.set_ylabel("Balanced accuracy on validation set")
ax.set_title("Figure 11 — Validation balanced accuracy vs decision threshold\n"
             f"(operating region; selected τ = {THRESHOLD}; "
             f"N = {len(vid_labels)} videos)")
ax.legend(loc="lower right", framealpha=0.95)
ax.grid(alpha=0.3)

fig.tight_layout()

out_png = "figure_11_threshold_curve.png"
out_svg = "figure_11_threshold_curve.svg"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
print(f"\nSaved -> {out_png}")
print(f"Saved -> {out_svg}")
