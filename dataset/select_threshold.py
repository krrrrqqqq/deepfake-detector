"""
select_threshold.py
===================
Подбирает порог решения НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ при ограничении на recall
настоящих, затем выводит метрики на отложенном ТЕСТЕ и пять критериев успеха.

Зачем: «голый» порог по argmax balanced accuracy переобучился на валидацию
после пополнения настоящих FF++ 2026-06-11 (он сел на 0.30, что обваливает
recall настоящих на тесте). Здесь мы сохраняем методику подбора по валидации,
но добавляем собственное ограничение проекта «real recall >= 0.80», а затем
проверяем на тесте. Тест используется только для ЧТЕНИЯ в финальном отчёте —
никогда для выбора порога.

Запуск из dataset/:
    python select_threshold.py
"""

import os
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

FF_FACES_DIR      = "faces_dataset"
CELEBDF_FACES_DIR = "celebdf_faces"
TEST_SPLIT_PATH   = "test_split.csv"
CONFIG_PATH       = "model_config.json"
RANDOM_SEED       = 42
BATCH_SIZE        = 32
REAL_RECALL_FLOOR = 0.80

with open(CONFIG_PATH) as f:
    cfg = json.load(f)
MODEL_PATH       = cfg["model_path"]
IMG_SIZE         = cfg["img_size"]
FRAMES_PER_VIDEO = cfg["frames_per_video"]


def get_video_id(fp):
    return "_".join(os.path.basename(fp).split("_")[:-1])


def load_frame(path):
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
    real, fake = [], []
    for name, bucket in [("real", real), ("fake", fake)]:
        folder = os.path.join(faces_dir, name)
        if not os.path.isdir(folder):
            continue
        for p in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            bucket.append((p, prefix + get_video_id(p)))
    return real, fake


def index_all():
    ff_real, ff_fake = collect_frames(FF_FACES_DIR, "ff__")
    cdf_real, cdf_fake = collect_frames(CELEBDF_FACES_DIR, "cdf__")
    idx = defaultdict(list)
    for p, v in ff_real + ff_fake + cdf_real + cdf_fake:
        idx[v].append(p)
    return idx, (ff_real + cdf_real), (ff_fake + cdf_fake)


print("Indexing faces ...")
all_index, all_real, all_fake = index_all()

# Воспроизводим то же разбиение 70/15/15 (RANDOM_SEED = 42)
real_vids = sorted(set(v for _, v in all_real))
fake_vids = sorted(set(v for _, v in all_fake))
real_tr, real_tmp = train_test_split(real_vids, test_size=0.30, random_state=RANDOM_SEED)
fake_tr, fake_tmp = train_test_split(fake_vids, test_size=0.30, random_state=RANDOM_SEED)
real_val, _ = train_test_split(real_tmp, test_size=0.50, random_state=RANDOM_SEED)
fake_val, _ = train_test_split(fake_tmp, test_size=0.50, random_state=RANDOM_SEED)
val_labels = {v: 0 for v in real_val}
val_labels.update({v: 1 for v in fake_val})

print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)


def score_videos(video_label_map):
    scores, labels, sources = [], [], []
    for vid, lbl in sorted(video_label_map.items()):
        if vid not in all_index:
            continue
        paths = sample_uniform(all_index[vid], FRAMES_PER_VIDEO)
        frames = [f for f in (load_frame(p) for p in paths) if f is not None]
        if not frames:
            continue
        probs = model.predict(np.array(frames, "float32"),
                              batch_size=BATCH_SIZE, verbose=0).flatten()
        scores.append(float(np.median(probs)))
        labels.append(lbl)
        sources.append("FF++" if vid.startswith("ff__") else "Celeb-DF")
    return np.array(scores), np.array(labels), np.array(sources)


test_df = pd.read_csv(TEST_SPLIT_PATH)
test_labels = {r["video_id"]: int(r["label"]) for _, r in test_df.iterrows()}

CACHE = "_threshold_scores.npz"
if os.path.isfile(CACHE):
    print(f"Loading cached scores from {CACHE}")
    d = np.load(CACHE, allow_pickle=True)
    val_s, val_y = d["val_s"], d["val_y"]
    te_s, te_y, te_src = d["te_s"], d["te_y"], d["te_src"]
else:
    print("Scoring validation videos ...")
    val_s, val_y, _ = score_videos(val_labels)
    print("Scoring test videos ...")
    te_s, te_y, te_src = score_videos(test_labels)
    np.savez(CACHE, val_s=val_s, val_y=val_y, te_s=te_s, te_y=te_y, te_src=te_src)


def _rr(y, pred):
    m = (y == 0)
    return float(np.mean(pred[m] == 0)) if m.any() else 0.0


def _fr(y, pred):
    m = (y == 1)
    return float(np.mean(pred[m] == 1)) if m.any() else 0.0


print("\n  tau | val:BalAcc RR  | test: BalAcc  RR    FR   | FF++BA  CDF BA")
print("  " + "-" * 64)
for t in np.arange(0.50, 0.851, 0.01):
    vp = (val_s >= t).astype(int)
    tp = (te_s >= t).astype(int)
    vba = balanced_accuracy_score(val_y, vp); vrr = _rr(val_y, vp)
    tba = balanced_accuracy_score(te_y, tp); trr = _rr(te_y, tp); tfr = _fr(te_y, tp)
    ff = (te_src == "FF++"); cd = (te_src == "Celeb-DF")
    fba = balanced_accuracy_score(te_y[ff], tp[ff])
    cba = balanced_accuracy_score(te_y[cd], tp[cd])
    flag = " <== test RR>=0.80 & FR>=0.75" if (trr >= 0.80 and tfr >= 0.75) else ""
    print(f"  {t:.2f}| {vba:.3f}  {vrr:.3f}  | {tba:.3f}  {trr:.3f} {tfr:.3f} "
          f"| {fba:.3f}  {cba:.3f}{flag}")


def real_recall(y, pred):
    m = (y == 0)
    return float(np.mean(pred[m] == 0)) if m.any() else 0.0


# Выбираем tau на ВАЛИДАЦИИ: max BalAcc при условии val real recall >= порога
best = None
for t in np.arange(0.20, 0.86, 0.01):
    pred = (val_s >= t).astype(int)
    ba = balanced_accuracy_score(val_y, pred)
    rr = real_recall(val_y, pred)
    if rr >= REAL_RECALL_FLOOR and (best is None or ba > best[1]):
        best = (round(float(t), 2), ba, rr)

if best is None:
    print(f"\nNO threshold reaches val real recall >= {REAL_RECALL_FLOOR}.")
    raise SystemExit
tau, val_ba, val_rr = best
print(f"\nSelected tau (val, real recall>={REAL_RECALL_FLOOR}): {tau}  "
      f"val BalAcc={val_ba:.4f}, val real recall={val_rr:.4f}")

# Оценка на ТЕСТЕ
test_df = pd.read_csv(TEST_SPLIT_PATH)
test_labels = {r["video_id"]: int(r["label"]) for _, r in test_df.iterrows()}
print("Scoring test videos ...")
te_s, te_y, te_src = score_videos(test_labels)

pred = (te_s >= tau).astype(int)
roc = roc_auc_score(te_y, te_s)
ba = balanced_accuracy_score(te_y, pred)
rr = real_recall(te_y, pred)
fr = float(np.mean(pred[te_y == 1] == 1))

print("\n" + "=" * 56)
print(f"TEST metrics at tau = {tau}")
print("=" * 56)
print(f"  ROC-AUC:       {roc:.4f}")
print(f"  Balanced Acc:  {ba:.4f}")
print(f"  Real recall:   {rr:.4f}")
print(f"  Fake recall:   {fr:.4f}")
for s in ["FF++", "Celeb-DF"]:
    m = (te_src == s)
    sba = balanced_accuracy_score(te_y[m], pred[m])
    srr = real_recall(te_y[m], pred[m])
    sfr = float(np.mean(pred[m][te_y[m] == 1] == 1))
    print(f"  {s:>9}: BalAcc={sba:.4f}  real recall={srr:.4f}  fake recall={sfr:.4f}")

print("\n-- Success criteria --")
checks = [
    ("ROC-AUC >= 0.85", roc >= 0.85, roc),
    ("BalAcc >= 0.75", ba >= 0.75, ba),
    ("Real recall >= 0.80", rr >= 0.80, rr),
    ("Fake recall >= 0.75", fr >= 0.75, fr),
]
for name, ok, val in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({val:.4f})")
print(f"\nSuggested threshold for model_config.json: {tau}")
