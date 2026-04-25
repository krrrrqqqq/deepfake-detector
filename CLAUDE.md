# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deepfake detection system based on fine-tuned EfficientNet-B0 (224×224). Trained on a combined dataset of FaceForensics++ and Celeb-DF v2 with video-level train/val/test splitting. A legacy SVM pipeline is also available as a fallback.

---

## Running the Pipeline

All dataset scripts must be run from `dataset/` as the working directory (they use relative paths):

```bash
cd dataset

# 1. Prepare raw data
python prepare_dataset.py          # dataset_labels.csv (all FF++ videos)
python prepare_celebdf.py          # celebdf_subset/ (890 real + 300 fake)

# 2. Extract faces from both datasets
python extract_faces.py            # faces_dataset/real/ + faces_dataset/fake/ (ALL FF++ videos)
python extract_faces_celebdf.py    # celebdf_faces/real/ + celebdf_faces/fake/

# 3. Fine-tune on combined dataset (GPU recommended, CPU ~2-3h)
python finetune_combined.py        # efficientnet_combined.keras + test_split.csv + model_config.json

# 4. Evaluate on held-out test set
python test_combined.py            # metrics on test split (reads threshold from model_config.json)

# Web app (run from project root)
python app.py                      # http://localhost:5000
```

### Legacy SVM pipeline (kept for reference, not recommended)

```bash
python split_dataset.py            # train_split.csv + val_split.csv
python extract_frames.py           # extracted_frames/ (diagnostic)
python extract_features.py         # X_train.npy (N,3584), y_train.npy
python train_svm.py                # svm_model.pkl, scaler.pkl
python extract_features_celebdf.py # X_test.npy (N,3584), y_test.npy
python test_celebdf.py             # SVM metrics on Celeb-DF
```

Install dependencies: `pip install -r requirements.txt`

---

## Architecture

### Primary — Fine-tuned EfficientNet-B0 (combined dataset)

1. MediaPipe face detection → crop with 20% padding (central-crop fallback)
2. Fine-tuned EfficientNet-B0 (224×224) with sigmoid head → per-frame fake probability
3. Median of per-frame probabilities → video-level score
4. Threshold from `model_config.json` → REAL / UNCERTAIN / FAKE

The three-way verdict uses a **fixed uncertainty band** `[UNCERTAIN_LOW, UNCERTAIN_HIGH] = [0.40, 0.85]` in `app.py`, decoupled from the threshold. Rationale: score ≈ 0.5 is intrinsically ambiguous regardless of where the decision threshold sits (currently 0.79), so the band is anchored to the score itself rather than to the threshold. The optimised threshold falls *inside* the band — scores right at the boundary are maximally uncertain. Scores below 0.40 are confidently REAL, scores above 0.85 are confidently FAKE; anything in between is reported as UNCERTAIN with the raw fake-probability. Confidence for REAL/FAKE is rescaled: 50% at the band edge → 100% at the extreme.

Training uses two-phase approach:
- **Phase 1 (warm-up, 3 epochs):** Frozen base, only Dense head trains (lr=1e-3)
- **Phase 2 (fine-tune, up to 30 epochs):** Top 30 layers unfrozen (lr=1e-5)

### Key design decisions

**EfficientNet-B0 (224×224) instead of B4 (380×380):** ~5× faster on CPU. B4's extra capacity only matters for cross-dataset generalisation, which we dropped.

**Symmetric augmentation:** The SAME augmentation is applied to both real and fake. Asymmetric augmentation (heavy for real, light for fake) teaches the model to detect augmentation artefacts instead of deepfake artefacts — val_accuracy stays at 50%.

**Per-sample weights across (source, label) cells instead of class_weight:** `class_weight` only balances real vs fake. But after adding all 890 Celeb-DF real videos, a second imbalance appeared — within the real class, Celeb-DF outweighs FF++ by ~3×. The model latched onto source as a shortcut: FF++ real → FAKE, Celeb-DF fake → REAL. Per-sample weighting rebalances all four `(source, label)` cells equally via `w = n_total / (4 * n_cell)`: minority cells (FF++ real, Celeb-DF fake) get ~×1.99, majority cells (~×0.67). Subsumes `class_weight`.

**Monitor val_auc (not val_accuracy):** AUC is threshold-independent and robust to class imbalance. EarlyStopping and ModelCheckpoint both use `val_auc` with `mode="max"`.

**Video-level threshold optimisation:** After training, validation frames are grouped by video, aggregated via median, and the threshold maximising **balanced accuracy** is saved to `model_config.json`. F1 is degenerate on imbalanced data (it can reward "predict all positive"); balanced accuracy penalises that. The test script reads the saved threshold automatically.

**Median aggregation (not mean):** Robust to outlier frames. One noisy/low-quality frame doesn't drag the whole video score.

**tf.data.cache():** JPEG decode+resize is cached in RAM after the first epoch. Subsequent epochs skip disk I/O entirely.

### `app.py` model priority

On startup, `app.py` searches for models in this order:
1. `efficientnet_combined.keras` (combined fine-tuned — preferred)
2. `efficientnet_finetuned.keras` (FF++-only fine-tuned — fallback)
3. SVM pipeline (`svm_model.pkl` + `scaler.pkl` — last resort)

`app.py` reads `img_size`, `threshold`, and `aggregation` from `model_config.json` at startup — the fine-tuned model is served at whatever resolution training used (currently 224).

---

## Dataset Details

| Dataset | Real | Fake | Total |
|---|---|---|---|
| FaceForensics++ | 300 videos | 900 videos (Deepfakes + FaceSwap + Face2Face) | 1200 |
| Celeb-DF v2 | 890 videos (all Celeb-real + YouTube-real) | 300 videos (subset) | 1190 |
| **Combined** | **1190 videos** | **1200 videos** | **2390** |

Combined dataset is split 70/15/15 at the video level:
- Train: ~1673 videos (~16,721 frames)
- Val: ~358 videos (~3,580 frames)
- Test: ~359 videos (held-out, saved to `test_split.csv`)

Within-class source imbalance: FF++ real (2250 train frames) vs Celeb-DF real (6071 train frames); FF++ fake (6340) vs Celeb-DF fake (2060). Per-sample weights rebalance all four cells equally.

### Current metrics (held-out test set, 359 videos)

| Metric | Value |
|---|---|
| ROC-AUC | 0.898 |
| PR-AUC | 0.892 |
| Balanced Accuracy | 0.794 |
| F1-score | 0.787 |
| Precision / Recall (fake) | 0.816 / 0.761 |
| TNR (real recall, overall) | 0.827 |
| Threshold (median aggregation) | 0.79 |
| Per-frame separation (fake μ − real μ) | 0.550 |

### Per-source breakdown (what overall metrics hide)

| Source | Real recall | Fake recall | BalAcc |
|---|---|---|---|
| FaceForensics++ | 0.500 | 0.770 | 0.635 |
| Celeb-DF v2 | 0.933 | 0.732 | 0.833 |

FF++ real recall of 0.50 (was 0.30 before per-sample weighting) is a known limitation driven by data starvation: only 300 FF++ real videos available, versus 890 Celeb-DF real. The uncertainty band `[0.40, 0.85]` in `app.py` absorbs most borderline FF++ real videos (per-frame median 0.75 falls inside the band) into UNCERTAIN rather than forcing a FAKE verdict.

---

## Critical Rules

**Fake filename prefix:** Files in `faces_dataset/fake/` must use format `{Method}__{video}.mp4_{frame}.jpg`. Without the method prefix, all three FF++ methods overwrite each other.

**`extract_faces.py` uses `dataset_labels.csv`** (ALL FF++ videos), not `train_split.csv`.

**BGR→RGB conversion:** `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` must appear in every script that loads images for EfficientNet.

**Symmetric augmentation only:** Never use different augmentation for real vs fake frames — this causes the model to learn augmentation patterns instead of deepfake artefacts.

**Do NOT use val_accuracy for EarlyStopping** with imbalanced data. Use `val_auc` with `mode="max"`.

**MediaPipe API (v0.10+):** Use `mediapipe.tasks.python.vision.FaceDetector` — the old `mp.solutions.face_detection` API does not exist in v0.10+.

---

## Saved Artefacts

| File | Producer | Consumer |
|---|---|---|
| `efficientnet_combined.keras` | `finetune_combined.py` | `test_combined.py`, `app.py` |
| `test_split.csv` | `finetune_combined.py` | `test_combined.py` |
| `model_config.json` | `finetune_combined.py` | `test_combined.py` |

`model_config.json` contains: threshold, img_size, backbone, aggregation method, frames_per_video. The test script reads all of these automatically — no manual threshold editing needed.

---

## Known Issues (resolved)

| Issue | Fix applied |
|---|---|
| Duplicate fake filenames (296 instead of 720 videos) | Method prefix in filename |
| Haar Cascade 41% face detection rate | Replaced with MediaPipe (98.4%) |
| Missing BGR→RGB in Celeb-DF pipeline | Added in `extract_features_celebdf.py` |
| Frame-level vs video-level mismatch between train/test | Both scripts now use mean pooling per video |
| Class imbalance 1:3 — SVM predicted fake for almost everything | class_weight in fine-tuning |
| Cross-dataset accuracy ~50% (SVM on frozen features) | Fine-tuned EfficientNet on combined dataset |
| Asymmetric augmentation → model learns augmentation, not deepfakes | Switched to symmetric augmentation |
| val_accuracy=50% all epochs, model predicts all as one class | Monitor val_auc; class_weight; symmetric aug |
| Training too slow on CPU (8.5h with B4 380×380) | Switched to B0 224×224 (~2-3h) |
| Real recall 0.61 from 1:2 class imbalance (FP rate 39%) | Included all 890 available Celeb-DF real videos → 1:1 balance → real recall 0.77, BalAcc +9 p.p. |
| Within-class source imbalance: FF++ real recall 0.30 (model learned "domain = label" shortcut: FF-style → fake, CDF-style → real) | Per-sample weights across (source, label) cells replaced class_weight → FF++ real recall 0.30 → 0.50, Celeb-DF fake recall 0.66 → 0.73. Partial fix; full resolution needs more FF++ real data |
| Hard REAL/FAKE label misleading for scores near the boundary | Fixed uncertainty band `[0.40, 0.85]` in `app.py` — three-way verdict; raw fake-probability shown inside the band. Decoupled from the (moving) threshold so semantics stay stable across retrainings |

---

## Environment

- Python 3.13, TensorFlow >= 2.16, mediapipe 0.10.32, scikit-learn >= 1.4
- oneDNN warnings from TensorFlow are expected and harmless
- All scripts assume they are run from their own directory (relative paths to data folders)
