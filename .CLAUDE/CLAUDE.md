# Deepfake Detector — Project Instructions

## Project Overview

Deepfake detection system based on fine-tuned EfficientNet-B0 (224×224). Trained on combined FaceForensics++ + Celeb-DF v2 dataset with video-level train/val/test splitting. A legacy SVM pipeline is available as a fallback.

---

## Tech Stack

| Component | Technology |
|---|---|
| Classifier | Fine-tuned EfficientNet-B0, 224×224, sigmoid head, two-phase training |
| Face detection | MediaPipe 0.10+ — blaze_face_short_range.tflite |
| Imbalance handling | Per-sample weights across 4 (source, label) cells — balances both real/fake and FF++/Celeb-DF simultaneously |
| Primary metrics | ROC-AUC, Balanced Accuracy, per-source BalAcc |
| Threshold | Auto-tuned on val by balanced accuracy, saved to model_config.json |
| UI verdict | Three-way REAL / UNCERTAIN / FAKE via fixed band [0.40, 0.85] in app.py |
| Aggregation | Median of per-frame probabilities |
| Video processing | OpenCV — 10 uniform frames per video |
| Web interface | Flask + Jinja2 |

---

## Project Structure

```
deepfake_detector/
├── dataset/
│   ├── FaceForensics/              # Raw FF++ videos (DO NOT MODIFY)
│   ├── Celeb-DF-v2/                # Raw Celeb-DF videos (DO NOT MODIFY)
│   ├── celebdf_subset/             # 890 real + 300 fake Celeb-DF videos
│   ├── celebdf_faces/              # Cropped faces from Celeb-DF
│   ├── faces_dataset/              # Cropped faces from ALL FF++ videos
│   ├── dataset_labels.csv          # All FF++ video paths + labels (1200)
│   ├── test_split.csv              # Held-out test video IDs + labels
│   ├── model_config.json           # Threshold, img_size, backbone, aggregation
│   ├── efficientnet_combined.keras # Fine-tuned model
│   ├── blaze_face_short_range.tflite
│   ├── prepare_dataset.py
│   ├── prepare_celebdf.py
│   ├── extract_faces.py            # FF++ faces (uses dataset_labels.csv)
│   ├── extract_faces_celebdf.py    # Celeb-DF faces
│   ├── finetune_combined.py        # Main training (FF++ + Celeb-DF)
│   ├── test_combined.py            # Evaluate on held-out test split
│   └── (legacy SVM scripts)
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
└── requirements.txt
```

---

## Pipeline Execution Order

```
1. prepare_dataset.py          # dataset_labels.csv (all 1200 FF++ videos)
2. prepare_celebdf.py          # celebdf_subset/ (890 real + 300 fake)
3. extract_faces.py            # faces_dataset/ from ALL FF++ videos
4. extract_faces_celebdf.py    # celebdf_faces/ from Celeb-DF
5. finetune_combined.py        # efficientnet_combined.keras + test_split.csv + model_config.json
6. test_combined.py            # Metrics on held-out test split
```

---

## Dataset Details

| Dataset | Real | Fake | Total |
|---|---|---|---|
| FaceForensics++ | 300 | 900 (3 methods × 300) | 1200 |
| Celeb-DF v2 | 890 (all Celeb-real + YouTube-real) | 300 (subset) | 1190 |
| **Combined** | **1190** | **1200** | **2390** |

Split: 70% train / 15% val / 15% test (video-level). Although overall real/fake is ~1:1, within each class there is a source imbalance (Celeb-DF real ≈ 3× FF++ real, FF++ fake ≈ 3× Celeb-DF fake). Handled via per-sample weights across the four (source, label) cells.

Held-out test-set results (threshold 0.79, median aggregation): ROC-AUC **0.898**, BalAcc **0.794**, F1 **0.787**, Real recall **0.827**, Fake recall **0.761**. Per-source: FF++ BalAcc 0.635, Celeb-DF BalAcc 0.833.

---

## Critical Rules

### Symmetric augmentation ONLY
NEVER use different augmentation for real vs fake. Asymmetric augmentation (heavy for real, light for fake) causes the model to learn augmentation artefacts instead of deepfake artefacts. The symptom: ~86% train accuracy but val_accuracy stuck at exactly 50%.

### Monitor val_auc, not val_accuracy
With imbalanced data, val_accuracy is misleading. A model predicting everything as the majority class gets 67% accuracy on 1:2 data. Use val_auc (threshold-independent, class-balance-robust) for EarlyStopping and ModelCheckpoint with `mode="max"`.

### Per-sample weights across (source, label) cells, not class_weight
`class_weight` only balances real vs fake. Our data has a second imbalance (within real, Celeb-DF outweighs FF++ by ~3×; within fake, the reverse). If uncorrected, the model learns "source = label" as a shortcut — FF++ real recall collapsed to 0.30 even with perfect overall real/fake balance. Per-sample weights `w = n_total / (n_cells * n_cell)` equalise contribution of all four cells. Passed through `tf.data` as the third tuple element; Keras reads it automatically.

### Median aggregation, not mean
For combining per-frame probabilities into a video score, use median. It's robust to outlier frames.

### Threshold from model_config.json
Never hardcode threshold=0.5. The training script optimises threshold on the validation set and saves it. The test script loads it automatically.

### Three-way verdict with fixed uncertainty band
`app.py` defines `UNCERTAIN_LOW = 0.40` and `UNCERTAIN_HIGH = 0.85` — anchored to the score itself, not to the threshold. Scores below 0.40 → REAL; above 0.85 → FAKE; in between → UNCERTAIN with raw fake-probability shown. The optimised threshold (currently 0.79) falls inside the band. Outside the band, REAL/FAKE confidence is rescaled: band edge → 50%, extreme → 100%. Semantics stay stable across retrainings even if the threshold moves.

### BGR→RGB conversion
```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

### MediaPipe 0.10+ API
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
```

---

## Known Issues & History

| Issue | Status | Fix |
|---|---|---|
| Asymmetric augmentation → learns augmentation, not deepfakes | FIXED | Symmetric augmentation |
| val_accuracy=50% always, predicts all as one class | FIXED | val_auc + class_weight + symmetric aug |
| B4 380×380 too slow on CPU (8.5h) | FIXED | B0 224×224 (~2-3h) |
| Hardcoded threshold=0.5 mismatched with training | FIXED | Auto-save to model_config.json |
| mean(probs) sensitive to outlier frames | FIXED | median aggregation |
| Undersampling discards 2/3 of fake data | FIXED | class_weight instead |
| Cross-dataset SVM ~50% accuracy | FIXED | Combined dataset + fine-tuning |
| Real recall collapse (0.61) from 1:2 class imbalance | FIXED | Included all 890 Celeb-DF real videos → 1:1 balance → TNR 0.77, BalAcc 0.80 |
| Hidden "domain shortcut" — model learned FF→fake, CDF→real after 1:1 rebalance (FF++ real recall 0.30, Celeb-DF fake recall 0.66) | PARTIALLY FIXED | Per-sample weights across (source, label) cells → FF++ real 0.50, Celeb-DF fake 0.73. Full fix blocked by FF++ real data starvation (only 300 videos available) |
| Hard REAL/FAKE label unreliable near the decision boundary | FIXED | Fixed uncertainty band `[0.40, 0.85]` in app.py — three-way verdict, decoupled from threshold |

---

## Environment

- Python 3.13, TensorFlow >= 2.16, mediapipe 0.10.32, scikit-learn >= 1.4
- oneDNN warnings from TensorFlow are normal
- All scripts run from dataset/ directory

---

## What NOT to do

- Do NOT use asymmetric augmentation (different for real vs fake)
- Do NOT monitor val_accuracy for EarlyStopping on imbalanced data
- Do NOT hardcode threshold — always read from model_config.json (`test_combined.py`). In `app.py` the UI uses fixed band, not threshold.
- Do NOT undersample — use per-sample weights
- Do NOT fall back to plain `class_weight` alone — it misses the within-class source imbalance that caused the "domain shortcut" bug
- Do NOT use mean for video-level aggregation — use median
- Do NOT split by frame — always split by video ID
- Do NOT report only overall metrics — run the per-source breakdown in `test_combined.py`; overall BalAcc can hide source-specific recall collapse
