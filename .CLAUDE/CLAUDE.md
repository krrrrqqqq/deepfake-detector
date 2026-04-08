# Deepfake Detector — Project Instructions

## Project Overview

Hybrid deepfake detection system combining deep feature extraction (EfficientNet-B4) with classical ML classification (SVM). Trained on FaceForensics++, cross-dataset tested on Celeb-DF v2.

---

## Tech Stack

| Component | Technology |
|---|---|
| Feature extraction | TensorFlow/Keras — EfficientNetB4, weights=imagenet, include_top=False, pooling=avg |
| Face detection | MediaPipe 0.10+ — blaze_face_short_range.tflite |
| Classifier | sklearn SVC — kernel=rbf, C=10, gamma=scale, class_weight=balanced |
| Preprocessing | StandardScaler + L2 normalize (normalize from sklearn) |
| Video processing | OpenCV — 10 uniform frames per video |
| Web interface | Flask + Jinja2 |

---

## Project Structure

```
deepfake_detector/
├── dataset/
│   ├── FaceForensics/              # Raw FF++ videos (DO NOT MODIFY)
│   ├── Celeb-DF-v2/                # Raw Celeb-DF videos (DO NOT MODIFY)
│   ├── celebdf_subset/             # 300 real + 300 fake Celeb-DF videos
│   ├── celebdf_frames/             # Extracted frames from Celeb-DF
│   ├── celebdf_faces/              # Cropped faces from Celeb-DF
│   ├── extracted_frames/           # Extracted frames from FF++
│   ├── faces_dataset/              # Cropped faces from FF++ (real/ + fake/)
│   ├── dataset_labels.csv          # All video paths + labels
│   ├── train_split.csv             # 80% train (960 videos)
│   ├── val_split.csv               # 20% validation (240 videos)
│   ├── X_train.npy                 # FF++ embeddings (960, 1792)
│   ├── y_train.npy                 # FF++ labels (960,)
│   ├── X_test.npy                  # Celeb-DF embeddings (598, 1792)
│   ├── y_test.npy                  # Celeb-DF labels (598,)
│   ├── svm_model.pkl               # Trained SVM model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── blaze_face_short_range.tflite  # MediaPipe model file
│   ├── prepare_dataset.py
│   ├── split_dataset.py
│   ├── extract_frames.py
│   ├── extract_faces.py            # Uses MediaPipe 0.10+
│   ├── extract_features.py
│   ├── prepare_celebdf.py
│   ├── extract_frames_celebdf.py
│   ├── extract_features_celebdf.py
│   ├── train_svm.py
│   └── test_celebdf.py
├── model/
│   ├── efficientnet.py             # Legacy — do not use
│   └── train_model.py              # Legacy — do not use
├── static/
│   └── style.css
├── templates/
│   └── index.html
└── utils/
    ├── feature_extraction.py       # Empty placeholder
    └── preprocessing.py            # Empty placeholder
```

---

## Pipeline Execution Order

```
1. prepare_dataset.py          # Creates dataset_labels.csv from FF++
2. split_dataset.py            # Creates train_split.csv + val_split.csv
3. extract_frames.py           # Extracts 10 uniform frames per FF++ video
4. extract_faces.py            # MediaPipe face detection → faces_dataset/
5. extract_features.py         # EfficientNet-B4 → X_train.npy, y_train.npy
6. train_svm.py                # Trains SVM → svm_model.pkl, scaler.pkl
--- cross-dataset test ---
7. prepare_celebdf.py          # Copies 300+300 Celeb-DF videos to celebdf_subset/
8. extract_frames_celebdf.py   # Extracts 10 uniform frames per Celeb-DF video
9. extract_features_celebdf.py # EfficientNet-B4 → X_test.npy, y_test.npy
10. test_celebdf.py            # Final metrics on Celeb-DF
```

---

## Dataset Details

### FaceForensics++ (Train)
- **Real:** 240 videos from `original_sequences/youtube/c23/videos/`
- **Fake:** 720 videos — Deepfakes (240) + FaceSwap (240) + Face2Face (240)
- **Class ratio:** 1:3 (real:fake) — imbalanced, must handle in training
- **Compression:** c23

### Celeb-DF v2 (Test only — never use for training)
- **Real:** 298 videos (from Celeb-real + YouTube-real)
- **Fake:** 300 videos (from Celeb-synthesis)
- **Class ratio:** ~1:1

---

## Critical Implementation Rules

### File naming in faces_dataset/
Fake face images MUST include the manipulation method in the filename to avoid overwriting:
```
Format: {Method}__{video_name}_{frame_idx}.jpg
Example: Deepfakes__107_109.mp4_0.jpg
         FaceSwap__107_109.mp4_0.jpg
         Face2Face__107_109.mp4_0.jpg
```
Without this, all three methods share the same filename and overwrite each other, leaving only 296 unique videos instead of 720.

### Video-level embedding (not frame-level)
Each video is represented as a SINGLE 1792-dim vector = mean of all frame embeddings.
```python
embeddings = model.predict(images)       # shape: (n_frames, 1792)
video_embedding = np.mean(embeddings, axis=0)  # shape: (1792,)
```
Never classify individual frames and then majority-vote — the SVM receives one vector per video.

### Preprocessing must match between train and test
Always apply in this exact order:
```python
X = scaler.transform(X)   # StandardScaler (fit only on train)
X = normalize(X)          # L2 normalization
```

### BGR→RGB conversion is mandatory
OpenCV reads images as BGR. EfficientNet expects RGB.
```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
This must be present in BOTH extract_features.py AND extract_features_celebdf.py.

### Class imbalance handling
Training data is 1:3 (real:fake). Options (pick one):
- **Undersample fake to 240** — take 80 videos from each method (Deepfakes/FaceSwap/Face2Face)
- **Use class_weight=balanced** in SVC — partial fix, not sufficient alone
- Do NOT use gamma="auto" with SVC — at 1792 features it causes degenerate solutions
- Do NOT optimize with scoring="f1" in GridSearchCV — allows "predict all fake" to win

### MediaPipe face detection (v0.10+)
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.4)
detector = vision.FaceDetector.create_from_options(options)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = detector.detect(mp_image)
```
The old `mp.solutions.face_detection` API does NOT work in mediapipe 0.10+.

---

## Known Issues & History

### Issue 1: Duplicate filenames (FIXED)
- **Problem:** Deepfakes, FaceSwap, Face2Face all produce files like `107_109.mp4`. Old code used `os.path.basename(video_path)` as filename — each method overwrote the previous, leaving only 296/720 unique fake videos.
- **Fix:** Prefix filename with method name extracted from path.

### Issue 2: Haar Cascade low detection rate (FIXED)
- **Problem:** Old parameters `scaleFactor=1.3, minNeighbors=5` detected faces in only 41% of fake frames.
- **Fix:** Replaced with MediaPipe — 98.4% detection rate.

### Issue 3: Missing BGR→RGB in Celeb-DF pipeline (FIXED)
- **Problem:** `extract_features_celebdf.py` was missing `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- **Fix:** Added conversion.

### Issue 4: Frame-level vs video-level mismatch (FIXED)
- **Problem:** `extract_features_celebdf.py` treated each frame as a separate sample. `extract_features.py` averaged frames per video. Train/test distributions didn't match.
- **Fix:** Both scripts now produce one embedding per video via mean pooling.

### Issue 5: Class imbalance 1:3 (PENDING)
- **Problem:** 240 real vs 720 fake causes model to predict fake for almost everything. TN ≈ 2 on validation.
- **Fix needed:** Add undersample in train_svm.py — resample fake to 240 samples before training.

### Issue 6: probability=True destabilizes SVM (FIXED)
- **Problem:** With only ~428 training samples, Platt scaling (used by probability=True) causes unstable models — accuracy drops to 40%.
- **Fix:** Use probability=False. For AUC use decision_function instead of predict_proba.

---

## Current Metrics (latest run)

### FF++ Validation (with class imbalance, not yet fixed)
| Metric | Value |
|---|---|
| Accuracy | 49.5% |
| Precision | 66.9% |
| Recall | 64.6% |
| F1-score | 65.7% |
| TN | 2 (almost all real misclassified as fake) |

### Celeb-DF v2 Cross-Dataset Test
| Metric | Value |
|---|---|
| Accuracy | 58.9% |
| Precision | 56.3% |
| Recall | 80.3% |
| F1-score | 66.2% |
| ROC-AUC | 60.1% |

### Target metrics (after fixing class imbalance)
| Dataset | Accuracy | F1 |
|---|---|---|
| FF++ validation | ~85%+ | ~87%+ |
| Celeb-DF test | ~63%+ | ~63%+ |

---

## Environment

- Python 3.13
- TensorFlow (with oneDNN — warnings about oneDNN are normal, not errors)
- mediapipe 0.10.32
- scikit-learn
- OpenCV (cv2)
- pandas, numpy
- joblib

---

## What NOT to do

- Do NOT run `extract_faces.py` without clearing `faces_dataset/` first — old files accumulate
- Do NOT use `gamma="auto"` in SVC with high-dimensional features
- Do NOT use `scoring="f1"` in GridSearchCV — leads to degenerate "predict all fake" solution
- Do NOT use `probability=True` in SVC with small datasets (<500 samples)
- Do NOT add Celeb-DF data to training — it is reserved for cross-dataset evaluation only (methodology constraint from the paper)
- Do NOT modify `val_split.csv` — it is used for in-dataset validation reporting
