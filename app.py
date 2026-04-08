"""
Deepfake Detector — Flask web application
Accepts an image or short video, runs face detection + EfficientNet-B4
feature extraction, applies the trained SVM, and returns REAL / FAKE.
"""

import os
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, request
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import normalize
from werkzeug.utils import secure_filename

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
IMG_SIZE      = 380
NUM_FRAMES    = 10
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

# ── Load models once at startup ────────────────────────────────────────────────
print("Loading EfficientNet-B4 feature extractor...")
feature_model = EfficientNetB4(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

print("Loading SVM classifier and StandardScaler...")
svm    = joblib.load(os.path.join(DATASET_DIR, "svm_model.pkl"))
scaler = joblib.load(os.path.join(DATASET_DIR, "scaler.pkl"))

print("Loading MediaPipe face detector...")
_base_opts = python.BaseOptions(
    model_asset_path=os.path.join(DATASET_DIR, "blaze_face_short_range.tflite")
)
_det_opts = vision.FaceDetectorOptions(
    base_options=_base_opts,
    min_detection_confidence=0.4,
)
detector = vision.FaceDetector.create_from_options(_det_opts)
print("All models loaded. Ready.")


# ── Helper functions ───────────────────────────────────────────────────────────

def crop_face(frame: np.ndarray) -> np.ndarray:
    """
    Detect the highest-confidence face and return it with 20% padding.
    Falls back to a centre crop when no face is detected so every input
    still produces an embedding (matching the training pipeline's fallback).
    """
    h, w = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if result.detections:
        best  = max(result.detections, key=lambda d: d.categories[0].score)
        bbox  = best.bounding_box
        pad_x = int(bbox.width  * 0.2)
        pad_y = int(bbox.height * 0.2)
        x1 = max(0, bbox.origin_x - pad_x)
        y1 = max(0, bbox.origin_y - pad_y)
        x2 = min(w, bbox.origin_x + bbox.width  + pad_x)
        y2 = min(h, bbox.origin_y + bbox.height + pad_y)
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            return face

    # Fallback: central crop (same as training pipeline)
    margin = int(min(h, w) * 0.2)
    size   = min(h, w) - 2 * margin
    cy, cx = h // 2, w // 2
    half   = size // 2
    return frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half]


def prepare_face(frame: np.ndarray) -> np.ndarray | None:
    """Crop → resize → BGR→RGB → preprocess for EfficientNet."""
    face = crop_face(frame)
    if face is None or face.size == 0:
        return None
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")
    face = preprocess_input(face)
    return face


def embed_frames(images: list[np.ndarray]) -> np.ndarray:
    """
    Run EfficientNet-B4 on a list of preprocessed face arrays and
    return a single 1792-dim video-level embedding (mean pooling).
    This exactly mirrors extract_features.py and extract_features_celebdf.py.
    """
    arr        = np.array(images)                               # (N, 380, 380, 3)
    embeddings = feature_model.predict(arr, batch_size=32, verbose=0)  # (N, 1792)
    return np.mean(embeddings, axis=0)                          # (1792,)


def extract_image_embedding(path: str) -> np.ndarray | None:
    img  = cv2.imread(path)
    if img is None:
        return None
    face = prepare_face(img)
    if face is None:
        return None
    return embed_frames([face])


def extract_video_embedding(path: str) -> np.ndarray | None:
    cap          = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    if total_frames < NUM_FRAMES:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    images = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        face = prepare_face(frame)
        if face is not None:
            images.append(face)
    cap.release()

    if not images:
        return None
    return embed_frames(images)


def predict_from_embedding(embedding: np.ndarray) -> tuple[str, float]:
    """
    Apply the same preprocessing chain used in train_svm.py and test_celebdf.py:
        1. scaler.transform  (StandardScaler — fit on training data only)
        2. normalize         (L2 normalisation)
    Then classify with the SVM. Confidence is derived from decision_function
    (probability=False avoids Platt scaling instability on small datasets).
    """
    X = embedding.reshape(1, -1)
    X = scaler.transform(X)
    X = normalize(X)

    prediction = svm.predict(X)[0]
    score      = float(svm.decision_function(X)[0])

    # Sigmoid maps the raw margin to a 0–100 % confidence value
    confidence = round(100.0 / (1.0 + np.exp(-abs(score))), 1)
    label      = "FAKE" if prediction == 1 else "REAL"
    return label, confidence


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result="Error",
                               confidence="No file field in request.")

    file = request.files["file"]
    if not file or file.filename == "":
        return render_template("index.html", result="Error",
                               confidence="No file selected.")

    filename = secure_filename(file.filename)  # guard against path traversal
    ext      = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT:
        return render_template("index.html", result="Error",
                               confidence=f"Unsupported file type: {ext}")

    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        if ext in ALLOWED_VIDEO_EXT:
            embedding = extract_video_embedding(save_path)
        else:
            embedding = extract_image_embedding(save_path)

        if embedding is None:
            return render_template("index.html", result="Error",
                                   confidence="Could not extract a face from the uploaded file.")

        result, confidence = predict_from_embedding(embedding)
        return render_template("index.html", result=result, confidence=confidence)

    except Exception as exc:
        return render_template("index.html", result="Error", confidence=str(exc))
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
