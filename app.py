"""
Deepfake Detector — Flask web application
==========================================
Inference: fine-tuned EfficientNet-B0 (224×224) trained on the combined
FF++ + Celeb-DF dataset.

Reads model_config.json (produced by finetune_combined.py) for:
  - img_size     — input resolution the model expects
  - threshold    — auto-tuned on the validation set
  - aggregation  — how per-frame probabilities are combined (default: median)

Per-frame probabilities are aggregated to a single video-level score and
compared against the threshold to produce REAL / FAKE.

A legacy SVM pipeline is kept as a last-resort fallback for when the
fine-tuned model is absent.
"""

import os
import json
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
CONFIG_PATH   = os.path.join(DATASET_DIR, "model_config.json")
MODEL_PATH    = os.path.join(DATASET_DIR, "efficientnet_combined.keras")
DETECTOR_PATH = os.path.join(DATASET_DIR, "blaze_face_short_range.tflite")

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_FRAMES        = 10
# Fixed uncertainty band decoupled from the decision threshold: scores below
# UNCERTAIN_LOW are confidently real, scores above UNCERTAIN_HIGH are
# confidently fake, anything in between is reported as UNCERTAIN with the raw
# fake-probability. Bounds chosen so the decision threshold (currently 0.79)
# falls inside the band — scores right at the boundary are maximally unsure.
UNCERTAIN_LOW     = 0.40
UNCERTAIN_HIGH    = 0.85
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# ── Load model_config.json (defaults if missing) ──────────────────────────────
IMG_SIZE    = 224
THRESHOLD   = 0.5
AGGREGATION = "median"

if os.path.isfile(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    IMG_SIZE    = cfg.get("img_size",    IMG_SIZE)
    THRESHOLD   = cfg.get("threshold",   THRESHOLD)
    AGGREGATION = cfg.get("aggregation", AGGREGATION)
    print(f"Loaded model_config.json: img_size={IMG_SIZE}, "
          f"threshold={THRESHOLD}, aggregation={AGGREGATION}")
else:
    print("WARNING: model_config.json not found — using defaults "
          f"(img_size={IMG_SIZE}, threshold={THRESHOLD}, aggregation={AGGREGATION})")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ── Load models once at startup ────────────────────────────────────────────────
print("Loading MediaPipe face detector...")
_base_opts = python.BaseOptions(model_asset_path=DETECTOR_PATH)
_det_opts  = vision.FaceDetectorOptions(
    base_options=_base_opts,
    min_detection_confidence=0.4,
)
detector = vision.FaceDetector.create_from_options(_det_opts)

# Primary: fine-tuned EfficientNet-B0 on combined dataset
finetuned_model = None
if os.path.isfile(MODEL_PATH):
    print(f"Loading fine-tuned model from {MODEL_PATH} ...")
    import tensorflow as tf
    finetuned_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded. Input shape: {finetuned_model.input_shape}")
else:
    print(f"Fine-tuned model not found at {MODEL_PATH} — will use SVM fallback")

# Fallback: legacy SVM pipeline (loaded only when needed)
svm = scaler = feature_model = efn_preprocess = None
USE_STD = False
if finetuned_model is None:
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras.applications.efficientnet import preprocess_input as efn_preprocess
    from sklearn.preprocessing import normalize  # noqa: F401  (used in predict_svm)
    print("Loading EfficientNet-B4 + SVM fallback pipeline...")
    feature_model = EfficientNetB4(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(380, 380, 3),
    )
    svm    = joblib.load(os.path.join(DATASET_DIR, "svm_model.pkl"))
    scaler = joblib.load(os.path.join(DATASET_DIR, "scaler.pkl"))
    USE_STD = (scaler.n_features_in_ == 3584)
    print(f"SVM input dim: {scaler.n_features_in_} "
          f"({'mean+std' if USE_STD else 'mean only'})")

print("All models loaded. Ready.\n")


# ── Face cropping ──────────────────────────────────────────────────────────────

def crop_face(frame: np.ndarray) -> np.ndarray:
    """
    Detect highest-confidence face and return it with 20% padding.
    Falls back to a central crop when no face is detected.
    """
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

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

    margin = int(min(h, w) * 0.2)
    size   = min(h, w) - 2 * margin
    cy, cx = h // 2, w // 2
    half   = size // 2
    return frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half]


def prepare_face_for_model(frame: np.ndarray) -> np.ndarray | None:
    """
    For fine-tuned EfficientNet-B0: BGR→RGB, resize to IMG_SIZE,
    return float32 in [0, 255]. EfficientNetB0 has
    include_preprocessing=True, so it does ImageNet normalisation
    internally — pre-normalising here would double-normalise.
    """
    face = crop_face(frame)
    if face is None or face.size == 0:
        return None
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face.astype("float32")


def prepare_face_for_svm(frame: np.ndarray) -> np.ndarray | None:
    """For legacy SVM: 380×380 with EfficientNet-B4 ImageNet preprocessing."""
    face = crop_face(frame)
    if face is None or face.size == 0:
        return None
    face = cv2.resize(face, (380, 380))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return efn_preprocess(face.astype("float32"))


# ── Frame collection ──────────────────────────────────────────────────────────

def collect_faces(path: str, prep_fn) -> list[np.ndarray]:
    """Collect prepared face arrays from an image or video file."""
    ext = os.path.splitext(path)[1].lower()

    if ext in ALLOWED_IMAGE_EXT:
        img = cv2.imread(path)
        if img is None:
            return []
        f = prep_fn(img)
        return [f] if f is not None else []

    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = (
        list(range(total))
        if total < NUM_FRAMES
        else np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    )

    faces = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        f = prep_fn(frame)
        if f is not None:
            faces.append(f)
    cap.release()
    return faces


# ── Inference ─────────────────────────────────────────────────────────────────

def aggregate(probs: np.ndarray, method: str) -> float:
    if method == "mean":      return float(np.mean(probs))
    if method == "max":       return float(np.max(probs))
    if method == "top3_mean":
        k = min(3, len(probs))
        return float(np.mean(np.sort(probs)[-k:]))
    return float(np.median(probs))   # default + "median"


def predict_finetuned(faces: list[np.ndarray]) -> tuple[str, float]:
    """Per-frame inference → aggregate → REAL / UNCERTAIN / FAKE verdict."""
    arr   = np.array(faces, dtype="float32")
    probs = finetuned_model.predict(arr, batch_size=16, verbose=0).flatten()
    score = aggregate(probs, AGGREGATION)

    # In the uncertainty band we surface the raw fake-probability so the UI
    # can show which side the score leans toward, even though no hard label
    # is claimed.
    if UNCERTAIN_LOW <= score <= UNCERTAIN_HIGH:
        return "UNCERTAIN", round(score * 100, 1)

    if score > UNCERTAIN_HIGH:
        label      = "FAKE"
        confidence = 50.0 + (score - UNCERTAIN_HIGH) / max(1.0 - UNCERTAIN_HIGH, 1e-6) * 50.0
    else:
        label      = "REAL"
        confidence = 50.0 + (UNCERTAIN_LOW - score) / max(UNCERTAIN_LOW, 1e-6) * 50.0

    return label, round(confidence, 1)


def predict_svm(faces: list[np.ndarray]) -> tuple[str, float]:
    """SVM pipeline inference (fallback when fine-tuned model is absent)."""
    from sklearn.preprocessing import normalize

    arr        = np.array(faces, dtype="float32")
    embeddings = feature_model.predict(arr, batch_size=32, verbose=0)

    if USE_STD:
        video_emb = np.concatenate([np.mean(embeddings, axis=0),
                                    np.std(embeddings,  axis=0)])
    else:
        video_emb = np.mean(embeddings, axis=0)

    X = video_emb.reshape(1, -1)
    X = scaler.transform(X)
    X = normalize(X)

    prediction = svm.predict(X)[0]
    margin     = float(svm.decision_function(X)[0])
    confidence = round(100.0 / (1.0 + np.exp(-abs(margin))), 1)
    label      = "FAKE" if prediction == 1 else "REAL"
    return label, confidence


# ── Routes ────────────────────────────────────────────────────────────────────

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

    filename = secure_filename(file.filename)
    ext      = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT:
        return render_template("index.html", result="Error",
                               confidence=f"Unsupported file type: {ext}")

    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        prep_fn = (prepare_face_for_model if finetuned_model is not None
                   else prepare_face_for_svm)
        faces = collect_faces(save_path, prep_fn)

        if not faces:
            return render_template(
                "index.html", result="Error",
                confidence="Could not extract a face from the uploaded file."
            )

        if finetuned_model is not None:
            result, confidence = predict_finetuned(faces)
        else:
            result, confidence = predict_svm(faces)

        return render_template("index.html", result=result, confidence=confidence)

    except Exception as exc:
        return render_template("index.html", result="Error", confidence=str(exc))
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
