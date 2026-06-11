"""
Детектор дипфейков — веб-приложение на Flask
=============================================
Инференс: дообученная EfficientNet-B0 (224×224), обученная на объединённом
датасете FF++ + Celeb-DF.

Читает model_config.json (создаётся finetune_combined.py) для получения:
  - img_size     — входное разрешение, которое ожидает модель
  - threshold    — порог, автоматически подобранный на валидационной выборке
  - aggregation  — способ свёртки покадровых вероятностей (по умолчанию: median)

Покадровые вероятности агрегируются в единый score уровня видео и
сравниваются с порогом для получения вердикта REAL / FAKE.

Legacy-конвейер на SVM сохранён как резервный путь на случай отсутствия
дообученной модели.
"""

import os
import json
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ── Пути ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
CONFIG_PATH   = os.path.join(DATASET_DIR, "model_config.json")
MODEL_PATH    = os.path.join(DATASET_DIR, "efficientnet_combined.keras")
DETECTOR_PATH = os.path.join(DATASET_DIR, "blaze_face_short_range.tflite")

# ── Константы ────────────────────────────────────────────────────────────────
NUM_FRAMES        = 10
# Фиксированная полоса неопределённости, отвязанная от порога решения: score
# ниже UNCERTAIN_LOW — уверенно настоящие, выше UNCERTAIN_HIGH — уверенно
# фейк, всё, что между — выдаётся как UNCERTAIN с сырой fake-вероятностью.
# Границы выбраны так, что порог решения (сейчас 0.79) попадает внутрь полосы —
# score прямо на границе максимально неоднозначен.
UNCERTAIN_LOW     = 0.40
UNCERTAIN_HIGH    = 0.85
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# ── Загрузка model_config.json (значения по умолчанию при отсутствии) ─────────
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
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 МБ

# ── Загрузка моделей один раз при старте ──────────────────────────────────────
print("Loading MediaPipe face detector...")
_base_opts = python.BaseOptions(model_asset_path=DETECTOR_PATH)
# min_detection_confidence=0.2 (а не 0.4): артефакты дипфейка занижают
# уверенность BlazeFace на манипулированных лицах. На FF++ Deepfakes клипах
# лицо ведущего крупное и явно присутствует, но получает оценку 0.1–0.4 —
# поэтому порог 0.4 отбрасывал настоящие лица как «лицо не найдено». 0.2
# возвращает их, оставаясь достаточно строгим, чтобы по-настоящему
# безлицевой контент (пейзаж, документ) по-прежнему давал ноль детекций и
# отклонялся выше по конвейеру.
_det_opts  = vision.FaceDetectorOptions(
    base_options=_base_opts,
    min_detection_confidence=0.2,
)
detector = vision.FaceDetector.create_from_options(_det_opts)

# Основная модель: дообученная EfficientNet-B0 на объединённом датасете
finetuned_model = None
if os.path.isfile(MODEL_PATH):
    print(f"Loading fine-tuned model from {MODEL_PATH} ...")
    import tensorflow as tf
    finetuned_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded. Input shape: {finetuned_model.input_shape}")
else:
    print(f"Fine-tuned model not found at {MODEL_PATH} — will use SVM fallback")

# Резервный путь: legacy-конвейер на SVM (загружается только при необходимости)
svm = scaler = feature_model = efn_preprocess = None
USE_STD = False
if finetuned_model is None:
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras.applications.efficientnet import preprocess_input as efn_preprocess
    from sklearn.preprocessing import normalize  # noqa: F401  (используется в predict_svm)
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


# ── Вырезание лица ────────────────────────────────────────────────────────────

def crop_face(frame: np.ndarray) -> np.ndarray | None:
    """
    Находит лицо с наибольшей уверенностью и возвращает его с запасом 20%.
    Возвращает None, если лицо не обнаружено. Кадры без лица пропускаются
    в collect_faces; если детекция провалилась на всех сэмплированных кадрах,
    запрос отклоняется выше по конвейеру — прогон инференса по не-лицевому
    контенту (пейзаж, документ, животное) даёт бессмысленный score.
    """
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.detections:
        return None

    best  = max(result.detections, key=lambda d: d.categories[0].score)
    bbox  = best.bounding_box
    pad_x = int(bbox.width  * 0.2)
    pad_y = int(bbox.height * 0.2)
    x1 = max(0, bbox.origin_x - pad_x)
    y1 = max(0, bbox.origin_y - pad_y)
    x2 = min(w, bbox.origin_x + bbox.width  + pad_x)
    y2 = min(h, bbox.origin_y + bbox.height + pad_y)
    face = frame[y1:y2, x1:x2]
    return face if face.size > 0 else None


def prepare_face_for_model(frame: np.ndarray) -> np.ndarray | None:
    """
    Для дообученной EfficientNet-B0: BGR→RGB, ресайз до IMG_SIZE,
    возврат float32 в диапазоне [0, 255]. У EfficientNetB0 включён
    include_preprocessing=True, поэтому ImageNet-нормализация делается
    внутри модели — нормализация здесь привела бы к двойной нормализации.
    """
    face = crop_face(frame)
    if face is None or face.size == 0:
        return None
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face.astype("float32")


def prepare_face_for_svm(frame: np.ndarray) -> np.ndarray | None:
    """Для legacy-SVM: 380×380 с ImageNet-препроцессингом EfficientNet-B4."""
    face = crop_face(frame)
    if face is None or face.size == 0:
        return None
    face = cv2.resize(face, (380, 380))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return efn_preprocess(face.astype("float32"))


# ── Сбор кадров ───────────────────────────────────────────────────────────────

def collect_faces(path: str, prep_fn) -> list[np.ndarray]:
    """Собирает подготовленные массивы лиц из файла изображения или видео."""
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


# ── Инференс ──────────────────────────────────────────────────────────────────

def aggregate(probs: np.ndarray, method: str) -> float:
    if method == "mean":      return float(np.mean(probs))
    if method == "max":       return float(np.max(probs))
    if method == "top3_mean":
        k = min(3, len(probs))
        return float(np.mean(np.sort(probs)[-k:]))
    return float(np.median(probs))   # по умолчанию + "median"


def predict_finetuned(faces: list[np.ndarray]) -> tuple[str, float]:
    """Покадровый инференс → агрегация → вердикт REAL / UNCERTAIN / FAKE."""
    arr   = np.array(faces, dtype="float32")
    probs = finetuned_model.predict(arr, batch_size=16, verbose=0).flatten()
    score = aggregate(probs, AGGREGATION)

    # Внутри полосы неопределённости показываем сырую fake-вероятность, чтобы
    # UI мог отобразить, в какую сторону клонится score, даже когда жёсткая
    # метка не выносится.
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
    """Инференс конвейера на SVM (резерв при отсутствии дообученной модели)."""
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


# ── Маршруты ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def _err(message: str):
    """Единый JSON-формат ошибки для AJAX-фронтенда."""
    return jsonify(status="error", message=message)


@app.route("/predict", methods=["POST"])
def predict():
    """Анализирует загруженное видео и возвращает вердикт в виде JSON.

    Браузер отправляет запрос через fetch() и отрисовывает результат на месте
    без перезагрузки страницы. Сервер удаляет свою копию файла после инференса.
    """
    if "file" not in request.files:
        return _err("No file field in request."), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return _err("No file selected."), 400

    # Берём расширение из ОРИГИНАЛЬНОГО имени. secure_filename() вырезает все
    # не-ASCII символы, поэтому кириллическое имя вроде "видео.mp4" схлопывается
    # в "mp4" и полностью теряет расширение — из-за чего проверка формата
    # ошибочно провалилась бы ещё до поиска лица. safe_name используется только
    # для пути на диске.
    display_name = file.filename
    ext          = os.path.splitext(display_name)[1].lower()

    # Этот детектор работает с видео: модель покадровая, а вердикт уровня видео
    # получается медианной агрегацией покадровых вероятностей. Одиночные
    # изображения отклоняются — у одного кадра нет агрегации, и он выходит за
    # рамки FF++/Celeb-DF видео-дипфейков, на которых модель обучалась и
    # оценивалась.
    if ext in ALLOWED_IMAGE_EXT:
        return _err("Image input is not supported — this detector is "
                    "video-based. Please upload a video file (MP4, MOV, AVI, "
                    "MKV)."), 415

    if ext not in ALLOWED_VIDEO_EXT:
        return _err(f"Unsupported file type: {ext or 'no extension'}. "
                    "Please upload a video file (MP4, MOV, AVI, MKV)."), 415

    # Строим безопасное имя для диска; заново прикрепляем проверенное расширение,
    # если secure_filename его срезал (полностью не-ASCII базовое имя).
    safe_name = secure_filename(display_name) or "upload"
    if not safe_name.lower().endswith(ext):
        safe_name += ext
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(save_path)

    try:
        prep_fn = (prepare_face_for_model if finetuned_model is not None
                   else prepare_face_for_svm)
        faces = collect_faces(save_path, prep_fn)

        if not faces:
            return _err("No face detected in the video. MediaPipe found no "
                        "face in any of the sampled frames, so there is "
                        "nothing to analyse. Please upload a video with a "
                        "clearly visible human face.")

        if finetuned_model is not None:
            result, confidence = predict_finetuned(faces)
        else:
            result, confidence = predict_svm(faces)

        return jsonify(status="ok", result=result, confidence=confidence)

    except Exception as exc:
        return _err(str(exc)), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
