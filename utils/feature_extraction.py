"""
EfficientNet-B4 feature extraction utilities shared by
extract_features.py, extract_features_celebdf.py, and app.py.

Key design decisions (from CLAUDE.md):
  - Model  : EfficientNetB4, weights=imagenet, include_top=False, pooling=avg
  - Input  : (380, 380, 3) — EfficientNet-B4 native resolution
  - Output : one 1792-dim vector per VIDEO (mean pooling across frames)
  - BGR→RGB: mandatory before preprocess_input — OpenCV reads BGR, EfficientNet expects RGB
"""

import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 380
_model   = None  # lazy singleton — loaded on first call to get_model()


def get_model() -> EfficientNetB4:
    """Return the shared EfficientNet-B4 feature extractor (lazy init)."""
    global _model
    if _model is None:
        _model = EfficientNetB4(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
    return _model


def load_and_preprocess_image(path: str) -> np.ndarray | None:
    """
    Load a face image from disk, resize, convert BGR→RGB, and apply
    EfficientNet-specific preprocessing.

    Returns None if the image cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR→RGB is mandatory
    img = img.astype("float32")
    img = preprocess_input(img)
    return img


def extract_video_embedding(images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    """
    Given a list of preprocessed face images (each shape (IMG_SIZE, IMG_SIZE, 3)),
    return a single 1792-dim video-level embedding via mean pooling.

    This is the canonical aggregation used in both FF++ and Celeb-DF pipelines.
    Never classify individual frames and majority-vote — the SVM expects one vector
    per video.
    """
    model      = get_model()
    arr        = np.array(images)                                   # (N, 380, 380, 3)
    embeddings = model.predict(arr, batch_size=batch_size, verbose=0)  # (N, 1792)
    return np.mean(embeddings, axis=0)                              # (1792,)
