"""
EfficientNet-B4 feature extraction utilities.

Output: 3584-dim vector per video = concat(mean, std) over frame embeddings.
  - mean (1792-dim): overall face appearance
  - std  (1792-dim): frame-to-frame inconsistency (key deepfake artefact)

BGR->RGB conversion is mandatory — OpenCV reads BGR, EfficientNet expects RGB.
"""

import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 380
_model   = None   # lazy singleton


def get_model() -> EfficientNetB4:
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
    """Load face image from disk, resize, BGR->RGB, EfficientNet preprocess."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    img = preprocess_input(img)
    return img


def extract_video_embedding(images: list[np.ndarray],
                            batch_size: int = 32) -> np.ndarray:
    """
    Given preprocessed face frames, return a 3584-dim video-level embedding:
        concat( mean(embeddings, axis=0), std(embeddings, axis=0) )

    The std component captures temporal inconsistency across frames,
    which is a characteristic artefact of deepfake synthesis.
    """
    model      = get_model()
    arr        = np.array(images)                                       # (N, 380, 380, 3)
    embeddings = model.predict(arr, batch_size=batch_size, verbose=0)  # (N, 1792)
    return np.concatenate([
        np.mean(embeddings, axis=0),   # (1792,)
        np.std(embeddings,  axis=0),   # (1792,)
    ])                                                                  # (3584,)
