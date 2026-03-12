import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import cv2

# Load pre-trained EfficientNet
base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    features = base_model.predict(img)
    return features