import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

DATASET_PATH = "faces_dataset"

IMG_SIZE = 380
FRAMES_PER_VIDEO = 10
BATCH_SIZE = 32

print("Loading EfficientNet-B4...")

model = EfficientNetB4(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    img = preprocess_input(img)
    return img


X = []
y = []

for label_name in ["real", "fake"]:

    label = 0 if label_name == "real" else 1
    folder = os.path.join(DATASET_PATH, label_name)

    print(f"\nProcessing {label_name}...")

    videos = defaultdict(list)

    # ИСПРАВЛЕНИЕ: группировка по video_id
    # Примеры имён файлов:
    #   real: '004.mp4_0.jpg'      -> video_id = '004.mp4'
    #   fake: '004_982.mp4_0.jpg'  -> video_id = '004_982.mp4'
    # Берём всё до последнего '_' (последний сегмент — номер кадра)
    for file in os.listdir(folder):
        video_id = "_".join(file.split("_")[:-1])
        videos[video_id].append(file)

    print(f"  Found {len(videos)} videos")

    for video_id, frames in tqdm(videos.items()):

        frames = sorted(frames)[:FRAMES_PER_VIDEO]

        images = []

        for frame in frames:
            img_path = os.path.join(folder, frame)
            img = load_image(img_path)
            if img is not None:
                images.append(img)

        if len(images) == 0:
            continue

        images = np.array(images)

        embeddings = model.predict(images, batch_size=BATCH_SIZE, verbose=0)

        video_embedding = np.mean(embeddings, axis=0)

        X.append(video_embedding)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("\nFinal dataset shape:")
print("X:", X.shape)
print("y:", y.shape)
print(f"Real: {sum(1 for v in y if v==0)}, Fake: {sum(1 for v in y if v==1)}")

np.save("X_train.npy", X)
np.save("y_train.npy", y)

print("\nFeature extraction completed.")