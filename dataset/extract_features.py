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

    for file in os.listdir(folder):
        # Имена файлов: Method__video.mp4_0.jpg
        # video_id = всё до последнего '_'
        video_id = "_".join(file.split("_")[:-1])
        videos[video_id].append(file)

    print(f"  Found {len(videos)} unique videos")

    for video_id, frames in tqdm(videos.items()):
        frames = sorted(frames)[:FRAMES_PER_VIDEO]
        images = []

        for frame in frames:
            img = load_image(os.path.join(folder, frame))
            if img is not None:
                images.append(img)

        if len(images) == 0:
            continue

        images = np.array(images)
        embeddings = model.predict(images, batch_size=BATCH_SIZE, verbose=0)

        # Level 2: concat mean + std across frames (3584-dim).
        # std captures frame-to-frame inconsistency — a key deepfake artefact.
        video_embedding = np.concatenate([
            np.mean(embeddings, axis=0),   # 1792-dim: average appearance
            np.std(embeddings,  axis=0),   # 1792-dim: temporal flicker / artifacts
        ])

        X.append(video_embedding)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("\nFinal dataset shape:")
print("X:", X.shape, "  (1792 mean + 1792 std = 3584-dim per video)")
print(f"Real: {np.sum(y==0)}, Fake: {np.sum(y==1)}")

np.save("X_train.npy", X)
np.save("y_train.npy", y)

print("\nFeature extraction completed.")