import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================
# ИСПРАВЛЕНИЯ:
# 1. Добавлена BGR→RGB конвертация (отсутствовала — критическая ошибка)
# 2. Теперь эмбеддинги усредняются на уровне видео (как в FF++ пайплайне)
#    Раньше каждый кадр был отдельным семплом — несоответствие с обучением
# 3. Добавлен параметр FRAMES_PER_VIDEO для контроля
# ============================================================

DATASET_PATH = "celebdf_faces"
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

    # ИСПРАВЛЕНИЕ: конвертация BGR→RGB (было пропущено в оригинале)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype("float32")
    img = preprocess_input(img)

    return img


X = []
y = []

print("Starting feature extraction for Celeb-DF (video-level)...")

for label_name, label in [("real", 0), ("fake", 1)]:

    folder = os.path.join(DATASET_PATH, label_name)
    print(f"\nProcessing {label_name}...")

    # ИСПРАВЛЕНИЕ: группировка кадров по видео — как в FF++ пайплайне
    videos = defaultdict(list)

    for file in os.listdir(folder):
        if not file.lower().endswith((".jpg", ".png")):
            continue
        # Имя файла: {video_id}_{frame_idx}.jpg
        video_id = "_".join(file.split("_")[:-1])
        videos[video_id].append(file)

    print(f"  Found {len(videos)} videos")

    for video_id, frames in tqdm(videos.items()):

        # Берём не более FRAMES_PER_VIDEO кадров
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

        # Batch inference
        embeddings = model.predict(images, batch_size=BATCH_SIZE, verbose=0)

        # Level 2: concat mean + std (must match extract_features.py exactly).
        # std captures temporal inconsistency — a key deepfake artefact.
        video_embedding = np.concatenate([
            np.mean(embeddings, axis=0),   # 1792-dim
            np.std(embeddings,  axis=0),   # 1792-dim
        ])

        X.append(video_embedding)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("\nFinal test dataset shape:")
print("X:", X.shape, "  (1792 mean + 1792 std = 3584-dim per video)")
print("y:", y.shape)
print(f"Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}")

np.save("X_test.npy", X)
np.save("y_test.npy", y)

print("\nFeature extraction completed.")