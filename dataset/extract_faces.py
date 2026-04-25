import cv2
import os
import shutil
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NUM_FRAMES = 10
OUTPUT_DIR = "faces_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("dataset_labels.csv")

MODEL_PATH = "blaze_face_short_range.tflite"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.4
)
detector = vision.FaceDetector.create_from_options(options)


def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def crop_face(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.detections:
        best = max(result.detections, key=lambda d: d.categories[0].score)
        bbox = best.bounding_box
        pad_x = int(bbox.width * 0.2)
        pad_y = int(bbox.height * 0.2)
        x1 = max(0, bbox.origin_x - pad_x)
        y1 = max(0, bbox.origin_y - pad_y)
        x2 = min(w, bbox.origin_x + bbox.width + pad_x)
        y2 = min(h, bbox.origin_y + bbox.height + pad_y)
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            return face

    # Fallback: центральный кроп
    margin = int(min(h, w) * 0.2)
    size = min(h, w) - 2 * margin
    cy, cx = h // 2, w // 2
    half = size // 2
    return frame[max(0, cy-half):cy+half, max(0, cx-half):cx+half]


print("Starting face extraction...")

# Очищаем папки перед запуском
for folder in ["real", "fake"]:
    path = os.path.join(OUTPUT_DIR, folder)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

detected = 0
total = 0

for i, row in df.iterrows():
    video_path = row["video_path"]
    label = row["label"]

    label_folder = "real" if label == 0 else "fake"
    save_folder = os.path.join(OUTPUT_DIR, label_folder)

    frames = extract_frames(video_path, NUM_FRAMES)

    # ИСПРАВЛЕНИЕ: уникальное имя = метод манипуляции + имя файла
    # Например: Deepfakes__107_109.mp4_0.jpg
    # Это решает проблему дублирования одинаковых имён из разных папок
    method = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(video_path))))
    video_basename = os.path.basename(video_path)
    unique_prefix = f"{method}__{video_basename}"

    for j, frame in enumerate(frames):
        face = crop_face(frame)
        total += 1

        if face is not None and face.size > 0:
            face = cv2.resize(face, (380, 380))
            filename = f"{unique_prefix}_{j}.jpg"
            cv2.imwrite(os.path.join(save_folder, filename), face)
            detected += 1

    if i % 100 == 0:
        print(f"Processed {i}/{len(df)} | Saved: {detected}/{total}")

print(f"\nDone. Saved: {detected}/{total} ({detected/total*100:.1f}%)")

real_count = len(os.listdir(os.path.join(OUTPUT_DIR, "real")))
fake_count = len(os.listdir(os.path.join(OUTPUT_DIR, "fake")))
print(f"faces_dataset/real: {real_count}")
print(f"faces_dataset/fake: {fake_count}")

# Проверяем уникальность
files = os.listdir(os.path.join(OUTPUT_DIR, "fake"))
video_ids = set("_".join(f.split("_")[:-1]) for f in files)
print(f"Unique fake videos: {len(video_ids)} (should be ~720)")