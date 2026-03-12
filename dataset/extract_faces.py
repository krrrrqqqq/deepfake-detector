import cv2
import os
import pandas as pd
import numpy as np

NUM_FRAMES = 10
OUTPUT_DIR = "faces_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("train_split.csv")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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


def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Попытка 1: стандартные параметры
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )

    # Попытка 2: ещё мягче если не нашли
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=1, minSize=(20, 20)
        )

    # Если совсем не нашли — пропускаем кадр (NO fallback)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # padding 20%
    pad = int(max(w, h) * 0.2)
    H, W = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    face = frame[y1:y2, x1:x2]
    return face


print("Starting face extraction...")

detected = 0
missed = 0

for i, row in df.iterrows():
    video_path = row["video_path"]
    label = row["label"]

    label_folder = "real" if label == 0 else "fake"
    save_folder = os.path.join(OUTPUT_DIR, label_folder)
    os.makedirs(save_folder, exist_ok=True)

    frames = extract_frames(video_path, NUM_FRAMES)

    for j, frame in enumerate(frames):
        face = detect_and_crop_face(frame)
        if face is not None and face.size > 0:
            face = cv2.resize(face, (380, 380))
            filename = f"{os.path.basename(video_path)}_{j}.jpg"
            cv2.imwrite(os.path.join(save_folder, filename), face)
            detected += 1
        else:
            missed += 1

    if i % 100 == 0:
        print(f"Processed {i}/{len(df)} | Detected: {detected} | Missed: {missed}")

print("\nDone.")
print(f"Detected: {detected}, Missed: {missed}")
real_count = len(os.listdir(os.path.join(OUTPUT_DIR, "real")))
fake_count = len(os.listdir(os.path.join(OUTPUT_DIR, "fake")))
print(f"faces_dataset/real: {real_count}")
print(f"faces_dataset/fake: {fake_count}")