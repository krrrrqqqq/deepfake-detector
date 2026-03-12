import cv2
import os
import pandas as pd
import numpy as np

# сколько кадров брать из каждого видео
NUM_FRAMES = 10

# создаём папку для кадров
OUTPUT_DIR = "extracted_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# загружаем train split
df = pd.read_csv("train_split.csv")

def extract_frames(video_path, output_folder, num_frames=10):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    saved_frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            saved_frames.append(frame)

    cap.release()
    return saved_frames


print("Starting frame extraction...")

for i, row in df.iterrows():
    video_path = row["video_path"]
    label = row["label"]

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    frames = extract_frames(video_path, video_output_folder, NUM_FRAMES)

    for j, frame in enumerate(frames):
        frame_path = os.path.join(video_output_folder, f"{video_name}_{j}.jpg")
        cv2.imwrite(frame_path, frame)

    if i % 50 == 0:
        print(f"Processed {i} videos")

print("Frame extraction completed.")