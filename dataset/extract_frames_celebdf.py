import cv2
import os
import numpy as np

# ============================================================
# ИСПРАВЛЕНИЯ:
# 1. Теперь берём ровно NUM_FRAMES кадров из каждого видео
#    (равномерная выборка, как в FF++ пайплайне)
# 2. Убрана зависимость от длины видео (каждый 30-й кадр)
# ============================================================

DATASET_PATH = "celebdf_subset"
OUTPUT_PATH = "celebdf_frames"
NUM_FRAMES = 10  # Должно совпадать с FF++ пайплайном

os.makedirs(OUTPUT_PATH, exist_ok=True)

for label in ["real", "fake"]:

    video_dir = os.path.join(DATASET_PATH, label)
    output_dir = os.path.join(OUTPUT_PATH, label)

    os.makedirs(output_dir, exist_ok=True)

    for video_name in os.listdir(video_dir):

        if not video_name.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            continue

        # Равномерная выборка NUM_FRAMES кадров — как в FF++ пайплайне
        if total_frames < NUM_FRAMES:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

        saved = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                video_base = os.path.splitext(video_name)[0]
                frame_name = f"{video_base}_{saved}.jpg"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                saved += 1

        cap.release()

print("Frame extraction done.")