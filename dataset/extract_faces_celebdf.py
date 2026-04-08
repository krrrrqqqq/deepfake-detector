import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import shutil

# Number of frames to sample per video — must match FF++ pipeline
NUM_FRAMES = 10

DATASET_PATH = "celebdf_subset"
OUTPUT_PATH = "celebdf_faces"
MODEL_PATH = "blaze_face_short_range.tflite"

# MediaPipe 0.10+ API (replaces deprecated mp.solutions.face_detection)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.4
)
detector = vision.FaceDetector.create_from_options(options)


def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= num_frames:
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
    """Detect and crop the highest-confidence face. Falls back to centre crop."""
    h, w = frame.shape[:2]
    # MediaPipe requires RGB input
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.detections:
        best = max(result.detections, key=lambda d: d.categories[0].score)
        bbox = best.bounding_box
        # 20% padding — identical to extract_faces.py so train and test faces match
        pad_x = int(bbox.width * 0.2)
        pad_y = int(bbox.height * 0.2)
        x1 = max(0, bbox.origin_x - pad_x)
        y1 = max(0, bbox.origin_y - pad_y)
        x2 = min(w, bbox.origin_x + bbox.width + pad_x)
        y2 = min(h, bbox.origin_y + bbox.height + pad_y)
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            return face

    # Fallback: central crop so every video still produces at least one embedding
    margin = int(min(h, w) * 0.2)
    size = min(h, w) - 2 * margin
    cy, cx = h // 2, w // 2
    half = size // 2
    return frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half]


print("Starting CelebDF face extraction...")

# Clear output directories before running to avoid stale data accumulation
for folder in ["real", "fake"]:
    path = os.path.join(OUTPUT_PATH, folder)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

detected = 0
total = 0
video_count = 0

for label in ["real", "fake"]:
    video_dir = os.path.join(DATASET_PATH, label)
    save_dir = os.path.join(OUTPUT_PATH, label)

    for video_name in os.listdir(video_dir):
        if not video_name.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, video_name)
        frames = extract_frames(video_path, NUM_FRAMES)
        video_base = os.path.splitext(video_name)[0]

        for j, frame in enumerate(frames):
            face = crop_face(frame)
            total += 1

            if face is not None and face.size > 0:
                face = cv2.resize(face, (380, 380))
                filename = f"{video_base}_{j}.jpg"
                cv2.imwrite(os.path.join(save_dir, filename), face)
                detected += 1

        video_count += 1
        if video_count % 50 == 0:
            print(f"Processed {video_count} videos | Saved: {detected}/{total}")

rate = detected / total * 100 if total > 0 else 0
print(f"\nDone. Saved: {detected}/{total} ({rate:.1f}%)")

real_count = len(os.listdir(os.path.join(OUTPUT_PATH, "real")))
fake_count = len(os.listdir(os.path.join(OUTPUT_PATH, "fake")))
print(f"celebdf_faces/real: {real_count}")
print(f"celebdf_faces/fake: {fake_count}")
