import cv2
import os
import numpy as np

# сколько кадров извлекаем из видео
NUM_FRAMES = 10

# путь к видео
DATASET_PATH = "celebdf_subset"

# куда сохраняем лица
OUTPUT_PATH = "celebdf_faces"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# загрузка Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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


def detect_and_crop_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # выбираем самое большое лицо
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = frame[y:y+h, x:x+w]

    return face


print("Starting CelebDF face extraction...")

video_count = 0

for label in ["real", "fake"]:

    video_dir = os.path.join(DATASET_PATH, label)

    save_dir = os.path.join(OUTPUT_PATH, label)
    os.makedirs(save_dir, exist_ok=True)

    for video_name in os.listdir(video_dir):

        video_path = os.path.join(video_dir, video_name)

        frames = extract_frames(video_path, NUM_FRAMES)

        for i, frame in enumerate(frames):

            face = detect_and_crop_face(frame)

            if face is None:
                continue

            # resize для EfficientNet-B4
            face = cv2.resize(face, (380, 380))

            filename = f"{video_name}_{i}.jpg"

            save_path = os.path.join(save_dir, filename)

            cv2.imwrite(save_path, face)

        video_count += 1

        if video_count % 50 == 0:
            print(f"Processed {video_count} videos")

print("CelebDF face extraction completed.")