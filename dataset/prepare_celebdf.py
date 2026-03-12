import os
import random
import shutil

BASE_PATH = "Celeb-DF-v2"
OUTPUT_PATH = "celebdf_subset"

REAL_PATHS = [
    os.path.join(BASE_PATH, "Celeb-real"),
    os.path.join(BASE_PATH, "YouTube-real")
]

FAKE_PATH = os.path.join(BASE_PATH, "Celeb-synthesis")

os.makedirs(os.path.join(OUTPUT_PATH, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "fake"), exist_ok=True)

# собираем real видео
real_videos = []

for p in REAL_PATHS:
    for f in os.listdir(p):
        if f.endswith(".mp4"):
            real_videos.append(os.path.join(p, f))

fake_videos = [
    os.path.join(FAKE_PATH, f)
    for f in os.listdir(FAKE_PATH)
    if f.endswith(".mp4")
]

random.shuffle(real_videos)
random.shuffle(fake_videos)

real_videos = real_videos[:300]
fake_videos = fake_videos[:300]

print("Copying real videos...")

for v in real_videos:
    shutil.copy(v, os.path.join(OUTPUT_PATH, "real"))

print("Copying fake videos...")

for v in fake_videos:
    shutil.copy(v, os.path.join(OUTPUT_PATH, "fake"))

print("Done.")