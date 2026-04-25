import os
import random
import shutil

# Take ALL available real videos (Celeb-real + YouTube-real) to fix the
# class imbalance that was causing real-video recall of 0.61. Fake is
# still subsampled to 300 to keep ~1:1 balance against the 900 FF++ fakes
# already in the combined dataset.
random.seed(42)  # deterministic fake subsample

BASE_PATH = "Celeb-DF-v2"
OUTPUT_PATH = "celebdf_subset"

REAL_PATHS = [
    os.path.join(BASE_PATH, "Celeb-real"),
    os.path.join(BASE_PATH, "YouTube-real")
]

FAKE_PATH = os.path.join(BASE_PATH, "Celeb-synthesis")

os.makedirs(os.path.join(OUTPUT_PATH, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "fake"), exist_ok=True)

# собираем real видео (все доступные)
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

random.shuffle(fake_videos)
fake_videos = fake_videos[:300]

print(f"Found {len(real_videos)} real videos (taking all)")
print(f"Found fakes, subsampling to {len(fake_videos)}")

print("Copying real videos...")

for v in real_videos:
    shutil.copy(v, os.path.join(OUTPUT_PATH, "real"))

print("Copying fake videos...")

for v in fake_videos:
    shutil.copy(v, os.path.join(OUTPUT_PATH, "fake"))

print(f"Done. Output: {OUTPUT_PATH}/real ({len(real_videos)}) "
      f"+ {OUTPUT_PATH}/fake ({len(fake_videos)})")