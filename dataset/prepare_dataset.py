import os
import pandas as pd

# Path to the FaceForensics folder — relative to dataset/ (the script's working dir)
BASE_PATH = "FaceForensics"

data = []

# 1️⃣ Real videos
real_path = os.path.join(BASE_PATH, "original_sequences", "youtube", "c23", "videos")

for file in os.listdir(real_path):
    if file.endswith(".mp4"):
        full_path = os.path.join(real_path, file)
        data.append([full_path, 0])  # 0 = real

# 2️⃣ Fake videos (Deepfakes, FaceSwap, Face2Face)
fake_folders = [
    os.path.join(BASE_PATH, "manipulated_sequences", "Deepfakes", "c23", "videos"),
    os.path.join(BASE_PATH, "manipulated_sequences", "FaceSwap", "c23", "videos"),
    os.path.join(BASE_PATH, "manipulated_sequences", "Face2Face", "c23", "videos"),
]

for folder in fake_folders:
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            full_path = os.path.join(folder, file)
            data.append([full_path, 1])  # 1 = fake

# Создаём DataFrame
df = pd.DataFrame(data, columns=["video_path", "label"])

# Сохраняем
df.to_csv("dataset_labels.csv", index=False)

print("Dataset prepared!")
print(df["label"].value_counts())