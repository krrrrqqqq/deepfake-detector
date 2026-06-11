import os
import pandas as pd

# Путь к папке FaceForensics — относительно dataset/ (рабочая директория скрипта)
BASE_PATH = "FaceForensics"

data = []

# 1️⃣ Настоящие видео
real_path = os.path.join(BASE_PATH, "original_sequences", "youtube", "c23", "videos")

for file in os.listdir(real_path):
    if file.endswith(".mp4"):
        full_path = os.path.join(real_path, file)
        data.append([full_path, 0])  # 0 = настоящее

# 2️⃣ Фейковые видео (Deepfakes, FaceSwap, Face2Face)
fake_folders = [
    os.path.join(BASE_PATH, "manipulated_sequences", "Deepfakes", "c23", "videos"),
    os.path.join(BASE_PATH, "manipulated_sequences", "FaceSwap", "c23", "videos"),
    os.path.join(BASE_PATH, "manipulated_sequences", "Face2Face", "c23", "videos"),
]

for folder in fake_folders:
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            full_path = os.path.join(folder, file)
            data.append([full_path, 1])  # 1 = фейк

# Создаём DataFrame
df = pd.DataFrame(data, columns=["video_path", "label"])

# Сохраняем
df.to_csv("dataset_labels.csv", index=False)

print("Dataset prepared!")
print(df["label"].value_counts())
