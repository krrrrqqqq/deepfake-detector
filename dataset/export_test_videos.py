"""Копирует все видео из test_split.csv в папку на рабочем столе.

Сопоставляет каждый video_id с исходным .mp4 и копирует файлы,
раскладывая по подпапкам real/ и fake/, чтобы отложенный тестовый набор
можно было перенести на другую машину (например, ноутбук для защиты диплома).

Запуск из каталога dataset/:
    python export_test_videos.py
"""

import os
import csv
import shutil

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SPLIT = os.path.join(DATASET_DIR, "test_split.csv")

# Расположение исходных видео FF++
FF_REAL_DIR = os.path.join(
    DATASET_DIR, "FaceForensics", "original_sequences", "youtube", "c23", "videos")
FF_FAKE_DIR = os.path.join(
    DATASET_DIR, "FaceForensics", "manipulated_sequences")  # + {method}/c23/videos
# Расположение исходных видео Celeb-DF
CDF_REAL_DIR = os.path.join(DATASET_DIR, "celebdf_subset", "real")
CDF_FAKE_DIR = os.path.join(DATASET_DIR, "celebdf_subset", "fake")


def find_desktop() -> str:
    """Возвращает путь к рабочему столу, учитывая перенаправление в OneDrive."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "OneDrive", "Рабочий стол"),
        os.path.join(home, "Рабочий стол"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]  # запасной вариант ~/Desktop, даже если его ещё нет


def resolve_source(video_id: str, label: str) -> str | None:
    """Сопоставляет video_id из test_split с путём к исходному .mp4 (или None)."""
    if video_id.startswith("cdf__"):
        name = video_id[len("cdf__"):]
        base = CDF_REAL_DIR if label == "0" else CDF_FAKE_DIR
        return os.path.join(base, name + ".mp4")

    if video_id.startswith("ff__"):
        rest = video_id[len("ff__"):]            # напр. "youtube__004.mp4" / "Deepfakes__035_036.mp4"
        method, filename = rest.split("__", 1)   # filename уже оканчивается на .mp4
        if method == "youtube":                  # настоящее
            return os.path.join(FF_REAL_DIR, filename)
        return os.path.join(FF_FAKE_DIR, method, "c23", "videos", filename)  # фейк

    return None


def main():
    dest_root = os.path.join(find_desktop(), "test_videos_defense")
    dest_real = os.path.join(dest_root, "real")
    dest_fake = os.path.join(dest_root, "fake")
    os.makedirs(dest_real, exist_ok=True)
    os.makedirs(dest_fake, exist_ok=True)

    copied, missing = 0, []
    with open(TEST_SPLIT, newline="") as f:
        for row in csv.DictReader(f):
            vid, label = row["video_id"], row["label"]
            src = resolve_source(vid, label)
            if not src or not os.path.isfile(src):
                missing.append((vid, src))
                continue
            # Сохраняем полный video_id в имени файла, чтобы он отслеживался в test_split.csv
            out_name = vid if vid.endswith(".mp4") else vid + ".mp4"
            dest_dir = dest_real if label == "0" else dest_fake
            shutil.copy2(src, os.path.join(dest_dir, out_name))
            copied += 1

    print(f"Copied {copied} videos -> {dest_root}")
    print(f"  real/ and fake/ subfolders created")
    if missing:
        print(f"\n{len(missing)} source file(s) NOT found:")
        for vid, src in missing[:20]:
            print(f"  {vid}  =>  {src}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
