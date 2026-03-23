import os
import shutil
import random

# -----------------------------
# CONFIGURATION
# -----------------------------
RAW_DIR = "dataset/raw"        # source images
OUTPUT_DIR = "dataset"         # final dataset

EMOTIONS = ["angry", "happy", "sad"]

SPLIT_RATIO = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15
}

random.seed(42)

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
for split in SPLIT_RATIO:
    for emotion in EMOTIONS:
        os.makedirs(os.path.join(OUTPUT_DIR, split, emotion), exist_ok=True)

# -----------------------------
# SPLIT IMAGES
# -----------------------------
for emotion in EMOTIONS:
    emotion_path = os.path.join(RAW_DIR, emotion)
    images = os.listdir(emotion_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(SPLIT_RATIO["train"] * total)
    val_end = train_end + int(SPLIT_RATIO["validation"] * total)

    split_map = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_map.items():
        for file in files:
            src = os.path.join(emotion_path, file)
            dst = os.path.join(OUTPUT_DIR, split, emotion, file)
            shutil.copy(src, dst)

    print(f"✅ {emotion}: {total} images split")

print("\n🎉 Dataset split completed successfully!")
