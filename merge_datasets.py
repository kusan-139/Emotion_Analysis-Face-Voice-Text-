"""
merge_datasets.py

This script merges:
- FER2013  (CSV format)
- RAF-DB Kaggle version (folders: DATASET/train/1..7, DATASET/test/1..7)

into a single folder-based dataset:

dataset_combined/
    train/angry/...
    train/disgust/...
    ...
    val/...
    test/...

IMPORTANT:
- You do NOT need original RAF-DB format.
- Just keep the Kaggle format exactly as:

  data/rafdb/DATASET/
      train/1..7/
      test/1..7/
      train_labels.csv
      test_labels.csv
"""

import os
import csv
import cv2
import shutil
from tqdm import tqdm
import numpy as np

from config import DATA_DIR, COMBINED_DATASET_DIR, EMOTION2IDX, IDX2EMOTION, INPUT_SIZE

# ---- FER2013 PATH ----
FER_CSV_PATH = os.path.join(DATA_DIR, "fer2013", "fer2013.csv")

# ---- RAF (KAGGLE) PATH ----
# Expected structure:
# data/rafdb/DATASET/train/1..7/*.jpg
# data/rafdb/DATASET/test/1..7/*.jpg
RAF_KAGGLE_ROOT = os.path.join(DATA_DIR, "rafdb", "DATASET")


def ensure_folders():
    """Create base combined dataset folders (flat, per emotion)."""
    os.makedirs(COMBINED_DATASET_DIR, exist_ok=True)
    for label in EMOTION2IDX.keys():
        os.makedirs(os.path.join(COMBINED_DATASET_DIR, label), exist_ok=True)


# =========================
# FER2013 HANDLING
# =========================

def process_fer2013():
    """
    FER2013: CSV with columns [emotion, pixels, Usage]
    Emotion mapping already matches our 0-6 scheme.
    """
    print("=== Processing FER2013 ===")
    if not os.path.exists(FER_CSV_PATH):
        print(f"[WARN] FER2013 csv not found at {FER_CSV_PATH} - skipping FER part.")
        return

    with open(FER_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(tqdm(reader, desc="FER2013 rows")):
            emotion_idx = int(row["emotion"])
            if emotion_idx not in IDX2EMOTION:
                continue

            emotion_name = IDX2EMOTION[emotion_idx]
            pixels = row["pixels"]
            pixel_vals = np.array(list(map(int, pixels.split())), dtype=np.uint8)
            img = pixel_vals.reshape(48, 48)

            # grayscale -> RGB, resize to 224x224
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))

            filename = f"fer_{i:06d}.jpg"
            out_path = os.path.join(COMBINED_DATASET_DIR, emotion_name, filename)
            cv2.imwrite(out_path, img_resized)


# =========================
# RAF KAGGLE HANDLING
# =========================

def raf_folder_to_emotion(folder_name: str) -> str:
    """
    Kaggle RAF folders are named 1..7.
    Mapping (same as RAF-DB basic):

        1 - surprise
        2 - fear
        3 - disgust
        4 - happy
        5 - sad
        6 - angry
        7 - neutral
    """
    code = int(folder_name)
    mapping = {
        1: "surprise",
        2: "fear",
        3: "disgust",
        4: "happy",
        5: "sad",
        6: "angry",
        7: "neutral",
    }
    return mapping.get(code, None)


def process_rafdb_kaggle():
    """
    Use Kaggle RAF dataset in format:

        data/rafdb/DATASET/train/1..7/*.jpg
        data/rafdb/DATASET/test/1..7/*.jpg

    We simply loop through train and test folders, read each image,
    map folder name to emotion, resize to 224x224, and save into our
    COMBINED_DATASET_DIR/<emotion>/.
    """
    print("=== Processing RAF-DB (Kaggle format) ===")
    if not os.path.exists(RAF_KAGGLE_ROOT):
        print(f"[WARN] RAF Kaggle root not found at {RAF_KAGGLE_ROOT} - skipping RAF part.")
        return

    for split in ["train", "test"]:
        split_dir = os.path.join(RAF_KAGGLE_ROOT, split)
        if not os.path.exists(split_dir):
            print(f"[WARN] Split folder not found: {split_dir}")
            continue

        class_folders = sorted(d for d in os.listdir(split_dir)
                               if os.path.isdir(os.path.join(split_dir, d)))

        for cls in class_folders:
            emo_name = raf_folder_to_emotion(cls)
            if emo_name is None:
                print(f"[WARN] Unknown RAF class folder: {cls}, skipping.")
                continue

            cls_dir = os.path.join(split_dir, cls)
            files = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for fname in tqdm(files, desc=f"RAF {split}/{cls} -> {emo_name}", leave=False):
                img_path = os.path.join(cls_dir, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

                # build output filename with split + cls prefix to avoid collisions
                out_name = f"raf_{split}_{cls}_{fname}"
                out_path = os.path.join(COMBINED_DATASET_DIR, emo_name, out_name)
                cv2.imwrite(out_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))


# =========================
# SPLIT INTO TRAIN / VAL / TEST
# =========================

def split_train_val_test():
    """
    Split dataset_combined per-class into train/val/test
    80/10/10 split using folder copies.

    Final structure:
        dataset_combined/
            train/
                angry/...
            val/
                angry/...
            test/
                angry/...
    """
    print("=== Splitting into train/val/test ===")
    base = COMBINED_DATASET_DIR
    tmp_dir = os.path.join(base, "_all")
    os.makedirs(tmp_dir, exist_ok=True)

    # Move existing emotion folders into _all
    for emo in list(EMOTION2IDX.keys()):
        src = os.path.join(base, emo)
        dst = os.path.join(tmp_dir, emo)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)

    # Create train/val/test directories
    for split in ["train", "val", "test"]:
        for emo in EMOTION2IDX.keys():
            os.makedirs(os.path.join(base, split, emo), exist_ok=True)

    # Perform split for each emotion
    for emo in EMOTION2IDX.keys():
        emo_dir = os.path.join(tmp_dir, emo)
        if not os.path.exists(emo_dir):
            continue

        files = sorted(os.listdir(emo_dir))
        n = len(files)
        if n == 0:
            continue

        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for fname in train_files:
            shutil.move(os.path.join(emo_dir, fname),
                        os.path.join(base, "train", emo, fname))
        for fname in val_files:
            shutil.move(os.path.join(emo_dir, fname),
                        os.path.join(base, "val", emo, fname))
        for fname in test_files:
            shutil.move(os.path.join(emo_dir, fname),
                        os.path.join(base, "test", emo, fname))

    print("Cleaning temporary _all directory...")
    shutil.rmtree(tmp_dir, ignore_errors=True)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    ensure_folders()
    process_fer2013()       # uses fer2013.csv
    process_rafdb_kaggle()  # uses DATASET/train & DATASET/test
    split_train_val_test()
    print("✅ Merged dataset created under:", COMBINED_DATASET_DIR)
