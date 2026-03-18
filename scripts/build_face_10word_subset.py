import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset")
OUTPUT_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_10words\data")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# Experiment settings
# =========================
TARGET_FRAMES = 60
FEATURE_DIM = 411  # pose + face + left hand + right hand
MAX_WORDS = 10

MAX_TRAIN_SAMPLES = 5000
MAX_VAL_SAMPLES = 1000
MAX_TEST_SAMPLES = 1000


def clean_sentence(text):
    """
    Clean sentence labels:
    - remove speaker names
    - lowercase
    - remove punctuation
    - normalize spaces
    """
    text = str(text)
    text = re.sub(r"^[A-Z\s]+:\s*", "", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_sequence_length(sequence):
    """
    Convert variable-length sequence to exactly TARGET_FRAMES.
    - pad with zeros if too short
    - uniformly sample if too long
    """
    original_len = len(sequence)

    if original_len < TARGET_FRAMES:
        pad_len = TARGET_FRAMES - original_len
        padding = np.zeros((pad_len, FEATURE_DIM), dtype=np.float32)
        return np.vstack([sequence, padding])

    if original_len > TARGET_FRAMES:
        indices = np.linspace(0, original_len - 1, TARGET_FRAMES, dtype=int)
        return sequence[indices]

    return sequence


def load_clip_sequence_with_face(clip_folder):
    """
    Read all JSON frames from one clip and build a (60, 411) sequence.
    """
    json_files = sorted(clip_folder.glob("*.json"))
    sequence = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        people = data.get("people", [])

        if len(people) == 0:
            frame_features = [0.0] * FEATURE_DIM
        else:
            person = people[0]

            pose = person.get("pose_keypoints_2d", [])
            face = person.get("face_keypoints_2d", [])
            left_hand = person.get("hand_left_keypoints_2d", [])
            right_hand = person.get("hand_right_keypoints_2d", [])

            frame_features = pose + face + left_hand + right_hand

            if len(frame_features) != FEATURE_DIM:
                frame_features = (
                    frame_features[:FEATURE_DIM]
                    + [0.0] * max(0, FEATURE_DIM - len(frame_features))
                )

        sequence.append(frame_features)

    sequence = np.array(sequence, dtype=np.float32)
    return fix_sequence_length(sequence)


def save_labels(path, labels):
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")


def build_subset(split, max_samples):
    """
    Build a filtered subset for one split using:
    - face-inclusive features
    - cleaned sentences
    - only labels with <= MAX_WORDS
    """
    labels_path = BASE_PATH / "labels" / f"{split}.csv"
    json_root = BASE_PATH / split / "json"

    df = pd.read_csv(labels_path, sep="\t")
    df["SENTENCE_NAME"] = df["SENTENCE_NAME"].astype(str).str.strip()
    df["SENTENCE"] = df["SENTENCE"].astype(str).str.strip()
    df["clean_sentence"] = df["SENTENCE"].apply(clean_sentence)

    # Filter to short sentences only
    df = df[df["clean_sentence"].str.split().str.len() <= MAX_WORDS].copy()

    label_map = dict(zip(df["SENTENCE_NAME"], df["clean_sentence"]))

    X = []
    y = []
    clip_names = []

    clip_folders = sorted([p for p in json_root.iterdir() if p.is_dir()])

    print(f"\nBuilding {split} subset...")
    print(f"Filtered labels available: {len(label_map)}")

    for i, clip_folder in enumerate(clip_folders, start=1):
        clip_name = clip_folder.name.strip()

        if clip_name not in label_map:
            continue

        sequence = load_clip_sequence_with_face(clip_folder)

        X.append(sequence)
        y.append(label_map[clip_name])
        clip_names.append(clip_name)

        if len(X) % 100 == 0:
            print(f"Collected {len(X)} samples for {split}")

        if len(X) >= max_samples:
            break

    X = np.array(X, dtype=np.float32)

    np.save(OUTPUT_PATH / f"X_{split}_face_10w.npy", X)
    save_labels(OUTPUT_PATH / f"y_{split}_face_10w.txt", y)
    save_labels(OUTPUT_PATH / f"clip_names_{split}_face_10w.txt", clip_names)

    print(f"{split} subset done.")
    print("X shape:", X.shape)
    print("Labels:", len(y))
    if y:
        print("Example label:", y[0])


if __name__ == "__main__":
    # Build a manageable first experiment:
    # face keypoints included + sentences up to 10 words.
    build_subset("train", MAX_TRAIN_SAMPLES)
    build_subset("val", MAX_VAL_SAMPLES)
    build_subset("test", MAX_TEST_SAMPLES)