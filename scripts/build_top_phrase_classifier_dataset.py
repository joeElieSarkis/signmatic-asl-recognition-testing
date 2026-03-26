import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset")
OUTPUT_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\top_phrases_classifier\data")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
TARGET_FRAMES = 60
FEATURE_DIM = 411   # pose + face + left hand + right hand

TOP_K = 30
MAX_TRAIN_PER_CLASS = 120
MAX_VAL_PER_CLASS = 40
MAX_TEST_PER_CLASS = 40


def clean_sentence(text):
    text = str(text)
    text = re.sub(r"^[A-Z\s]+:\s*", "", text)   # remove speaker names
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)         # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_sequence_length(sequence):
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


def save_lines(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(str(item) + "\n")


def load_labels_df(split):
    labels_path = BASE_PATH / "labels" / f"{split}.csv"
    df = pd.read_csv(labels_path, sep="\t")
    df["SENTENCE_NAME"] = df["SENTENCE_NAME"].astype(str).str.strip()
    df["SENTENCE"] = df["SENTENCE"].astype(str).str.strip()
    df["clean_sentence"] = df["SENTENCE"].apply(clean_sentence)
    return df


# -------------------------
# Step 1: find top phrases from TRAIN
# -------------------------
train_df = load_labels_df("train")
train_counts = Counter(train_df["clean_sentence"])
top_phrases = [phrase for phrase, _ in train_counts.most_common(TOP_K)]
top_phrase_set = set(top_phrases)

print(f"Top {TOP_K} phrases selected from training data:")
for i, phrase in enumerate(top_phrases[:20], start=1):
    print(f"{i}. {phrase} ({train_counts[phrase]})")

save_lines(OUTPUT_PATH / "top_phrases.txt", top_phrases)

# label mapping
label_to_id = {phrase: idx for idx, phrase in enumerate(top_phrases)}
save_lines(OUTPUT_PATH / "label_map.txt", [f"{phrase}\t{idx}" for phrase, idx in label_to_id.items()])


def build_split(split, max_per_class):
    df = load_labels_df(split)
    df = df[df["clean_sentence"].isin(top_phrase_set)].copy()

    label_map = dict(zip(df["SENTENCE_NAME"], df["clean_sentence"]))
    json_root = BASE_PATH / split / "json"

    X = []
    y = []
    y_text = []
    clip_names = []

    used_per_class = defaultdict(int)
    clip_folders = sorted([p for p in json_root.iterdir() if p.is_dir()])

    print(f"\nBuilding {split} split...")
    print(f"Available filtered rows: {len(df)}")

    for clip_folder in clip_folders:
        clip_name = clip_folder.name.strip()

        if clip_name not in label_map:
            continue

        phrase = label_map[clip_name]
        class_id = label_to_id[phrase]

        if used_per_class[class_id] >= max_per_class:
            continue

        sequence = load_clip_sequence_with_face(clip_folder)

        X.append(sequence)
        y.append(class_id)
        y_text.append(phrase)
        clip_names.append(clip_name)
        used_per_class[class_id] += 1

        if len(X) % 100 == 0:
            print(f"Collected {len(X)} samples for {split}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(OUTPUT_PATH / f"X_{split}_topphrases.npy", X)
    np.save(OUTPUT_PATH / f"y_{split}_topphrases.npy", y)
    save_lines(OUTPUT_PATH / f"y_text_{split}_topphrases.txt", y_text)
    save_lines(OUTPUT_PATH / f"clip_names_{split}_topphrases.txt", clip_names)

    print(f"{split} done.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    split_counts = Counter(y_text)
    print("Top classes in split:")
    for phrase, cnt in split_counts.most_common(10):
        print(f"{phrase}: {cnt}")


if __name__ == "__main__":
    build_split("train", MAX_TRAIN_PER_CLASS)
    build_split("val", MAX_VAL_PER_CLASS)
    build_split("test", MAX_TEST_PER_CLASS)