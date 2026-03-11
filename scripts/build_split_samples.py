import json
from pathlib import Path
import pandas as pd
import numpy as np

split = "test"

base_path = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset")
labels_path = base_path / "labels" / f"{split}.csv"
json_root = base_path / split / "json"

df = pd.read_csv(labels_path, sep="\t")
df["SENTENCE_NAME"] = df["SENTENCE_NAME"].astype(str).str.strip()
df["SENTENCE"] = df["SENTENCE"].astype(str).str.strip()

label_map = dict(zip(df["SENTENCE_NAME"], df["SENTENCE"]))

samples = []

clip_folders = sorted([p for p in json_root.iterdir() if p.is_dir()])
print(f"Found {len(clip_folders)} clip folders")

for i, clip_folder in enumerate(clip_folders[:10]):  # only first 10 for now
    clip_name = clip_folder.name.strip()

    if clip_name not in label_map:
        print(f"Skipping {clip_name} (no label found)")
        continue

    json_files = sorted(clip_folder.glob("*.json"))
    sequence = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        people = data.get("people", [])

        if len(people) == 0:
            frame_features = [0.0] * 201
        else:
            person = people[0]
            pose = person.get("pose_keypoints_2d", [])
            left_hand = person.get("hand_left_keypoints_2d", [])
            right_hand = person.get("hand_right_keypoints_2d", [])
            frame_features = pose + left_hand + right_hand

            if len(frame_features) != 201:
                frame_features = frame_features[:201] + [0.0] * max(0, 201 - len(frame_features))

        sequence.append(frame_features)

    sequence = np.array(sequence, dtype=np.float32)

    samples.append({
        "clip_name": clip_name,
        "sequence": sequence,
        "sentence": label_map[clip_name]
    })

    print(f"[{i+1}] {clip_name} -> {sequence.shape} -> {label_map[clip_name]}")

print(f"\nBuilt {len(samples)} samples.")
if samples:
    print("Example sample:")
    print("Clip:", samples[0]["clip_name"])
    print("Shape:", samples[0]["sequence"].shape)
    print("Sentence:", samples[0]["sentence"])