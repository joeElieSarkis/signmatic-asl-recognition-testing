import json
from pathlib import Path
import numpy as np

TARGET_FRAMES = 60
FEATURE_DIM = 201

clip_folder = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset\test\json\-fZc293MpJk_2-1-rgb_front")

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
        left_hand = person.get("hand_left_keypoints_2d", [])
        right_hand = person.get("hand_right_keypoints_2d", [])

        frame_features = pose + left_hand + right_hand

        if len(frame_features) != FEATURE_DIM:
            frame_features = frame_features[:FEATURE_DIM] + [0.0] * max(0, FEATURE_DIM - len(frame_features))

    sequence.append(frame_features)

sequence = np.array(sequence, dtype=np.float32)

original_len = len(sequence)

if original_len < TARGET_FRAMES:
    pad_len = TARGET_FRAMES - original_len
    padding = np.zeros((pad_len, FEATURE_DIM), dtype=np.float32)
    sequence_fixed = np.vstack([sequence, padding])

elif original_len > TARGET_FRAMES:
    # uniform sampling across the whole clip
    indices = np.linspace(0, original_len - 1, TARGET_FRAMES, dtype=int)
    sequence_fixed = sequence[indices]

else:
    sequence_fixed = sequence

print("Original shape:", sequence.shape)
print("Fixed shape:", sequence_fixed.shape)
print("Original frames:", original_len)
print("First frame, first 5 values:", sequence_fixed[0][:5])
print("Last frame, first 5 values:", sequence_fixed[-1][:5])