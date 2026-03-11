import json
from pathlib import Path
import numpy as np

clip_folder = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset\test\json\_fZbAxSSbX4_0-5-rgb_front")

json_files = sorted(clip_folder.glob("*.json"))

print("Number of frames found:", len(json_files))

sequence = []

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = data.get("people", [])

    if len(people) == 0:
        # if no person detected, use zeros
        frame_features = [0.0] * 201
    else:
        person = people[0]

        pose = person.get("pose_keypoints_2d", [])
        left_hand = person.get("hand_left_keypoints_2d", [])
        right_hand = person.get("hand_right_keypoints_2d", [])

        frame_features = pose + left_hand + right_hand

        # just in case, force correct length
        if len(frame_features) != 201:
            frame_features = frame_features[:201] + [0.0] * max(0, 201 - len(frame_features))

    sequence.append(frame_features)

sequence = np.array(sequence, dtype=np.float32)

print("Sequence shape:", sequence.shape)
print("First frame, first 10 values:", sequence[0][:10])