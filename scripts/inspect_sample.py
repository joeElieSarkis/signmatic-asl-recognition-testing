import json
from pathlib import Path

sample_file = Path("dataset/master_thesis_dataset/test/json/_fZbAxSSbX4_0-5-rgb_front/_fZbAxSSbX4_0-5-rgb_front_000000000000_keypoints.json")

with open(sample_file, "r", encoding="utf-8") as f:
    data = json.load(f)

people = data.get("people", [])

print("Number of people detected:", len(people))

if len(people) == 0:
    print("No person detected in this frame.")
else:
    person = people[0]

    pose = person.get("pose_keypoints_2d", [])
    face = person.get("face_keypoints_2d", [])
    left_hand = person.get("hand_left_keypoints_2d", [])
    right_hand = person.get("hand_right_keypoints_2d", [])

    print("Pose length:", len(pose))
    print("Face length:", len(face))
    print("Left hand length:", len(left_hand))
    print("Right hand length:", len(right_hand))

    frame_features = pose + left_hand + right_hand
    print("Total features without face:", len(frame_features))