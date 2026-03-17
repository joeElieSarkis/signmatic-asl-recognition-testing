import json  # For reading the keypoint JSON files
import re  # For cleaning sentence text using regular expressions
from pathlib import Path  # For working with file/folder paths cleanly
import pandas as pd  # For reading the label CSV/TSV files
import numpy as np  # For numerical arrays and saving .npy files

# Root folder of the raw dataset
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset")

# Folder where the processed training-ready files will be saved
OUTPUT_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\data")

# Create the output folder if it does not already exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# We want every clip to end up with exactly 60 frames
TARGET_FRAMES = 60

# Each frame contains:
# 25 body keypoints * 3 values (x, y, confidence) = 75
# 21 left hand keypoints * 3 values = 63
# 21 right hand keypoints * 3 values = 63
# Total = 201 features per frame
FEATURE_DIM = 201


def clean_sentence(text):
    """
    Clean the text labels before training.

    What this function does:
    1. Converts the input to string just in case.
    2. Removes speaker names such as 'DAVID CLEMEN:' if they exist.
    3. Converts everything to lowercase.
    4. Removes punctuation.
    5. Removes extra spaces.

    Example:
    'DAVID CLEMEN: Hi!'
    -> 'hi'
    """
    text = str(text)

    # Remove speaker names written in uppercase followed by a colon
    text = re.sub(r"^[A-Z\s]+:\s*", "", text)

    # Convert text to lowercase so 'Hi' and 'hi' become the same token
    text = text.lower()

    # Remove punctuation, keeping only letters/numbers and spaces
    text = re.sub(r"[^\w\s]", "", text)

    # Replace multiple spaces with a single space and trim edges
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_clip_sequence(clip_folder):
    """
    Load one clip folder and convert it into a sequence of frame features.

    Input:
    - clip_folder: a folder containing many JSON files, one per frame

    Output:
    - A NumPy array of shape (60, 201) after length normalization
    """
    # Get all JSON files in this clip and sort them in frame order
    json_files = sorted(clip_folder.glob("*.json"))

    # This list will store the feature vector for each frame
    sequence = []

    for json_file in json_files:
        # Read the current frame JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # OpenPose stores detected people in a "people" list
        people = data.get("people", [])

        # If no person was detected in this frame, use a zero vector
        if len(people) == 0:
            frame_features = [0.0] * FEATURE_DIM
        else:
            # Use the first detected person
            person = people[0]

            # Extract body keypoints (25 keypoints * 3 values = 75)
            pose = person.get("pose_keypoints_2d", [])

            # Extract left hand keypoints (21 * 3 = 63)
            left_hand = person.get("hand_left_keypoints_2d", [])

            # Extract right hand keypoints (21 * 3 = 63)
            right_hand = person.get("hand_right_keypoints_2d", [])

            # Concatenate body + left hand + right hand
            # We intentionally ignore face keypoints
            frame_features = pose + left_hand + right_hand

            # Safety check:
            # If for any reason the frame does not have exactly 201 features,
            # trim or pad it so the dimension is always correct.
            if len(frame_features) != FEATURE_DIM:
                frame_features = (
                    frame_features[:FEATURE_DIM]
                    + [0.0] * max(0, FEATURE_DIM - len(frame_features))
                )

        # Add this frame's feature vector to the sequence
        sequence.append(frame_features)

    # Convert the sequence list into a NumPy array
    # Shape here is (number_of_frames, 201)
    sequence = np.array(sequence, dtype=np.float32)

    # Convert variable-length sequence into fixed-length (60, 201)
    return fix_sequence_length(sequence)


def fix_sequence_length(sequence):
    """
    Force every clip to have exactly TARGET_FRAMES frames.

    Cases:
    - If the clip is shorter than 60 frames -> pad with zero frames
    - If the clip is longer than 60 frames -> uniformly sample 60 frames
    - If it is already 60 frames -> keep it as is
    """
    original_len = len(sequence)

    # Case 1: clip is shorter than the target length
    if original_len < TARGET_FRAMES:
        pad_len = TARGET_FRAMES - original_len

        # Create zero frames to fill the missing part
        padding = np.zeros((pad_len, FEATURE_DIM), dtype=np.float32)

        # Stack original frames + zero padding vertically
        return np.vstack([sequence, padding])

    # Case 2: clip is longer than the target length
    if original_len > TARGET_FRAMES:
        # Uniform sampling:
        # Pick 60 evenly spaced frame indices across the whole clip
        indices = np.linspace(0, original_len - 1, TARGET_FRAMES, dtype=int)
        return sequence[indices]

    # Case 3: clip already has exactly 60 frames
    return sequence


def build_split(split):
    """
    Build one full dataset split: train, val, or test.

    This function:
    1. Reads the label file for the chosen split
    2. Cleans the sentence labels
    3. Loops through all clip folders
    4. Converts each clip into a fixed-size skeleton sequence
    5. Matches each clip with its cleaned sentence label
    6. Saves the final X and y files
    """
    # Path to the label file for this split
    labels_path = BASE_PATH / "labels" / f"{split}.csv"

    # Path to the JSON clip folders for this split
    json_root = BASE_PATH / split / "json"

    # Read the labels file
    # sep="\t" because the file is tab-separated
    df = pd.read_csv(labels_path, sep="\t")

    # Clean up the important text columns
    df["SENTENCE_NAME"] = df["SENTENCE_NAME"].astype(str).str.strip()
    df["SENTENCE"] = df["SENTENCE"].astype(str).str.strip()

    # Create a cleaned version of the sentence labels
    df["clean_sentence"] = df["SENTENCE"].apply(clean_sentence)

    # Build a mapping:
    # clip folder name -> cleaned sentence
    label_map = dict(zip(df["SENTENCE_NAME"], df["clean_sentence"]))

    # Lists to store the final dataset
    X = []  # skeleton sequences
    y = []  # cleaned sentence labels
    clip_names = []  # clip names for debugging/reference

    # Get all clip folders inside this split
    clip_folders = sorted([p for p in json_root.iterdir() if p.is_dir()])

    print(f"\nBuilding {split} split...")
    print(f"Found {len(clip_folders)} clip folders")

    for i, clip_folder in enumerate(clip_folders, start=1):
        clip_name = clip_folder.name.strip()

        # Skip this clip if it has no matching label
        if clip_name not in label_map:
            continue

        # Convert raw JSON frames into a fixed-size sequence
        sequence = load_clip_sequence(clip_folder)

        # Store the sequence and its matching label
        X.append(sequence)
        y.append(label_map[clip_name])
        clip_names.append(clip_name)

        # Print progress every 500 clips
        if i % 500 == 0:
            print(f"Processed {i}/{len(clip_folders)} clips")

    # Convert the full input list into a single NumPy array
    # Final shape will be something like:
    # (number_of_samples, 60, 201)
    X = np.array(X, dtype=np.float32)

    # Save the skeleton sequences
    np.save(OUTPUT_PATH / f"X_{split}.npy", X)

    # Save the cleaned sentence labels
    with open(OUTPUT_PATH / f"y_{split}.txt", "w", encoding="utf-8") as f:
        for sentence in y:
            f.write(sentence + "\n")

    # Save the clip names as an extra reference file
    with open(OUTPUT_PATH / f"clip_names_{split}.txt", "w", encoding="utf-8") as f:
        for clip_name in clip_names:
            f.write(clip_name + "\n")

    # Print a summary at the end
    print(f"{split} done.")
    print("X shape:", X.shape)
    print("Number of labels:", len(y))
    if y:
        print("Example label:", y[0])


if __name__ == "__main__":
    # Build the test split first to verify the preprocessing pipeline
    # Will do the same for val and train in the next commits
    build_split("test")