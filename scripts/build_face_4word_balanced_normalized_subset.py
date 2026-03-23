import shutil
from pathlib import Path
import numpy as np

# =========================
# Paths
# =========================
SOURCE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced\data")
OUTPUT_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced_normalized\data")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# Feature layout
# =========================
POSE_KP = 25
FACE_KP = 70
LH_KP = 21
RH_KP = 21

TOTAL_KP = POSE_KP + FACE_KP + LH_KP + RH_KP
FEATURE_DIM = TOTAL_KP * 3  # 411

# BODY_25 pose indices
NECK_IDX = 1
R_SHOULDER_IDX = 2
L_SHOULDER_IDX = 5


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def save_labels(path, labels):
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")


def get_xyc(frame, kp_idx):
    base = kp_idx * 3
    return frame[base], frame[base + 1], frame[base + 2]


def normalize_frame(frame):
    """
    Normalize one frame:
    - center x,y around neck
    - scale by shoulder distance
    - keep confidence unchanged
    """
    frame = frame.copy()

    neck_x, neck_y, neck_c = get_xyc(frame, NECK_IDX)
    rs_x, rs_y, rs_c = get_xyc(frame, R_SHOULDER_IDX)
    ls_x, ls_y, ls_c = get_xyc(frame, L_SHOULDER_IDX)

    valid_neck = neck_c > 0 and not (neck_x == 0 and neck_y == 0)
    valid_shoulders = rs_c > 0 and ls_c > 0 and not (
        (rs_x == 0 and rs_y == 0) or (ls_x == 0 and ls_y == 0)
    )

    if valid_neck:
        center_x, center_y = neck_x, neck_y
    else:
        center_x, center_y = 0.0, 0.0

    if valid_shoulders:
        shoulder_dist = np.sqrt((rs_x - ls_x) ** 2 + (rs_y - ls_y) ** 2)
        scale = shoulder_dist if shoulder_dist > 1e-6 else 1.0
    else:
        scale = 1.0

    for kp in range(TOTAL_KP):
        base = kp * 3
        x = frame[base]
        y = frame[base + 1]
        c = frame[base + 2]

        if not (x == 0 and y == 0):
            frame[base] = (x - center_x) / scale
            frame[base + 1] = (y - center_y) / scale

        frame[base + 2] = c

    return frame


def normalize_sequence(sequence):
    normalized = np.zeros_like(sequence, dtype=np.float32)
    for i in range(sequence.shape[0]):
        normalized[i] = normalize_frame(sequence[i])
    return normalized.astype(np.float32)


def process_split(split):
    x_path = SOURCE_PATH / f"X_{split}_face_4w_balanced.npy"
    y_path = SOURCE_PATH / f"y_{split}_face_4w_balanced.txt"
    clip_path = SOURCE_PATH / f"clip_names_{split}_face_4w_balanced.txt"

    X = np.load(x_path)
    y = load_labels(y_path)
    clip_names = load_labels(clip_path)

    print(f"\nProcessing {split} split...")
    print("Original shape:", X.shape)

    X_norm = np.zeros_like(X, dtype=np.float32)

    for i in range(len(X)):
        X_norm[i] = normalize_sequence(X[i])

        if (i + 1) % 100 == 0:
            print(f"Normalized {i+1}/{len(X)} samples")

    np.save(OUTPUT_PATH / f"X_{split}_face_4w_balanced_norm.npy", X_norm)
    save_labels(OUTPUT_PATH / f"y_{split}_face_4w_balanced_norm.txt", y)
    save_labels(OUTPUT_PATH / f"clip_names_{split}_face_4w_balanced_norm.txt", clip_names)

    print(f"{split} normalized done.")
    print("Saved shape:", X_norm.shape)
    print("Example first frame, first 12 values:")
    print(X_norm[0, 0, :12])


if __name__ == "__main__":
    process_split("train")
    process_split("val")
    process_split("test")

    # Copy vocab because labels are unchanged
    src_vocab = SOURCE_PATH / "vocab_face_4w_balanced.json"
    dst_vocab = OUTPUT_PATH / "vocab_face_4w_balanced_norm.json"

    if src_vocab.exists():
        shutil.copy(src_vocab, dst_vocab)
        print("\nCopied vocab file to normalized experiment folder.")