from pathlib import Path
import json
from collections import Counter

# =========================
# Paths
# =========================

# Folder where subset labels are stored
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_10words\data")

# We will also save the vocabulary in the same folder
OUTPUT_PATH = BASE_PATH


def load_labels(path):
    """
    Load text labels from a .txt file.
    Each line corresponds to one sentence.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def tokenize(sentence):
    """
    Split a sentence into words (tokens).
    Example:
    "thank you" -> ["thank", "you"]
    """
    return sentence.split()


# =========================
# Load all labels
# =========================

train_labels = load_labels(BASE_PATH / "y_train_face_10w.txt")
val_labels = load_labels(BASE_PATH / "y_val_face_10w.txt")
test_labels = load_labels(BASE_PATH / "y_test_face_10w.txt")

# =========================
# Count word frequencies
# =========================

# Counter will count how many times each word appears
counter = Counter()

# We build vocabulary ONLY from training data
for sentence in train_labels:
    counter.update(tokenize(sentence))

# =========================
# Create vocabulary
# =========================

# Special tokens used by the model
vocab = {
    "<pad>": 0,   # padding token (for fixed-length sequences)
    "<sos>": 1,   # start of sentence
    "<eos>": 2,   # end of sentence
    "<unk>": 3,   # unknown word
}

# Add all words from training set into vocab
for word, _ in counter.items():
    if word not in vocab:
        vocab[word] = len(vocab)

# =========================
# Save vocabulary
# =========================

with open(OUTPUT_PATH / "vocab_face_10w.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

# =========================
# Print summary
# =========================

print("Vocabulary size:", len(vocab))

print("\nFirst 20 tokens:")
for i, (word, idx) in enumerate(vocab.items()):
    print(word, "->", idx)
    if i >= 19:
        break