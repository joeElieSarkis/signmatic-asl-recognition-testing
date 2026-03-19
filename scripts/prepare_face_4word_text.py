from pathlib import Path
import json
from collections import Counter

# Folder containing the 4-word subset labels
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words\data")


def load_labels(path):
    """Load one sentence per line from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def tokenize(sentence):
    """Simple word-level tokenization."""
    return sentence.split()


# Load labels
train_labels = load_labels(BASE_PATH / "y_train_face_4w.txt")
val_labels = load_labels(BASE_PATH / "y_val_face_4w.txt")
test_labels = load_labels(BASE_PATH / "y_test_face_4w.txt")

# Build vocabulary from TRAIN only
counter = Counter()
for sentence in train_labels:
    counter.update(tokenize(sentence))

# Special tokens
vocab = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}

# Add training words
for word, _ in counter.items():
    if word not in vocab:
        vocab[word] = len(vocab)

# Save vocab
with open(BASE_PATH / "vocab_face_4w.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print("Vocabulary size:", len(vocab))
print("\nFirst 20 tokens:")
for i, (word, idx) in enumerate(vocab.items()):
    print(word, "->", idx)
    if i >= 19:
        break