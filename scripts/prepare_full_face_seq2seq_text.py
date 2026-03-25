from pathlib import Path
import json
from collections import Counter

BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\full_face_seq2seq\data")
MIN_FREQ = 3


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def tokenize(sentence):
    return sentence.split()


train_labels = load_labels(BASE_PATH / "y_train_face_full.txt")

counter = Counter()
for sentence in train_labels:
    counter.update(tokenize(sentence))

vocab = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}

for word, freq in counter.items():
    if freq >= MIN_FREQ and word not in vocab:
        vocab[word] = len(vocab)

with open(BASE_PATH / "vocab_face_full.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print("Vocabulary size:", len(vocab))
print("Min frequency:", MIN_FREQ)
print("\nFirst 20 tokens:")
for i, (word, idx) in enumerate(vocab.items()):
    print(word, "->", idx)
    if i >= 19:
        break