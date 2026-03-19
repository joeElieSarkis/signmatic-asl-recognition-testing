import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced\data")
MODEL_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced\models\best_face_4w_balanced_transformer.pt")
OUTPUT_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
MAX_TOKENS = 6
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.3
INPUT_DIM = 411
SEQ_LEN = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_labels(path):
    """Load one sentence per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerSignModel(nn.Module):
    """Same model architecture used during training."""
    def __init__(self, input_dim, d_model, num_heads, num_layers, ff_dim, dropout, vocab_size, max_tokens):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQ_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, vocab_size * max_tokens)

        self.vocab_size = vocab_size
        self.max_tokens = max_tokens

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        x = self.fc(x)
        x = x.view(-1, self.max_tokens, self.vocab_size)
        return x


def ids_to_sentence(token_ids, idx_to_word):
    """
    Convert token IDs to readable sentence.
    Stops at <eos>, ignores <pad> and <sos>.
    """
    words = []

    for idx in token_ids:
        word = idx_to_word.get(int(idx), "<unk>")

        if word == "<eos>":
            break

        if word in ["<pad>", "<sos>"]:
            continue

        words.append(word)

    return " ".join(words)


# =========================
# Load vocab
# =========================
with open(BASE_PATH / "vocab_face_4w_balanced.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

idx_to_word = {idx: word for word, idx in vocab.items()}
vocab_size = len(vocab)

# =========================
# Load balanced test set
# =========================
X_test = np.load(BASE_PATH / "X_test_face_4w_balanced.npy")
y_test = load_labels(BASE_PATH / "y_test_face_4w_balanced.txt")
clip_names = load_labels(BASE_PATH / "clip_names_test_face_4w_balanced.txt")

print("Loaded test samples:", len(X_test))

# =========================
# Load model
# =========================
model = TransformerSignModel(
    input_dim=INPUT_DIM,
    d_model=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    vocab_size=vocab_size,
    max_tokens=MAX_TOKENS
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")

# =========================
# Evaluate
# =========================
correct = 0
results = []

with torch.no_grad():
    for i in range(len(X_test)):
        x = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        outputs = model(x)
        predicted_ids = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()
        predicted_sentence = ids_to_sentence(predicted_ids, idx_to_word)

        true_sentence = y_test[i]
        clip_name = clip_names[i]

        is_correct = predicted_sentence == true_sentence
        if is_correct:
            correct += 1

        results.append({
            "clip": clip_name,
            "true": true_sentence,
            "pred": predicted_sentence,
            "correct": is_correct
        })

accuracy = correct / len(X_test) if len(X_test) > 0 else 0.0

print("\n==============================")
print("Exact Match Accuracy:", f"{accuracy:.4f}")
print("Correct Predictions:", correct)
print("Total Samples:", len(X_test))

# =========================
# Print a few examples
# =========================
print("\nSample predictions:")
for item in results[:10]:
    print("\n------------------------------")
    print("Clip:   ", item["clip"])
    print("True:   ", item["true"])
    print("Pred:   ", item["pred"])
    print("Match:  ", item["correct"])

# =========================
# Save all results
# =========================
results_file = OUTPUT_PATH / "evaluation_results_face_4w_balanced.txt"

with open(results_file, "w", encoding="utf-8") as f:
    f.write(f"Exact Match Accuracy: {accuracy:.4f}\n")
    f.write(f"Correct Predictions: {correct}\n")
    f.write(f"Total Samples: {len(X_test)}\n\n")

    for item in results:
        f.write(f"Clip: {item['clip']}\n")
        f.write(f"True: {item['true']}\n")
        f.write(f"Pred: {item['pred']}\n")
        f.write(f"Match: {item['correct']}\n")
        f.write("-" * 40 + "\n")

print("\nSaved evaluation file to:")
print(results_file)