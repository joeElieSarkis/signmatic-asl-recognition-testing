from pathlib import Path
import numpy as np
import torch
from torch import nn

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\top10_phrases_classifier\data")
MODEL_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\top10_phrases_classifier\models\best_top10_phrase_transformer_classifier.pt")
OUTPUT_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\top10_phrases_classifier")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.3
INPUT_DIM = 411
SEQ_LEN = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


class PositionalEncoding(nn.Module):
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


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, ff_dim, dropout, num_classes):
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
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        x = self.fc(x)
        return x


label_map_lines = load_lines(BASE_PATH / "label_map.txt")
id_to_label = {}
for line in label_map_lines:
    phrase, idx = line.rsplit("\t", 1)
    id_to_label[int(idx)] = phrase

num_classes = len(id_to_label)

X_test = np.load(BASE_PATH / "X_test_top10.npy")
y_test = np.load(BASE_PATH / "y_test_top10.npy")
y_text_test = load_lines(BASE_PATH / "y_text_test_top10.txt")
clip_names = load_lines(BASE_PATH / "clip_names_test_top10.txt")

print("Loaded test samples:", len(X_test))

model = TransformerClassifier(
    input_dim=INPUT_DIM,
    d_model=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    num_classes=num_classes
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")

correct = 0
results = []

with torch.no_grad():
    for i in range(len(X_test)):
        x = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        outputs = model(x)
        pred_id = outputs.argmax(dim=1).item()

        pred_text = id_to_label[pred_id]
        true_text = y_text_test[i]

        is_correct = pred_text == true_text
        if is_correct:
            correct += 1

        results.append({
            "clip": clip_names[i],
            "true": true_text,
            "pred": pred_text,
            "correct": is_correct
        })

accuracy = correct / len(X_test) if len(X_test) > 0 else 0.0

print("\n==============================")
print("Classification Accuracy:", f"{accuracy:.4f}")
print("Correct Predictions:", correct)
print("Total Samples:", len(X_test))

print("\nSample predictions:")
for item in results[:15]:
    print("\n------------------------------")
    print("Clip:   ", item["clip"])
    print("True:   ", item["true"])
    print("Pred:   ", item["pred"])
    print("Match:  ", item["correct"])

results_file = OUTPUT_PATH / "evaluation_results_top10_phrase_classifier.txt"
with open(results_file, "w", encoding="utf-8") as f:
    f.write(f"Classification Accuracy: {accuracy:.4f}\n")
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