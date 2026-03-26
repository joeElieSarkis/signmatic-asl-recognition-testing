from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\top10_phrases_classifier\data")
MODEL_OUTPUT = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\top10_phrases_classifier\models")
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.3
INPUT_DIM = 411
SEQ_LEN = 60
PATIENCE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


class PhraseDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


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


label_map = load_lines(BASE_PATH / "label_map.txt")
num_classes = len(label_map)

train_dataset = PhraseDataset(
    BASE_PATH / "X_train_top10.npy",
    BASE_PATH / "y_train_top10.npy"
)

val_dataset = PhraseDataset(
    BASE_PATH / "X_val_top10.npy",
    BASE_PATH / "y_val_top10.npy"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TransformerClassifier(
    input_dim=INPUT_DIM,
    d_model=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    num_classes=num_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

best_val_loss = float("inf")
epochs_without_improvement = 0

print(f"Using device: {DEVICE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Number of classes: {num_classes}")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

        preds = outputs.argmax(dim=1)
        train_correct += (preds == y_batch).sum().item()
        train_total += y_batch.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total if train_total > 0 else 0.0

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    print(
        f"Epoch {epoch+1}/{EPOCHS} - "
        f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
        f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODEL_OUTPUT / "best_top10_phrase_transformer_classifier.pt")
        print("Saved new best model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training finished.")
print("Best validation loss:", best_val_loss)