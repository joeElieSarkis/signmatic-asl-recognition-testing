import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_10words\data")
MODEL_OUTPUT = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_10words\models")
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
MAX_TOKENS = 12          # enough for <=10 words plus <sos> and <eos>
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.1
INPUT_DIM = 411          # pose + face + hands
SEQ_LEN = 60             # fixed frame length

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def tokenize_sentence(sentence, vocab):
    tokens = ["<sos>"] + sentence.split() + ["<eos>"]
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

    if len(ids) < MAX_TOKENS:
        ids += [vocab["<pad>"]] * (MAX_TOKENS - len(ids))
    else:
        ids = ids[:MAX_TOKENS]

    return ids


class SignDataset(Dataset):
    def __init__(self, x_path, y_path, vocab):
        self.X = np.load(x_path)
        self.y_text = load_labels(y_path)
        self.vocab = vocab
        self.y_ids = [tokenize_sentence(sent, vocab) for sent in self.y_text]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)   # (60, 411)
        y = torch.tensor(self.y_ids[idx], dtype=torch.long)  # (MAX_TOKENS,)
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerSignModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, ff_dim, dropout, vocab_size, max_tokens):
        super().__init__()

        # Project 411-dim frame features into Transformer embedding size
        self.input_proj = nn.Linear(input_dim, d_model)

        # Add positional information to frame sequence
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQ_LEN)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pool encoded sequence into one representation
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Predict fixed token sequence
        self.fc = nn.Linear(d_model, vocab_size * max_tokens)

        self.vocab_size = vocab_size
        self.max_tokens = max_tokens

    def forward(self, x):
        # x: (batch, 60, 411)
        x = self.input_proj(x)          # (batch, 60, d_model)
        x = self.pos_encoder(x)         # add positional encoding
        x = self.encoder(x)             # (batch, 60, d_model)

        # Pool over time dimension
        x = x.transpose(1, 2)           # (batch, d_model, 60)
        x = self.pool(x).squeeze(-1)    # (batch, d_model)

        # Predict all output token positions at once
        x = self.fc(x)                  # (batch, vocab_size * max_tokens)
        x = x.view(-1, self.max_tokens, self.vocab_size)
        return x


# =========================
# Load vocabulary
# =========================
with open(BASE_PATH / "vocab_face_10w.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

# =========================
# Load datasets
# =========================
train_dataset = SignDataset(
    BASE_PATH / "X_train_face_10w.npy",
    BASE_PATH / "y_train_face_10w.txt",
    vocab
)

val_dataset = SignDataset(
    BASE_PATH / "X_val_face_10w.npy",
    BASE_PATH / "y_val_face_10w.txt",
    vocab
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# Build model
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

criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")

# =========================
# Training loop
# =========================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)  # (batch, max_tokens, vocab_size)

        loss = criterion(
            outputs.reshape(-1, vocab_size),
            y_batch.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                y_batch.reshape(-1)
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_OUTPUT / "best_face_10w_transformer.pt")
        print("Saved new best model.")

print("Training finished.")
print("Best validation loss:", best_val_loss)