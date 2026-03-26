import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\full_face_seq2seq_normalized\data")
MODEL_OUTPUT = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\full_face_seq2seq_normalized\models")
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
MAX_TOKENS = 15
BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-5
D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.3
INPUT_DIM = 411
SEQ_LEN = 60
PATIENCE = 8
LABEL_SMOOTHING = 0.0
GRAD_CLIP = 1.0

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
        self.y_ids = [tokenize_sentence(sent, vocab) for sent in self.y_text]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y_ids[idx], dtype=torch.long)
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


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        d_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        dropout,
        max_src_len,
        max_tgt_len,
        pad_idx
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.src_projection = nn.Linear(input_dim, d_model)
        self.src_norm = nn.LayerNorm(d_model)

        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_norm = nn.LayerNorm(d_model)

        self.src_pos_encoding = PositionalEncoding(d_model, max_len=max_src_len)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len=max_tgt_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, src, tgt_input):
        src = self.src_projection(src)
        src = self.src_norm(src)
        src = self.src_pos_encoding(src)

        tgt = self.tgt_embedding(tgt_input)
        tgt = self.tgt_norm(tgt)
        tgt = self.tgt_pos_encoding(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), src.device)
        tgt_padding_mask = (tgt_input == self.pad_idx)

        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        output = self.output_layer(output)
        return output


with open(BASE_PATH / "vocab_face_full_norm.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab)
pad_idx = vocab["<pad>"]

train_dataset = SignDataset(
    BASE_PATH / "X_train_face_full_norm.npy",
    BASE_PATH / "y_train_face_full_norm.txt",
    vocab
)

val_dataset = SignDataset(
    BASE_PATH / "X_val_face_full_norm.npy",
    BASE_PATH / "y_val_face_full_norm.txt",
    vocab
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Seq2SeqTransformer(
    input_dim=INPUT_DIM,
    vocab_size=vocab_size,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    max_src_len=SEQ_LEN,
    max_tgt_len=MAX_TOKENS,
    pad_idx=pad_idx
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx,
    label_smoothing=LABEL_SMOOTHING
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

best_val_loss = float("inf")
epochs_without_improvement = 0
stop_training = False

print(f"Using device: {DEVICE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Vocab size: {vocab_size}")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        tgt_input = y_batch[:, :-1]
        tgt_output = y_batch[:, 1:]

        optimizer.zero_grad()

        outputs = model(X_batch, tgt_input)
        loss = criterion(
            outputs.reshape(-1, vocab_size),
            tgt_output.reshape(-1)
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf loss detected during training. Stopping.")
            stop_training = True
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

    if stop_training:
        break

    train_loss = train_loss / train_batches if train_batches > 0 else float("inf")

    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            tgt_input = y_batch[:, :-1]
            tgt_output = y_batch[:, 1:]

            outputs = model(X_batch, tgt_input)
            loss = criterion(
                outputs.reshape(-1, vocab_size),
                tgt_output.reshape(-1)
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf loss detected during validation.")
                continue

            val_loss += loss.item()
            val_batches += 1

    val_loss = val_loss / val_batches if val_batches > 0 else float("inf")
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODEL_OUTPUT / "best_full_face_seq2seq_norm_transformer.pt")
        print("Saved new best model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training finished.")
print("Best validation loss:", best_val_loss)