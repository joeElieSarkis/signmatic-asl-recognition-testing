import json
import numpy as np
import torch
from pathlib import Path
from torch import nn

# =========================
# PATHS
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced\data")
MODEL_PATH = BASE_PATH.parent / "models" / "best_face_4w_balanced_seq2seq_transformer.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 411
MAX_TOKENS = 6


# =========================
# LOAD VOCAB
# =========================
with open(BASE_PATH / "vocab_face_4w_balanced.json") as f:
    vocab = json.load(f)

idx_to_word = {i: w for w, i in vocab.items()}


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =========================
# MODEL (same as training)
# =========================
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
        self.d_model = d_model

        self.src_projection = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

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
        src = self.src_projection(src) * (self.d_model ** 0.5)
        src = self.src_pos_encoding(src)

        tgt = self.tgt_embedding(tgt_input) * (self.d_model ** 0.5)
        tgt = self.tgt_pos_encoding(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), src.device)
        tgt_padding_mask = (tgt_input == self.pad_idx)

        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        return self.output_layer(output)


# =========================
# LOAD MODEL
# =========================
model = Seq2SeqTransformer(
    input_dim=INPUT_DIM,
    vocab_size=len(vocab),
    d_model=256,
    num_heads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    ff_dim=512,
    dropout=0.2,
    max_src_len=60,
    max_tgt_len=MAX_TOKENS,
    pad_idx=vocab["<pad>"]
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# =========================
# GREEDY DECODE
# =========================
def decode(x):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    tokens = [vocab["<sos>"]]

    for _ in range(MAX_TOKENS):
        tgt = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x, tgt)

        next_token = out[0, -1].argmax().item()

        if next_token == vocab["<eos>"]:
            break

        tokens.append(next_token)

    words = [idx_to_word[t] for t in tokens if t not in [vocab["<sos>"], vocab["<pad>"]]]
    return " ".join(words)


# =========================
# TEST
# =========================
X = np.load(BASE_PATH / "X_test_face_4w_balanced.npy")

for i in [0,1,2,10,20]:
    print("\nSample", i)
    print("Prediction:", decode(X[i]))