import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

# =========================
# PATHS
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_10words_normalized\data")
MODEL_PATH = BASE_PATH.parent / "models" / "best_face_10w_norm_seq2seq_transformer.pt"

# =========================
# SETTINGS
# =========================
MAX_TOKENS = 12
D_MODEL = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.2
INPUT_DIM = 411
SEQ_LEN = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_labels(path):
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

        output = self.output_layer(output)
        return output


def ids_to_sentence(token_ids, idx_to_word):
    words = []
    for idx in token_ids:
        word = idx_to_word.get(int(idx), "<unk>")

        if word == "<eos>":
            break
        if word in ["<pad>", "<sos>"]:
            continue

        words.append(word)

    return " ".join(words)


with open(BASE_PATH / "vocab_face_10w_norm.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

idx_to_word = {idx: word for word, idx in vocab.items()}
vocab_size = len(vocab)
pad_idx = vocab["<pad>"]
sos_idx = vocab["<sos>"]
eos_idx = vocab["<eos>"]

X_test = np.load(BASE_PATH / "X_test_face_10w_norm.npy")
y_test = load_labels(BASE_PATH / "y_test_face_10w_norm.txt")
clip_names = load_labels(BASE_PATH / "clip_names_test_face_10w_norm.txt")

print("Loaded test samples:", len(X_test))

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

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")


def greedy_decode(model, src, max_tokens, sos_idx, eos_idx, pad_idx):
    generated = [sos_idx]

    for _ in range(max_tokens - 1):
        tgt_input = torch.tensor([generated], dtype=torch.long, device=src.device)
        outputs = model(src, tgt_input)

        next_token = outputs[:, -1, :].argmax(dim=-1).item()
        generated.append(next_token)

        if next_token == eos_idx:
            break

    while len(generated) < max_tokens:
        generated.append(pad_idx)

    return generated


with torch.no_grad():
    for i in [0, 1, 2, 10, 20]:
        x = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        predicted_ids = greedy_decode(
            model=model,
            src=x,
            max_tokens=MAX_TOKENS,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx
        )

        predicted_sentence = ids_to_sentence(predicted_ids, idx_to_word)

        print("\n==============================")
        print("Clip:", clip_names[i])
        print("True sentence:     ", y_test[i])
        print("Predicted sentence:", predicted_sentence)
        print("Predicted token IDs:", predicted_ids)