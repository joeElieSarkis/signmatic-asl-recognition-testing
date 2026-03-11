import pandas as pd
import re
from pathlib import Path

base_path = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset")

labels_path = base_path / "labels" / "train.csv"

df = pd.read_csv(labels_path, sep="\t")

def clean_sentence(text):
    text = str(text)

    # remove speaker names like "DAVID CLEMEN:"
    text = re.sub(r"^[A-Z\s]+:\s*", "", text)

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # trim whitespace
    text = text.strip()

    return text

df["clean_sentence"] = df["SENTENCE"].apply(clean_sentence)

print("Original examples:")
print(df["SENTENCE"].head(10))

print("\nCleaned examples:")
print(df["clean_sentence"].head(10))