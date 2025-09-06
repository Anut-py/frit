# This file saves embeddings to disk
# Do not run this anymore, all embeddings have been saved already in /workspace/embeddings/math.pt

import sys
from pathlib import Path
import os

parent = Path().resolve().parent
sys.path.insert(0, str(parent))

import torch
from tqdm import tqdm
from model.embedding import gen_embeddings

batch_size = 256  # Adjust based on memory

# Load corpus
with open("/workspace/corpora/math_corpus.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

file_name = f"/workspace/embeddings/math.pt"

embeddings = []
# Write each batch as a list of tensors
for i in tqdm(range(0, len(lines), batch_size)):
    batch_lines = lines[i : i + batch_size]
    embedding = gen_embeddings(batch_lines)

    for emb in embedding:
        embeddings.append(emb.cpu())

embeddings = torch.stack(embeddings)

with open(file_name, "wb") as embedding_file:
    torch.save(embeddings, embedding_file)
print(f"Saved embeddings to {file_name}")