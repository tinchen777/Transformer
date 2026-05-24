# -*- coding: utf-8 -*-
"""
A simple demo for training the custom DecoderOnlyTransformer
on the HuggingFace `locuslab/TOFU` dataset (question-answer pairs about
fictitious authors).

Usage
-----
    python train_tofu.py
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer

from transformer import DecoderOnlyTransformer, ModuleConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Data
DATASET_NAME = "locuslab/TOFU"
DATASET_CONFIG = "full"        # 4000 QA pairs about 200 fictitious authors
MAX_LEN = 192                  # truncation length for QA pair
NUM_TRAIN_SAMPLES = 512        # subset size for a fast demo (set None for all)

# Training
BATCH_SIZE = 8
EPOCHS = 3
LR = 3e-4
GRAD_CLIP = 1.0
LOG_EVERY = 20

# Model — small from-scratch decoder-only transformer
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
D_FF = 1024
DROP_PROB = 0.1


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
# GPT-2 BPE tokenizer is a convenient default. It uses `<|endoftext|>` as both
# BOS and EOS; we register the same token as PAD so the existing pad-mask logic
# in the Transformer keeps working.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token       # reuse <|endoftext|>
PAD_ID = tokenizer.pad_token_id                 # == eos_token_id
EOS_ID = tokenizer.eos_token_id
BOS_ID = tokenizer.bos_token_id or EOS_ID
VOCAB_SIZE = tokenizer.vocab_size               # 50257


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def format_example(question: str, answer: str) -> str:
    """Concatenate a QA pair into a single causal-LM training string."""
    return f"Question: {question}\nAnswer: {answer}"


class TOFUDataset(Dataset):
    """
    Wraps the HuggingFace TOFU dataset, tokenizing each QA pair into a
    fixed-length tensor padded with the EOS/PAD token.
    """

    def __init__(self, hf_dataset, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

        texts = [
            format_example(row["question"], row["answer"])
            for row in hf_dataset
        ]
        encoded = tokenizer(
            texts,
            max_length=max_len - 1,     # leave 1 slot for the trailing EOS
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        # Append EOS and right-pad to `max_len`
        self.input_ids = []
        for ids in encoded:
            ids = ids + [EOS_ID]
            ids = ids + [PAD_ID] * (max_len - len(ids))
            self.input_ids.append(ids[:max_len])

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.input_ids[idx]


# ---------------------------------------------------------------------------
# Train / Generate
# ---------------------------------------------------------------------------
def train(model: DecoderOnlyTransformer, loader: DataLoader) -> None:
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))

    model.train()
    step = 0
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)                # [B, T]
            inputs = batch[:, :-1]                  # [B, T-1]
            targets = batch[:, 1:]                  # [B, T-1]

            logits = model(inputs)                  # [B, T-1, V]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running += loss.item()
            step += 1
            if step % LOG_EVERY == 0:
                avg = running / LOG_EVERY
                ppl = math.exp(min(avg, 20))
                print(f"epoch {epoch} | step {step:>5d} | loss {avg:.4f} | ppl {ppl:8.2f}")
                running = 0.0


@torch.no_grad()
def generate(model: DecoderOnlyTransformer, question: str, max_new_tokens: int = 80) -> str:
    model.eval()
    prompt = f"Question: {question}\nAnswer:"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        strategy="sample",
        temperature=0.8,
        top_k=50,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)

    print(f"device: {DEVICE}")
    print(f"loading dataset {DATASET_NAME} ({DATASET_CONFIG}) ...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    if NUM_TRAIN_SAMPLES is not None:
        ds = ds.select(range(min(NUM_TRAIN_SAMPLES, len(ds))))
    print(f"  examples: {len(ds)}  | sample fields: {list(ds.column_names)}")

    train_ds = TOFUDataset(ds, tokenizer, MAX_LEN)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    config = ModuleConfig(
        pad_idx=PAD_ID,
        sos_idx=BOS_ID,
        eos_idx=EOS_ID,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        d_qk=D_MODEL // N_HEADS,
        d_v=D_MODEL // N_HEADS,
        max_len=MAX_LEN,
        drop_prob=DROP_PROB,
    )
    model = DecoderOnlyTransformer(vocab_size=VOCAB_SIZE, config=config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f}M")

    print("training ...")
    train(model, loader)

    print("\nsample generations:")
    for q in [
        "What is the full name of the author born in Taipei, Taiwan on 05/11/1991?",
        "Where was the author Hina Ameen born?",
        "What genre does the author primarily write in?",
    ]:
        print("-" * 60)
        print(generate(model, q))


if __name__ == "__main__":
    main()
