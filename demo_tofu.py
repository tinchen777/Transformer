
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
import csv
from tqdm import tqdm
import yaml
import scopos

from data.tofu import *
from transformer import DecoderOnlyTransformer, ModuleConfig


# DEVICE = 'cpu'
DEVICE = 'cuda:0'
# DEVICE = 'mps'  # macbook的GPU

# Model hyperparameters
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_FF = 4096
DROP_PROB = 0.1

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 1000
LR = 3e-4
GRAD_CLIP = 1.0
SAVE_EVERY = 10

NAME = "D_FF_4096"
TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_FOLDER = f"save/TOFU/{TAG}{NAME}"
os.makedirs(SAVE_FOLDER, exist_ok=True)


######
# DATA
######
loader = DataLoader(
    tokenized_datasets,  # type:ignore
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,        # Windows / notebook 上先用 0,Linux 可以调大
    pin_memory=False,      # 用 GPU 时建议 True,加速 host->device 拷贝
    drop_last=True,       # 训练时丢掉最后不够一个 batch 的尾巴
    collate_fn=data_collator,  # <-- 关键就这一行
)

#######
# MODEL
#######
model = DecoderOnlyTransformer(
    vocab_size=VOCAB_SIZE,
    config=ModuleConfig(
        pad_idx=PAD_ID,
        bos_idx=BOS_ID,
        eos_idx=EOS_ID,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        drop_prob=DROP_PROB,
        max_len=MAX_LEN,
    )
).to(DEVICE)

print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"  model params: {n_params/1e6:.2f}M")

# save config
with open(f"{SAVE_FOLDER}/config.yaml", "w") as f:
    yaml.safe_dump({
        **model.decoder.config.__dict__,
        "vocab_size": VOCAB_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "grad_clip": GRAD_CLIP,
    }, f)

def train(model: DecoderOnlyTransformer):
    print("开始训练Decoder-Only Transformer模型...")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))

    model.train()

    scopos.update(stage="training", epoch=scopos.progress(total=EPOCHS), loss="N/A")

    with open(f"{SAVE_FOLDER}/result.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Iter", "Loss"])

        for epoch in range(1, EPOCHS + 1):
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"[Training TOFU] Epoch {epoch:03d}/{EPOCHS:03d}")
            # pbar.set_description(f"[Training TOFU] Epoch {epoch:03d}/{EPOCHS:03d}")

            for i, batch in pbar:
                input_ids = batch["input_ids"].to(DEVICE)          # [bsz, len_seq]
                attention_mask = batch["attention_mask"].to(DEVICE) # [bsz, len_seq]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [bsz, 1, 1, len_seq], 扩展成 4D 以适配模型的 mask 输入
                labels = batch["labels"].to(DEVICE)                # [bsz, len_seq], Q 段和 pad 段是 -100

                # forward
                logits = model(input_ids, padding_mask=attention_mask)  # [bsz, len_seq, vocab_size]

                # shift: 用位置 i 的输出预测位置 i+1 的 token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = criterion(
                    shift_logits.reshape(-1, logits.size(-1)),
                    shift_labels.reshape(-1),
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_str = f"{epoch:04d}"
                iter_str = f"{i:04d}"
                loss_str = f"{loss.item():.6f}"
                pbar.set_postfix_str(f"Loss: {loss_str}")
                # writerow: 记录 epoch, iter, loss 到 csv 文件
                writer.writerow([epoch_str, iter_str, loss_str])

                # # Decoder-Only 模型的输入和输出都是 input_ids, 但训练时要错开一格
                # dec_inputs = input_ids[:, :-1]       # [bsz, len_seq-1], 包含 sos, 不包含 eos
                # dec_outputs = input_ids[:, 1:]       # [bsz, len_seq-1], 包含 eos, 不包含 sos
                # dec_attention_mask = attention_mask[:, :-1]  # [bsz, len_seq-1]

                # logits = model(dec_inputs, dec_attention_mask)  # [bsz, len_seq-1, vocab_size]
                # loss = criterion(logits.view(-1, VOCAB_SIZE), dec_outputs.view(-1))

            scopos.update(stage="training", epoch=scopos.progress(value=epoch, total=EPOCHS), loss=loss_str)

            # save model checkpoint
            if epoch % SAVE_EVERY == 0:
                checkpoint_folder = f"{SAVE_FOLDER}/checkpoints"
                os.makedirs(checkpoint_folder, exist_ok=True)
                torch.save(model.state_dict(), f"{checkpoint_folder}/tofu_model_full_{epoch}.pth")
                print(f"Checkpoint saved!")

    return model

# for batch in loader:
#     input_ids      = batch["input_ids"]       # [bsz, L]
#     attention_mask = batch["attention_mask"]  # [bsz, L], pad mask
#     labels         = batch["labels"]          # [bsz, L], Q 段和 pad 段是 -100
    
#     # forward
#     logits = model(input_ids, attention_mask)  # [bsz, L, vocab]
    
#     # shift: 用位置 i 的输出预测位置 i+1 的 token
#     shift_logits = logits[:, :-1, :].contiguous()
#     shift_labels = labels[:, 1:].contiguous()
    
#     loss = criterion(
#         shift_logits.view(-1, vocab_size),
#         shift_labels.view(-1),
#         ignore_index=-100,   # 自动跳过 Q 段和 pad 段
#     )
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
    


model = train(model)


# class MyDecoder(nn.Module):
#     def __init__(self, V, D):
#         super().__init__()
#         self.tok_emb = nn.Embedding(V, D)
#         self.transformer = ...
#         # 没有独立的 lm_head

#     def forward(self, x):
#         h = self.tok_emb(x)                          # 查表: [B, L, D]
#         h = self.transformer(h)
#         logits = h @ self.tok_emb.weight.T           # 转置乘: [B, L, V]
#         return logits

# BUG
# Traceback (most recent call last):
#   File "/data/tianzhen/my_projects/ML/Transformer/demo_tofu.py", line 169, in <module>
#     model = train(model)
#             ^^^^^^^^^^^^
#   File "/data/tianzhen/my_projects/ML/Transformer/demo_tofu.py", line 109, in train
#     logits = model(input_ids, padding_mask=attention_mask)  # [bsz, len_seq, vocab_size]
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/data/tianzhen/.conda/envs/llm/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/data/tianzhen/.conda/envs/llm/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/data/tianzhen/my_projects/ML/Transformer/transformer/models.py", line 312, in forward
#     dec_outputs = self.decoder(
#                   ^^^^^^^^^^^^^
#   File "/data/tianzhen/.conda/envs/llm/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/data/tianzhen/.conda/envs/llm/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/data/tianzhen/my_projects/ML/Transformer/transformer/modules.py", line 232, in forward
#     x = self.pos_emb(x)
#         ^^^^^^^^^^^^^^^
#   File "/data/tianzhen/.conda/envs/llm/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/data/tianzhen/.conda/envs/llm/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/data/tianzhen/my_projects/ML/Transformer/transformer/embedding/positional_encoding.py", line 46, in forward
#     return tok + self.encoding[:len_seq, :]  # type: ignore
#            ~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RuntimeError: The size of tensor a (136) must match the size of tensor b (128) at non-singleton dimension 1