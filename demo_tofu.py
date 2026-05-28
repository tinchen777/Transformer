
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime

from data.tofu import *
from transformer import DecoderOnlyTransformer, ModuleConfig


# DEVICE = 'cpu'
DEVICE = 'cuda:1'
# DEVICE = 'mps'  # macbook的GPU

# Model hyperparameters
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_FF = 2048
DROP_PROB = 0.1

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 200
LR = 3e-4
GRAD_CLIP = 1.0
LOG_EVERY = 2

TAG = datetime.now().strftime("%Y%m%d_%H%M")

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
    )
).to(DEVICE)

print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"  model params: {n_params/1e6:.2f}M")


def train(model: DecoderOnlyTransformer):
    print("开始训练Decoder-Only Transformer模型...")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))

    model.train()

    for epoch in range(1, EPOCHS + 1):
        running = 0.0

        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)          # [bsz, T]
            attention_mask = batch["attention_mask"].to(DEVICE) # [bsz, T]
            labels = batch["labels"].to(DEVICE)                # [bsz, T], Q 段和 pad 段是 -100

            # forward
            logits = model(input_ids, attention_mask)  # [bsz, L, vocab]
            
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
            
            print('Epoch:', '%04d' % (epoch), 'Iter:', '%04d' % (i), 'loss =', '{:.6f}'.format(loss))

            # # Decoder-Only 模型的输入和输出都是 input_ids, 但训练时要错开一格
            # dec_inputs = input_ids[:, :-1]       # [bsz, T-1], 包含 sos, 不包含 eos
            # dec_outputs = input_ids[:, 1:]       # [bsz, T-1], 包含 eos, 不包含 sos
            # dec_attention_mask = attention_mask[:, :-1]  # [bsz, T-1]

            # logits = model(dec_inputs, dec_attention_mask)  # [bsz, T-1, vocab_size]
            # loss = criterion(logits.view(-1, VOCAB_SIZE), dec_outputs.view(-1))
        # path
        os.makedirs(f"save/{TAG}", exist_ok=True)

        torch.save(model.state_dict(), f"save/{TAG}/tofu_model_full_{epoch}.pth")
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

torch.save(model.state_dict(), f"save/{TAG}/tofu_model_full.pth")

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




