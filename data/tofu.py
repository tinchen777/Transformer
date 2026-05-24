
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


MAX_LEN    = 128

# ============================================================
# 1. 加载数据集
# ============================================================
dataset = load_dataset("locuslab/TOFU", name="full", split="train")
print(dataset[0])


# ============================================================
# 2. 分词器
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token       # reuse <|endoftext|>

PAD_ID     = tokenizer.pad_token_id             # == eos_token_id
EOS_ID     = tokenizer.eos_token_id
BOS_ID     = tokenizer.bos_token_id or EOS_ID
VOCAB_SIZE = len(tokenizer)                     # 用 len 而不是 vocab_size,
                                                # 这样以后加新特殊 token 也安全

print(PAD_ID, EOS_ID, BOS_ID, VOCAB_SIZE, len(tokenizer))


# ============================================================
# 3. 索引化 (文本 -> token id)
# ============================================================
def tokenize_function(batch):
    # batched=True 时, batch["question"] / batch["answer"] 都是 list[str]
    # 用清晰的分隔符,末尾加 eos 让模型学会停下
    texts = [
        f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}"
        for q, a in zip(batch["question"], batch["answer"])
    ]
    return tokenizer(
        texts,                          # tokenizer 接收 list[str] 合法
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,   # 去掉原始 question/answer 列
)

print(tokenized_datasets[0])


# 关键: 告诉 HF dataset 用 torch tensor 返回, 只保留这两列
# (现在每条样本: input_ids [T], attention_mask [T], 都是 int64 tensor)
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"],
)
 
print("单条样本:", {k: v.shape for k, v in tokenized_datasets[0].items()})
# 应该看到: input_ids torch.Size([128]), attention_mask torch.Size([128])


# ============================================================
# 4. DataLoader: 把 N 条 [T] 样本堆成 [B, T] 矩阵
# ============================================================
# HF dataset 直接可以当 PyTorch Dataset 用,
# 默认 collate_fn 会把 dict-of-tensors 自动 stack 成 [B, T]
loader = DataLoader(
    tokenized_datasets,  # type:ignore
    batch_size=8,
    shuffle=True,
    num_workers=0,        # Windows / notebook 上先用 0,Linux 可以调大
    pin_memory=False,      # 用 GPU 时建议 True,加速 host->device 拷贝
    drop_last=True,       # 训练时丢掉最后不够一个 batch 的尾巴
)


# ============================================================
# 4. 看一眼 batch 长什么样
# ============================================================
batch = next(iter(loader))
print("\nbatch keys:", list(batch.keys()))
print("input_ids      :", batch["input_ids"].shape,      batch["input_ids"].dtype)
print("attention_mask :", batch["attention_mask"].shape, batch["attention_mask"].dtype)
# input_ids:      torch.Size([8, 128]) torch.int64
# attention_mask: torch.Size([8, 128]) torch.int64
#
# 这就是你模型 forward 要接收的矩阵.
# 进入模型后, self.token_emb(input_ids) 会把它变成 [8, 128, d_model] 的 float 张量,
# 那一步才是 "token embedding".
 
 
# ============================================================
# 5. 训练循环里怎么用 (示意)
# ============================================================
# for batch in loader:
#     input_ids      = batch["input_ids"].to(device)      # [B, T] int64
#     attention_mask = batch["attention_mask"].to(device) # [B, T] int64 (1=real, 0=pad)
#
#     logits = model(input_ids, attention_mask=attention_mask)  # [B, T, V] float
#
#     # causal LM: 用 t 位置预测 t+1 位置
#     shift_logits = logits[:, :-1, :].contiguous()
#     shift_labels = input_ids[:, 1:].contiguous()
#     shift_labels = shift_labels.masked_fill(shift_labels == PAD_ID, -100)
#
#     loss = F.cross_entropy(
#         shift_logits.reshape(-1, VOCAB_SIZE),
#         shift_labels.reshape(-1),
#         ignore_index=-100,
#     )
#     loss.backward(); optimizer.step(); optimizer.zero_grad()


if __name__ == "__main__":
    for batch in loader:
        print("input_ids      :", batch["input_ids"].shape,      batch["input_ids"].dtype)
        print("attention_mask :", batch["attention_mask"].shape, batch["attention_mask"].dtype)
        break
