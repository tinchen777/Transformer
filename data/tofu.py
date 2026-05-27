
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


MAX_LEN    = 128

# ============================================================
# 1. 加载数据集
# ============================================================
dataset = load_dataset("locuslab/TOFU", name="full", split="train")


# ============================================================
# 2. 分词器
# ============================================================
"""
bos_token beginning of sequence 序列开始符
eos_token end of sequence序列结束符
pad_token padding 补齐 batch
unk_token unknown 词表外的 token
sep_token separator 分隔两个句子（BERT 类）
cls_token classification 句首分类位（BERT 类）
mask_token mask MLM 掩码位（BERT 类）
additional_special_tokens 额外自定义用户/模型自定义的特殊 token
"""

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
})
PAD_ID = tokenizer.pad_token_id
BOS_ID = tokenizer.bos_token_id
EOS_ID = tokenizer.eos_token_id
UNK_ID = tokenizer.unk_token_id
VOCAB_SIZE = len(tokenizer)  # 用 len 而不是 vocab_size, 这样以后加新特殊 token 也安全


# ============================================================
# 3. 索引化 (文本 -> token id)
# ============================================================
def tokenize_function(batch):
    input_ids_list, labels_list = [], []
    for q, a in zip(batch["question"], batch["answer"]):
        q_ids = tokenizer(f"Question: {q}\nAnswer: ", add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(f"{a}{tokenizer.eos_token}",   add_special_tokens=False)["input_ids"]
        input_ids = q_ids + a_ids
        labels = [-100] * len(q_ids) + a_ids  # Q 段全部 -100, 让模型不计算 Q 段的 loss
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "labels": labels_list}


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,   # 去掉原始 question/answer 列
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,                  # 动态 pad 到 batch 内最长
    label_pad_token_id=-100,       # labels 用 -100 pad
    pad_to_multiple_of=8,
)

# 关键: 告诉 HF dataset 用 torch tensor 返回, 只保留这两列
# (现在每条样本: input_ids [T], attention_mask [T], 都是 int64 tensor)
# tokenized_datasets.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask"],
# )


if __name__ == "__main__":
    print("PAD_ID:", PAD_ID)
    print("BOS_ID:", BOS_ID)
    print("EOS_ID:", EOS_ID)
    print("UNK_ID:", UNK_ID)
    print("VOCAB_SIZE:", VOCAB_SIZE)

    print(tokenizer.special_tokens_map)

    print("demo TOFU dataset and tokenizer...")
    print(tokenized_datasets.features)

    # 直接获取第一个样本（dataset[0] 返回一个 dict），不要迭代 dict 的键
    # sample = tokenized_datasets[0]
    # print(sample)
    # input_ids = sample["input_ids"]
    # attention_mask = sample["attention_mask"]
    # print(input_ids)
    # print(attention_mask)

    # # tokenizer.decode 需要 list[int] 或可迭代的 id 序列
    # try:
    #     ids = input_ids.tolist()
    # except Exception:
    #     ids = input_ids
    # text = tokenizer.convert_ids_to_tokens(ids)
    # print(text)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        tokenized_datasets,  # type:ignore
        batch_size=8,
        shuffle=True,
        collate_fn=data_collator,     # <-- 关键就这一行
        num_workers=0,
        # pin_memory=True,
    )
    print("demo dataloader...")
    for batch in loader:
        input_ids      = batch["input_ids"]       # [bsz, L]
        attention_mask = batch["attention_mask"]  # [bsz, L], pad mask
        labels         = batch["labels"]          # [bsz, L], Q 段和 pad 段是 -100
        print("input_ids:", input_ids, "shape:", input_ids.shape)
        print("attention_mask:", attention_mask, "shape:", attention_mask.shape)
        print("labels:", labels, "shape:", labels.shape)
        break
