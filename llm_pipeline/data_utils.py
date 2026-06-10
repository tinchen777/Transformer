# -*- coding: utf-8 -*-
"""
data_utils.py —— 数据下载、分词、DataLoader 构造

负责整个数据侧的流水线，微调和 unlearning 共用：

  HuggingFace Hub 下载原始数据（load_dataset）
        │
        ▼
  把每条 {question, answer} 拼成因果语言模型的训练序列并分词
        │   input_ids:  Question: ... \nAnswer: ... <eos>
        │   labels:     [-100, -100, ...,  答案部分的真实 token id]
        ▼
  DataCollatorForSeq2Seq 在组 batch 时动态 padding
        │
        ▼
  DataLoader（unlearning 用）或直接交给 Trainer（微调用）

关键概念（也是 SFT 数据处理的核心）：
  1. 因果语言模型的训练样本就是一条长序列，prompt 和 response 拼在一起；
  2. labels 里把 prompt 段设成 -100：PyTorch 的交叉熵默认忽略 -100，
     于是 loss 只在「答案」部分计算 —— 模型学的是「给定问题，生成答案」，
     而不是去背诵问题本身；
  3. attention_mask 是 padding mask（真实 token=1，pad=0），
     因果 mask 由模型内部自动构造，数据侧不用管。
"""

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# PyTorch CrossEntropyLoss 默认的 ignore_index：labels 中等于它的位置不计 loss
IGNORE_INDEX = -100

# TOFU 数据是英文 QA，统一用这个模板拼接（和 train_tofu.py 保持一致）
PROMPT_TEMPLATE = "Question: {question}\nAnswer: "


def format_prompt(question: str) -> str:
    """把一个问题套进模板，得到喂给模型的 prompt 文本（不含答案）。"""
    return PROMPT_TEMPLATE.format(question=question)


# ----------------------------------------------------------------------------- #
# 1. 下载原始数据
# ----------------------------------------------------------------------------- #
def load_raw_dataset(dataset_name: str, config_name: str, num_samples: int | None = None):
    """
    从 HuggingFace Hub 下载数据集。

    第一次运行会真正联网下载并缓存到 ~/.cache/huggingface/datasets，
    之后再调用会直接读本地缓存，不会重复下载。

    参数
    ----
    dataset_name : 例如 "locuslab/TOFU"
    config_name  : TOFU 的子集名，"full" / "forget10" / "retain90" 等
    num_samples  : 只取前 N 条（调试用）；None 表示全量
    """
    ds = load_dataset(dataset_name, name=config_name, split="train")
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    print(f"[data] 加载 {dataset_name}/{config_name}: {len(ds)} 条样本, "
          f"字段 = {ds.column_names}")
    return ds


# ----------------------------------------------------------------------------- #
# 2. 分词：文本 -> input_ids + labels
# ----------------------------------------------------------------------------- #
def tokenize_dataset(raw_dataset, tokenizer: PreTrainedTokenizerBase, max_len: int):
    """
    把原始 {question, answer} 数据集整体分词。

    返回的每条样本只有两列：
      input_ids : 问题 + 答案 + <eos> 的 token id 序列
      labels    : 与 input_ids 等长；问题段全是 -100，答案段是真实 id

    注意这里还没有 padding！padding 留到组 batch 时由 collator 动态完成
    （pad 到 batch 内最长即可，比一律 pad 到 max_len 省很多计算）。
    """

    def _tokenize_batch(batch):
        input_ids_list, labels_list = [], []
        for q, a in zip(batch["question"], batch["answer"]):
            # 问题段（prompt）：单独分词，目的是知道它有多长，
            # 这个长度就是 labels 里 -100 的个数（"前缀边界"）
            q_ids = tokenizer(format_prompt(q), add_special_tokens=False)["input_ids"]
            # 答案段：末尾拼上 eos_token，模型才能学会「答完要停下来」
            a_ids = tokenizer(a + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

            input_ids = q_ids + a_ids
            # label mask：问题段 -100（不计 loss），答案段保留真实 token id
            labels = [IGNORE_INDEX] * len(q_ids) + a_ids

            # 截断到 max_len（从右侧截：保留完整问题 + 尽量多的答案）
            input_ids_list.append(input_ids[:max_len])
            labels_list.append(labels[:max_len])
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized = raw_dataset.map(
        _tokenize_batch,
        batched=True,                              # 一次处理一批，比逐条快
        remove_columns=raw_dataset.column_names,   # 丢掉原始文本列，只留 token
        desc="tokenizing",
    )
    return tokenized


# ----------------------------------------------------------------------------- #
# 3. collator：组 batch 时动态 padding
# ----------------------------------------------------------------------------- #
def build_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorForSeq2Seq:
    """
    DataCollatorForSeq2Seq 会在每个 batch 内：
      - input_ids 用 pad_token_id 右填充到 batch 内最长；
      - labels 用 -100 填充（pad 位置当然也不该计 loss）；
      - 自动生成 attention_mask（真实 token=1，pad=0）。

    名字里虽然带 Seq2Seq，但它对「带 labels 的因果 LM 数据」同样适用，
    是 HF 生态里处理变长 labels 最方便的现成 collator。
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,                # 动态 pad 到 batch 内最长
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,        # pad 到 8 的倍数，对 GPU tensor core 更友好
    )


# ----------------------------------------------------------------------------- #
# 4. 一步到位的便捷函数
# ----------------------------------------------------------------------------- #
def build_tokenized_dataset(
    dataset_name: str,
    config_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    num_samples: int | None = None,
):
    """下载 + 分词，返回可直接喂给 Trainer / DataLoader 的数据集。"""
    raw = load_raw_dataset(dataset_name, config_name, num_samples)
    return tokenize_dataset(raw, tokenizer, max_len), raw


def build_dataloader(
    tokenized_dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    把分词后的数据集包成 PyTorch DataLoader（unlearning 的手写训练循环用；
    微调走 Trainer 时不需要这个，Trainer 内部会自己建 DataLoader）。
    """
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_collator(tokenizer),
    )
