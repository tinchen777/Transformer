"""
标准 SFT (Supervised Fine-Tuning) 脚本 —— 以 LLaMA 为例

本脚本演示的核心要点（对应前面讨论的内容）：
  1. prompt 与 response 拼接成一条因果序列；
  2. attention_mask 指的是 *padding mask*，由 collate 自动生成（不是因果 mask）；
  3. 因果 mask 由 LlamaForCausalLM 内部自动处理，无需手动构造；
  4. 用 *label mask*（把 prompt 部分的 label 设为 -100 / ignore_index）
     来实现「只对 response 计算 loss」—— 这才是区分 prompt/response 的正确机制，
     而不是去改 attention mask 让 prompt 变双向。

依赖：torch, transformers, datasets
  pip install torch transformers datasets
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss 的默认 ignore_index


# ----------------------------------------------------------------------------- #
# 1. 把单条样本编码成 input_ids + labels
# ----------------------------------------------------------------------------- #
def encode_example(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_len: int = 2048,
) -> dict[str, list[int]]:
    """
    把一条 {prompt, response} 编码为 token 序列，并构造 label mask。

    关键点：prompt 段的 labels 全部设为 IGNORE_INDEX，
    这样交叉熵 loss 只在 response 段（含结束符）上计算。
    """
    # 用聊天模板拼出 prompt 文本（不同模型模板不同，这里用通用接口）
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,  # 在末尾补上 assistant 起始标记
    )
    # response 文本 + 结束符（让模型学会何时停止）
    response_text = example["response"] + tokenizer.eos_token

    # 分别编码：注意 prompt 与 response 都不额外加特殊 token，避免重复
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + response_ids

    # 这里就是「前缀边界」prefix_len = len(prompt_ids) 的用武之地：
    # prompt 段 label 设为 -100（不计 loss），response 段保留真实 token id。
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids[:]

    # 截断（从右侧截，保留 prompt 开头与尽量多的 response）
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]

    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    """在内存中持有已编码样本的简单数据集。"""

    def __init__(self, raw_examples, tokenizer, max_len: int = 2048):
        self.data = [encode_example(ex, tokenizer, max_len) for ex in raw_examples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.data[idx]


# ----------------------------------------------------------------------------- #
# 2. collate_fn —— 在这里做 padding，并生成 padding mask
# ----------------------------------------------------------------------------- #
@dataclass
class SFTCollator:
    """
    把一个 batch 的变长样本 pad 到同一长度。

    - input_ids 用 pad_token_id 填充；
    - labels 用 IGNORE_INDEX 填充（padding 位置不该计 loss）；
    - attention_mask（= padding mask）：真实 token=1，padding=0。
      因果 mask 不在这里生成，由模型内部自动处理。

    采用左侧还是右侧 padding：训练时常用右 padding；若同时要做生成，
    decoder-only 模型推理一般用左 padding。这里按训练用右 padding。
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        batch_input_ids, batch_labels, batch_attn = [], [], []
        for f in features:
            ids, labels = f["input_ids"], f["labels"]
            n_pad = max_len - len(ids)

            batch_input_ids.append(ids + [pad_id] * n_pad)
            batch_labels.append(labels + [IGNORE_INDEX] * n_pad)
            # padding mask：真实位置 1，pad 位置 0
            batch_attn.append([1] * len(ids) + [0] * n_pad)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
        }


# ----------------------------------------------------------------------------- #
# 3. 训练入口
# ----------------------------------------------------------------------------- #
def main():
    model_name = "meta-llama/Llama-3.2-1B"  # 换成你有权限的任意 LLaMA 系模型

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # LLaMA 系 tokenizer 默认没有 pad_token，用 eos 兜底是常见做法
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # 有 FA2 可打开，因果性由 kernel 处理
    )
    # 因果 mask 由 LlamaForCausalLM 内部生成 —— 我们从不手动传它。

    # 示例数据（实际换成你的数据集，例如 datasets.load_dataset(...)）
    raw_train = [
        {"prompt": "用一句话解释什么是注意力机制。",
         "response": "注意力机制让模型在生成每个 token 时按相关性对输入的不同部分加权聚焦。"},
        {"prompt": "把 'hello world' 翻译成中文。",
         "response": "你好，世界。"},
    ]

    train_dataset = SFTDataset(raw_train, tokenizer, max_len=2048)
    collator = SFTCollator(tokenizer)

    args = TrainingArguments(
        output_dir="./sft-llama-out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,  # 省显存，长序列训练常用
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        # 注意：不要传 tokenizer 让 Trainer 自己 pad —— 我们已在 collator 里处理 labels
    )

    trainer.train()
    trainer.save_model("./sft-llama-out/final")
    tokenizer.save_pretrained("./sft-llama-out/final")


if __name__ == "__main__":
    main()
