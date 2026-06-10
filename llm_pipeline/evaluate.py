# -*- coding: utf-8 -*-
"""
evaluate.py —— 评估工具：困惑度 + 生成样例

微调和 unlearning 共用两类评估：

  1. 困惑度（perplexity, PPL）—— 定量指标
     PPL = exp(平均每个 token 的交叉熵 loss)。
     直觉：模型给「正确答案的每个 token」分配的平均概率的倒数。
       - 微调阶段：训练集/验证集 PPL 应该下降（模型记住了 TOFU 知识）；
       - unlearning 阶段：forget 集 PPL 应该上升（忘掉了），
         retain 集 PPL 应该基本不变（没伤到其他能力）。
     这一对指标就是 unlearning 论文里最基本的 forget quality vs utility。

  2. 生成样例 —— 定性观察
     拿几个数据集里的问题让模型实际生成答案，肉眼对比
     微调前/后、遗忘前/后的回答变化，比一个数字更直观。
"""

import math

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .data_utils import IGNORE_INDEX, format_prompt


# ----------------------------------------------------------------------------- #
# 1. 困惑度
# ----------------------------------------------------------------------------- #
@torch.no_grad()  # 评估不需要梯度，关掉可以省显存、加速
def compute_perplexity(model, dataloader: DataLoader, device: str) -> dict[str, float]:
    """
    在一个 dataloader 上计算平均 loss 和困惑度。

    细节：模型 forward 传入 labels 时返回的 output.loss 是
    「该 batch 内所有有效 token 的平均交叉熵」。不同 batch 的有效 token
    数不同，直接对 batch loss 求平均会有偏差，所以这里先乘回 token 数
    再统一除 —— 得到真正的「逐 token 平均」。
    """
    model.eval()  # 切到评估模式（关闭 dropout 等）
    total_loss, total_tokens = 0.0, 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)  # 传入 labels，模型内部自动算交叉熵

        # 有效 token 数：labels != -100 的位置。
        # 因果 LM 是用第 t 个位置预测第 t+1 个 token，模型内部会把 labels
        # 左移一位再算 loss，所以参与 loss 的是 labels[:, 1:] 里的有效位置。
        n_valid = (batch["labels"][:, 1:] != IGNORE_INDEX).sum().item()
        total_loss += output.loss.item() * n_valid
        total_tokens += n_valid

    avg_loss = total_loss / max(total_tokens, 1)
    # exp 里截断一下，防止 loss 极大时溢出（unlearning 后 forget 集会出现）
    ppl = math.exp(min(avg_loss, 30))
    return {"loss": avg_loss, "perplexity": ppl}


# ----------------------------------------------------------------------------- #
# 2. 生成样例
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def generate_answer(
    model,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    """
    给一个问题，让模型生成答案（贪心解码，结果可复现）。

    只返回新生成的部分（把 prompt 从输出里切掉），方便对比。
    """
    model.eval()
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                        # 贪心：每步取概率最大的 token
        pad_token_id=tokenizer.pad_token_id,    # 显式传入，避免警告
    )
    # output_ids 包含 prompt + 新生成内容；按 prompt 长度切掉前缀
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def show_generations(
    model,
    tokenizer: PreTrainedTokenizerBase,
    qa_pairs: list[dict],
    device: str,
    max_new_tokens: int = 64,
    title: str = "",
) -> None:
    """
    打印若干 (问题, 标准答案, 模型生成) 三元组，用于人工对比。

    qa_pairs : [{"question": ..., "answer": ...}, ...]，
               直接传 TOFU 原始数据集的若干条即可。
    """
    print(f"\n{'=' * 70}\n生成样例对比: {title}\n{'=' * 70}")
    for qa in qa_pairs:
        pred = generate_answer(model, tokenizer, qa["question"], device, max_new_tokens)
        print(f"[问题]     {qa['question']}")
        print(f"[标准答案] {qa['answer']}")
        print(f"[模型生成] {pred}")
        print("-" * 70)
