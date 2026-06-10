# -*- coding: utf-8 -*-
"""
common.py —— 四个 baseline 共享的配置基类和工具函数

所有 baseline 的实验设置（模型、数据、评估）保持完全一致，
只有"遗忘损失"不同 —— 这正是做 baseline 对比实验的基本要求：
控制变量，只比方法本身。
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from llm_pipeline.config import DEVICE
from llm_pipeline.data_utils import IGNORE_INDEX, build_dataloader, build_tokenized_dataset
from llm_pipeline.evaluate import compute_perplexity


# ----------------------------------------------------------------------------- #
# 1. 配置基类：所有方法共享的字段（数据、模型、评估），子类只加方法特有超参
# ----------------------------------------------------------------------------- #
@dataclass
class BaseUnlearnConfig:
    # ---- 模型：与 llm_pipeline 微调阶段保持一致 ----
    model_name: str = "gpt2"
    finetuned_adapter_dir: str = "outputs/finetune_lora/final"

    # ---- 数据：TOFU 的标准 forget/retain 切分 ----
    dataset_name: str = "locuslab/TOFU"
    forget_config: str = "forget10"
    retain_config: str = "retain90"
    max_len: int = 256
    num_eval_samples: int = 200     # 评估 PPL 时 forget/retain 各抽多少条

    # ---- 训练通用 ----
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    seed: int = 42

    # ---- 评估展示 ----
    max_new_tokens: int = 64
    demo_question_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])


# ----------------------------------------------------------------------------- #
# 2. 数据准备：所有 baseline 用同一套 forget/retain loader
# ----------------------------------------------------------------------------- #
@dataclass
class UnlearnData:
    """打包好的数据：训练 loader + 评估 loader + 展示用 QA + 原始分词数据集。"""
    forget_tokenized: object
    retain_tokenized: object
    forget_train_loader: object
    retain_train_loader: object
    forget_eval_loader: object
    retain_eval_loader: object
    demo_qa: list


def build_unlearn_data(cfg: BaseUnlearnConfig, tokenizer) -> UnlearnData:
    """下载 + 分词 forget/retain 两个子集，构造训练和评估 DataLoader。"""
    forget_tok, forget_raw = build_tokenized_dataset(
        cfg.dataset_name, cfg.forget_config, tokenizer, cfg.max_len)
    retain_tok, _ = build_tokenized_dataset(
        cfg.dataset_name, cfg.retain_config, tokenizer, cfg.max_len)

    n = cfg.num_eval_samples
    return UnlearnData(
        forget_tokenized=forget_tok,
        retain_tokenized=retain_tok,
        forget_train_loader=build_dataloader(forget_tok, tokenizer, cfg.batch_size, shuffle=True),
        retain_train_loader=build_dataloader(retain_tok, tokenizer, cfg.batch_size, shuffle=True),
        # 评估集取固定前 N 条且不 shuffle —— 保证"遗忘前/后"评估的是同一批数据
        forget_eval_loader=build_dataloader(
            forget_tok.select(range(min(n, len(forget_tok)))), tokenizer, cfg.batch_size, shuffle=False),
        retain_eval_loader=build_dataloader(
            retain_tok.select(range(min(n, len(retain_tok)))), tokenizer, cfg.batch_size, shuffle=False),
        demo_qa=[forget_raw[i] for i in cfg.demo_question_indices],
    )


def batch_to_device(batch: dict, device: str = DEVICE) -> dict:
    """把 collator 产出的 batch（dict of tensor）整体搬到目标设备。"""
    return {k: v.to(device) for k, v in batch.items()}


# ----------------------------------------------------------------------------- #
# 3. 序列对数概率：偏好类方法（NPO 等）的核心积木
# ----------------------------------------------------------------------------- #
def answer_log_prob(model, batch: dict) -> torch.Tensor:
    """
    计算每条样本"答案部分"的对数概率之和 log π(y|x)，返回形状 [B]。

    这是 DPO/NPO 这类偏好优化方法的基本量：
      log π(y|x) = Σ_t log p(y_t | x, y_<t)
    只对 labels != -100 的位置（即答案 token）求和，prompt 不算。

    实现要点（因果 LM 的"右移一位"）：
      logits[:, t] 预测的是第 t+1 个 token，
      所以 logits 去掉最后一位、labels 去掉第一位，二者按位对齐。
    """
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits[:, :-1, :]          # [B, T-1, V]
    labels = batch["labels"][:, 1:]            # [B, T-1]

    mask = labels != IGNORE_INDEX              # 答案 token 的位置
    # gather 需要合法下标，先把 -100 替换成 0（反正之后会被 mask 掉）
    safe_labels = labels.masked_fill(~mask, 0)

    # 用 float32 算 log_softmax，避免 bf16 下数值误差
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    return (token_log_probs * mask).sum(dim=-1)  # [B]


def token_kl_to_ref(model, ref_model, batch: dict) -> torch.Tensor:
    """
    当前模型与参考模型在每个有效 token 上的 KL 散度（取平均），标量。

    KL( π_ref || π_θ ) = Σ_v p_ref(v) · (log p_ref(v) − log p_θ(v))
    用在 retain 集上：约束当前模型的输出分布不要偏离参考模型太远，
    是比"retain NLL"更软的一种保持手段（NPO-KL 变体用它）。
    """
    logits = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
    with torch.no_grad():
        ref_logits = ref_model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()

    mask = (batch["labels"][:, 1:] != IGNORE_INDEX)          # [B, T-1]
    log_p = F.log_softmax(logits, dim=-1)
    ref_log_p = F.log_softmax(ref_logits, dim=-1)
    kl = (ref_log_p.exp() * (ref_log_p - log_p)).sum(-1)     # [B, T-1]
    return (kl * mask).sum() / mask.sum().clamp(min=1)


# ----------------------------------------------------------------------------- #
# 4. 评估快照 + 前后对比汇总（所有 baseline 输出统一格式，方便横向比较）
# ----------------------------------------------------------------------------- #
def evaluate_snapshot(model, data: UnlearnData, tag: str) -> dict:
    """同时评估 forget / retain PPL 并打印一行摘要。"""
    forget = compute_perplexity(model, data.forget_eval_loader, DEVICE)
    retain = compute_perplexity(model, data.retain_eval_loader, DEVICE)
    print(f"\n[eval/{tag}] forget: loss={forget['loss']:.4f} ppl={forget['perplexity']:.2f} | "
          f"retain: loss={retain['loss']:.4f} ppl={retain['perplexity']:.2f}")
    return {"forget": forget, "retain": retain}


def print_summary(method: str, before: dict, after: dict) -> None:
    """打印遗忘前后 forget/retain PPL 的对比小表。"""
    print(f"\n{'=' * 58}\n方法: {method}")
    print(f"{'指标':<24}{'遗忘前':>12}{'遗忘后':>12}")
    print("-" * 58)
    print(f"{'forget PPL (希望↑)':<24}"
          f"{before['forget']['perplexity']:>14.2f}{after['forget']['perplexity']:>14.2f}")
    print(f"{'retain PPL (希望≈不变)':<24}"
          f"{before['retain']['perplexity']:>14.2f}{after['retain']['perplexity']:>14.2f}")
    print("=" * 58)
