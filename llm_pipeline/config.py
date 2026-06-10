# -*- coding: utf-8 -*-
"""
config.py —— 所有实验超参数的集中配置

为什么单独放一个文件？
  做实验时最常改的就是超参数（学习率、epoch、LoRA 的秩 r 等）。
  把它们集中在 dataclass 里，而不是散落在训练代码各处，
  这样「换一组实验设置」= 「只改这个文件」，训练代码本身不用动。

dataclass 是 Python 自带的轻量写法：字段名 + 类型 + 默认值，
实例化后用 cfg.xxx 访问，非常适合做配置对象。
"""

from dataclasses import dataclass, field

import torch

# ----------------------------------------------------------------------------- #
# 通用：自动选择设备
#   有 GPU 就用 cuda，否则退回 cpu（cpu 上跑小模型小数据也能完整走通流程）
# ----------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型权重精度：
#   GPU 上用 bfloat16 省一半显存且数值稳定；CPU 上 bf16 矩阵乘很慢，用 float32。
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


@dataclass
class FinetuneConfig:
    """LoRA 微调（入口 1：finetune_lora.py）的全部超参数。"""

    # ---------------- 模型 ----------------
    # HuggingFace Hub 上的模型名。默认用 gpt2（124M 参数）：
    #   - 不需要申请权限（Llama 系需要在 HF 上同意协议）；
    #   - 足够小，CPU 也能跑通整个流程。
    # 想换大模型直接改这里，例如：
    #   "meta-llama/Llama-3.2-1B"、"Qwen/Qwen2.5-0.5B"
    # LoRA 的 target_modules 会在 model_utils.py 里根据模型结构自动选择。
    model_name: str = "gpt2"

    # ---------------- 数据 ----------------
    # TOFU 是专门为 LLM Unlearning 设计的基准数据集（虚构作家的问答对）。
    # 微调阶段用 "full"（4000 条 QA），让模型先「记住」这些知识，
    # 之后 unlearning 阶段再让它「忘掉」其中一部分。
    dataset_name: str = "locuslab/TOFU"
    dataset_config: str = "full"
    max_len: int = 256          # 单条样本（问题+答案）的最大 token 数，超出截断
    eval_ratio: float = 0.1     # 从训练集中切 10% 出来做验证集（监控是否过拟合）
    num_train_samples: int | None = None  # 调试时可设小数字（如 200）快速跑通；None = 全量

    # ---------------- LoRA ----------------
    # LoRA 的核心思想：冻结原模型所有权重 W，给指定层旁挂一对低秩矩阵 A、B，
    # 实际前向变成 W·x + (B·A)·x。只训练 A、B（参数量约为全量的 0.1%~1%）。
    lora_r: int = 16            # 低秩矩阵的秩 r：越大可学的「容量」越大，参数也越多
    lora_alpha: int = 32        # 缩放系数，LoRA 增量会乘 alpha/r。经验上常取 2*r
    lora_dropout: float = 0.05  # 对 LoRA 分支做 dropout，轻微正则化

    # ---------------- 训练 ----------------
    output_dir: str = "outputs/finetune_lora"  # checkpoint 和最终 adapter 的保存目录
    num_epochs: float = 3.0
    batch_size: int = 8                # 每张卡每步的样本数
    gradient_accumulation_steps: int = 2  # 梯度累积：等效 batch = 8*2 = 16
    learning_rate: float = 2e-4        # LoRA 只训练很少的新参数，学习率可以比
                                       # 全量微调（约 2e-5）大一个数量级
    warmup_ratio: float = 0.03         # 前 3% 的步数线性升温学习率，训练更稳
    weight_decay: float = 0.01
    logging_steps: int = 20
    seed: int = 42

    # ---------------- 训练后测试 ----------------
    # 微调完成后，用几个 TOFU 里的问题让模型生成答案，肉眼对比微调前后差异
    num_demo_questions: int = 4
    max_new_tokens: int = 64           # 生成答案的最大长度


@dataclass
class UnlearnConfig:
    """梯度上升 Unlearning（入口 2：unlearn_ga.py）的全部超参数。"""

    # ---------------- 模型 ----------------
    # 基座模型必须和微调阶段一致（LoRA adapter 是挂在它上面的）
    model_name: str = "gpt2"
    # 入口 1 训练完保存的 LoRA adapter 路录。Unlearning 从这个「已记住 TOFU
    # 知识」的模型出发，继续更新 LoRA 参数让它忘掉 forget 集。
    finetuned_adapter_dir: str = "outputs/finetune_lora/final"

    # ---------------- 数据 ----------------
    # TOFU 官方把数据切成了 forget / retain 两部分：
    #   forget10  = 要遗忘的 10% 数据（400 条 QA）
    #   retain90  = 其余 90% 数据，遗忘时要尽量保住这部分能力
    dataset_name: str = "locuslab/TOFU"
    forget_config: str = "forget10"
    retain_config: str = "retain90"
    max_len: int = 256
    num_eval_samples: int = 200   # 评估困惑度时各取多少条（全量太慢，抽样即可）

    # ---------------- Unlearning 训练 ----------------
    output_dir: str = "outputs/unlearn_ga"
    num_epochs: int = 3           # 在 forget 集上走几遍
    batch_size: int = 8
    learning_rate: float = 1e-4   # 比微调略小：梯度上升很容易把模型「炸坏」，
                                  # 学习率太大会连 retain 集的能力一起摧毁
    max_grad_norm: float = 1.0    # 梯度裁剪，进一步防止单步更新过猛

    # retain_weight：retain 集梯度下降项的权重 λ。
    #   总损失 = -loss_forget + λ * loss_retain
    #   λ = 0    → 纯梯度上升（GA），最简单但容易整体崩坏；
    #   λ > 0    → GA + 保持项（文献里常叫 Gradient Difference, GD），
    #              一边在 forget 集上「反着学」，一边在 retain 集上正常学，
    #              用来缓解灾难性遗忘。建议先用 1.0 跑一次感受区别。
    retain_weight: float = 1.0

    # 提前停止阈值：forget 集上的 loss 升到这个值就停。
    # 梯度上升没有自然的收敛点（loss 可以无限升高直到模型输出乱码），
    # 必须人为设一个「忘得差不多了」的界限。
    forget_loss_threshold: float = 8.0

    logging_steps: int = 10
    seed: int = 42

    # ---------------- 评估 ----------------
    num_demo_questions: int = 4
    max_new_tokens: int = 64

    # 评估时演示用的问题来源（forget 集里抽几条，看遗忘前后回答的变化）
    demo_question_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
