# -*- coding: utf-8 -*-
"""
model_utils.py —— 模型/分词器加载 + LoRA 注入

负责模型侧的三件事，微调和 unlearning 共用：
  1. 从 HuggingFace Hub 下载分词器和基座模型（第一次联网，之后走本地缓存）；
  2. 给基座模型注入 LoRA adapter（微调阶段用）；
  3. 把训练好的 LoRA adapter 重新挂回基座模型（unlearning 阶段用）。

LoRA 速记：
  原模型里某个线性层的权重 W (d×k) 被冻结；旁边加一对小矩阵
  B (d×r) 和 A (r×k)，r 远小于 d、k（比如 r=16）。前向变成：
      h = W·x + (alpha/r) · B·A·x
  只有 A、B 参与训练。训练完保存的 "adapter" 就是这些小矩阵，
  通常只有几 MB，而基座模型动辄几个 GB —— 这就是 LoRA 的省钱之处。
"""

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


# ----------------------------------------------------------------------------- #
# 1. 分词器
# ----------------------------------------------------------------------------- #
def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    下载并配置分词器。

    很多 decoder-only 模型（GPT-2、LLaMA 等）出厂时没有 pad_token，
    因为预训练阶段不需要 padding。微调要组 batch 就必须 pad，
    通用做法是直接复用 eos_token 当 pad_token。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 训练时用右侧 padding（labels 的 -100 对齐逻辑都按右 pad 写的）
    tokenizer.padding_side = "right"
    return tokenizer


# ----------------------------------------------------------------------------- #
# 2. 基座模型
# ----------------------------------------------------------------------------- #
def load_base_model(model_name: str, dtype: torch.dtype, device: str):
    """
    下载基座模型（带语言建模头的因果 LM）。

    AutoModelForCausalLM = 模型主干 + lm_head（把隐藏向量投影回词表 logits），
    并且 forward 时如果传入 labels，会自动完成「右移一位」的交叉熵计算，
    返回的 output.loss 就是标准的 next-token prediction loss。
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] 加载基座模型 {model_name}: {n_params / 1e6:.1f}M 参数, "
          f"dtype={dtype}, device={device}")
    return model


# ----------------------------------------------------------------------------- #
# 3. LoRA 注入
# ----------------------------------------------------------------------------- #
def pick_lora_target_modules(model) -> list[str]:
    """
    根据模型结构选择 LoRA 要挂载的线性层名字。

    不同架构里注意力投影层的命名不同：
      - GPT-2 把 Q/K/V 合并在一个叫 c_attn 的层里；
      - LLaMA / Qwen / Mistral 等是分开的 q_proj / k_proj / v_proj / o_proj。
    经验上挂在注意力的投影层上效果就很好（LoRA 原论文的做法），
    想更激进可以把 MLP 层（如 gate_proj/up_proj/down_proj）也加进来。
    """
    model_type = model.config.model_type  # 例如 "gpt2"、"llama"、"qwen2"
    mapping = {
        "gpt2": ["c_attn"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    if model_type in mapping:
        return mapping[model_type]
    # 不认识的架构就退而求其次：所有线性层都挂（peft 的内置关键字）
    print(f"[model] 未知架构 {model_type}，LoRA 挂载到全部线性层 (all-linear)")
    return "all-linear"  # type: ignore[return-value]


def add_lora(model, r: int, alpha: int, dropout: float):
    """
    给基座模型注入全新的（随机初始化的）LoRA adapter，用于微调阶段。

    get_peft_model 做的事情：
      1. 把基座模型所有参数 requires_grad 设为 False（冻结）；
      2. 在 target_modules 指定的层旁边插入 A、B 小矩阵（可训练）；
      3. 返回包装后的 PeftModel，用法和原模型完全一样（forward/save 等）。
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # 告诉 peft 这是因果语言模型任务
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=pick_lora_target_modules(model),
        bias="none",                    # 不训练 bias（标准做法）
    )
    peft_model = get_peft_model(model, lora_config)
    # 打印可训练参数占比，直观感受 LoRA 有多省：通常 < 1%
    peft_model.print_trainable_parameters()
    return peft_model


# ----------------------------------------------------------------------------- #
# 4. 加载已微调好的 LoRA adapter（unlearning 阶段用）
# ----------------------------------------------------------------------------- #
def load_finetuned_model(
    model_name: str,
    adapter_dir: str,
    dtype: torch.dtype,
    device: str,
    trainable: bool,
):
    """
    基座模型 + 已保存的 LoRA adapter = 微调后的完整模型。

    LoRA 训练保存的目录里只有 adapter 的小矩阵（adapter_model.safetensors）
    和配置（adapter_config.json），不包含基座权重，
    所以加载时必须先加载同一个基座模型，再把 adapter 挂回去。

    参数
    ----
    trainable : True  → unlearning 要继续更新 LoRA 参数；
                False → 只做评估/推理，全部冻结。
    """
    base_model = load_base_model(model_name, dtype, device)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=trainable)
    model.to(device)
    print(f"[model] 已从 {adapter_dir} 挂载 LoRA adapter (trainable={trainable})")
    return model
