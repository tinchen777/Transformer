# -*- coding: utf-8 -*-
"""
method.py —— RMU 的核心组件

RMU 的总损失（论文式 (1)(2)，符号略作简化）：

  L = L_forget + α · L_retain

  L_forget = E_{x∈forget} ‖ h_θ^(ℓ)(x) − c·u ‖²
     把 forget 数据在第 ℓ 层的隐藏表征推向固定目标 c·u。
     u 是训练开始时随机采样的单位向量（之后不变），c 是放大系数。
     直觉：第 ℓ 层之后的所有层都是按"正常表征分布"训练出来的，
     一旦第 ℓ 层对 forget 输入输出一个"超出正常范围的乱码方向"，
     下游层就提取不出有效信息 —— 知识虽然可能还残留在浅层，
     但整条通路被掐断了。

  L_retain = E_{x∈retain} ‖ h_θ^(ℓ)(x) − h_frozen^(ℓ)(x) ‖²
     retain 数据的表征必须和冻结的原模型保持一致 —— 这是"利用性"
     的保障，与你研究里"微调保持保留概念泛化"的目标一致，
     只是 RMU 在表征层面做，而不是在输出概率层面做。

参数更新范围：只解冻少数几层 MLP 的输出投影矩阵
  （LLaMA 系叫 down_proj，GPT-2 叫 c_proj），其余全部冻结。
"""

import torch

from llm_pipeline.data_utils import IGNORE_INDEX


# ----------------------------------------------------------------------------- #
# 1. 控制向量：随机单位向量 u，放大 c 倍
# ----------------------------------------------------------------------------- #
def make_control_vector(hidden_size: int, steering_coeff: float,
                        device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    生成 RMU 的固定目标向量 c·u，形状 [hidden_size]。

    论文做法：每个分量从均匀分布采样，再归一化成单位向量。
    注意它只在训练开始时生成一次，整个训练过程保持不变 ——
    所有 forget 表征都被推向同一个"乱码方向"。
    """
    u = torch.rand(hidden_size, device=device, dtype=torch.float32)
    u = u / u.norm()
    return (steering_coeff * u).to(dtype)


# ----------------------------------------------------------------------------- #
# 2. 取第 ℓ 层的隐藏表征
# ----------------------------------------------------------------------------- #
def get_layer_hidden(model, batch: dict, layer_id: int) -> torch.Tensor:
    """
    前向一次，返回第 layer_id 层（0 开始计）的输出表征 [B, T, d]。

    transformers 的 output_hidden_states=True 会返回一个元组：
      hidden_states[0]   = 词嵌入层的输出
      hidden_states[k]   = 第 k-1 层 transformer block 的输出
    所以"第 layer_id 层的输出" = hidden_states[layer_id + 1]。
    """
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    return output.hidden_states[layer_id + 1]


# ----------------------------------------------------------------------------- #
# 3. 两项 MSE 损失（都只在答案 token 上计算）
# ----------------------------------------------------------------------------- #
def _answer_mask(batch: dict) -> torch.Tensor:
    """答案 token 的位置掩码 [B, T]（labels != -100 的位置）。"""
    return batch["labels"] != IGNORE_INDEX


def masked_mse(h: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    只在 mask=True 的 token 位置上算 MSE。

    h      : [B, T, d] 当前模型表征
    target : [B, T, d] 或可广播的 [d]（控制向量）
    mask   : [B, T]
    """
    diff2 = (h.float() - target.float()) ** 2          # [B, T, d]
    per_token = diff2.mean(dim=-1)                     # [B, T]，每个 token 的均方差
    return (per_token * mask).sum() / mask.sum().clamp(min=1)


def rmu_forget_loss(model, batch: dict, layer_id: int,
                    control_vec: torch.Tensor) -> torch.Tensor:
    """forget 项：把 forget 数据的第 ℓ 层表征推向 c·u。"""
    h = get_layer_hidden(model, batch, layer_id)
    return masked_mse(h, control_vec, _answer_mask(batch))


def rmu_retain_loss(model, frozen_model, batch: dict, layer_id: int) -> torch.Tensor:
    """retain 项：retain 数据的表征与冻结原模型保持一致。"""
    h = get_layer_hidden(model, batch, layer_id)
    with torch.no_grad():
        h_frozen = get_layer_hidden(frozen_model, batch, layer_id)
    return masked_mse(h, h_frozen, _answer_mask(batch))


# ----------------------------------------------------------------------------- #
# 4. 参数冻结：只解冻指定层 MLP 的输出投影
# ----------------------------------------------------------------------------- #
def get_mlp_out_proj(model, layer_idx: int):
    """
    按架构取第 layer_idx 层 MLP 的输出投影模块。

    不同架构命名不同：
      GPT-2 : model.transformer.h[i].mlp.c_proj
      LLaMA/Qwen/Mistral : model.model.layers[i].mlp.down_proj
    """
    model_type = model.config.model_type
    if model_type == "gpt2":
        return model.transformer.h[layer_idx].mlp.c_proj
    if model_type in ("llama", "qwen2", "mistral"):
        return model.model.layers[layer_idx].mlp.down_proj
    raise ValueError(f"不认识的架构 {model_type}，请在 get_mlp_out_proj 里补充映射")


def freeze_except_mlp_layers(model, update_layer_ids: list[int]) -> list[torch.nn.Parameter]:
    """
    冻结全模型，只解冻 update_layer_ids 指定层的 MLP 输出投影。
    返回可训练参数列表（交给优化器）。
    """
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = []
    for idx in update_layer_ids:
        proj = get_mlp_out_proj(model, idx)
        for p in proj.parameters():
            p.requires_grad_(True)
            trainable.append(p)

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[rmu] 可训练参数: {n_train / 1e6:.2f}M / {n_total / 1e6:.1f}M "
          f"({100 * n_train / n_total:.2f}%)，层 = {update_layer_ids}")
    return trainable
