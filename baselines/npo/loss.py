# -*- coding: utf-8 -*-
"""
loss.py —— NPO 损失函数

背景：为什么需要 NPO？
  梯度上升（GA）的损失是 -log π_θ(y|x)，对它上升等价于最小化 log π_θ。
  问题在于 log π_θ 没有下界（概率可以无限趋近 0，log 趋近 -∞），
  所以 GA 的"油门"永远踩到底，训练后期梯度反而越来越大，
  很快把整个模型拖垮（论文称为 catastrophic collapse）。

NPO 的做法：
  从 DPO 的偏好损失出发，删掉"偏好的正样本"那一项，只留下
  "把 forget 数据当负样本"的一项，得到（论文式 (3)）：

      L_NPO = (2/β) · E_{(x,y)∈forget} [ log( 1 + (π_θ(y|x) / π_ref(y|x))^β ) ]

  其中 π_ref 是遗忘开始前的模型（这里 = 微调后的模型），全程冻结。

为什么它比 GA 稳定？关键看损失曲面：
  - 当 π_θ ≈ π_ref（还没忘）：损失大，梯度也大，使劲忘；
  - 当 π_θ << π_ref（已经忘了）：log(1+x) ≈ x → 0，
    损失和梯度都自动衰减到 0 —— 相当于自带"刹车"，
    忘到位之后就不再继续破坏模型。
  论文证明 NPO 走向崩坏的速度比 GA 指数级慢（Theorem 1）。

数值稳定的实现形式：
  令 r = log π_θ(y|x) − log π_ref(y|x)（答案 token 对数概率之和的差），
      log(1 + e^{βr}) = −log σ(−βr)
  所以  L_NPO = −(2/β) · E[ log σ(−β·r) ]
  用 PyTorch 的 F.logsigmoid 实现，避免 exp 溢出。
"""

import torch
import torch.nn.functional as F

from ..common import answer_log_prob


def npo_loss(model, ref_model, forget_batch: dict, beta: float) -> tuple[torch.Tensor, dict]:
    """
    计算一个 forget batch 上的 NPO 损失。

    返回 (loss, 日志字典)。日志里带上当前/参考模型的对数概率差，
    方便观察"遗忘进度"（差值越负说明 forget 概率被压得越低）。
    """
    # 当前模型对答案的 log π_θ(y|x)（带梯度）
    logp_theta = answer_log_prob(model, forget_batch)            # [B]

    # 参考模型的 log π_ref(y|x)（冻结，不需要梯度）
    with torch.no_grad():
        logp_ref = answer_log_prob(ref_model, forget_batch)      # [B]

    # r = log(π_θ / π_ref)，逐样本
    log_ratio = logp_theta - logp_ref                            # [B]

    # L_NPO = -(2/β)·E[ logσ(-β·r) ]
    loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio).mean()

    logs = {
        "npo_loss": loss.item(),
        "log_ratio": log_ratio.mean().item(),   # 越负 = forget 概率被压得越低
    }
    return loss, logs
