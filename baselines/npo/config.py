# -*- coding: utf-8 -*-
"""NPO 的超参数配置。共享字段见 baselines/common.py 的 BaseUnlearnConfig。"""

from dataclasses import dataclass

from ..common import BaseUnlearnConfig


@dataclass
class NPOConfig(BaseUnlearnConfig):
    output_dir: str = "outputs/baseline_npo"

    # β：NPO 的逆温度（论文中的 beta，默认 0.1，与 DPO 习惯一致）。
    #   β 越小 → 损失越平缓、遗忘越温和；
    #   β → 0 的极限下 NPO 退化成梯度上升（论文 Proposition 1）。
    beta: float = 0.1

    # retain 项模式（对应论文的三个变体）：
    #   "none" → 纯 NPO，只有 forget 损失；
    #   "rt"   → NPO-RT：加 retain 集上的标准交叉熵（论文式 (5)）；
    #   "kl"   → NPO-KL：加 retain 集上与参考模型的 KL 散度（论文式 (4)）。
    retain_mode: str = "rt"
    retain_weight: float = 1.0   # retain 项的权重 λ
