# -*- coding: utf-8 -*-
"""ReLearn 的超参数配置。共享字段见 baselines/common.py 的 BaseUnlearnConfig。"""

from dataclasses import dataclass

from ..common import BaseUnlearnConfig


@dataclass
class ReLearnConfig(BaseUnlearnConfig):
    output_dir: str = "outputs/baseline_relearn"

    # 每个 forget 问题生成几条不同的"无害替代答案"。
    # 论文强调多样性：同一问题配多个不同说法的安全回答，
    # 模型学到的是"这类问题就该这样模糊带过"的行为模式，
    # 而不是死记某一句固定话术（那样泛化差、也容易被绕过）。
    num_safe_variants: int = 3

    # retain 数据混入量 = len(增强后的 forget 数据) × retain_mix_ratio。
    # 训练时把两者混在一起 shuffle —— 一边覆盖旧记忆，一边复习保留知识，
    # 防止模型把"模糊带过"泛化到所有问题上（过度遗忘）。
    retain_mix_ratio: float = 1.0

    # ReLearn 就是标准 SFT（正向梯度下降），训练设置和普通微调一致
    num_epochs: int = 3
    learning_rate: float = 1e-4
