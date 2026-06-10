# -*- coding: utf-8 -*-
"""RMU 的超参数配置。共享字段见 baselines/common.py 的 BaseUnlearnConfig。"""

from dataclasses import dataclass, field

from ..common import BaseUnlearnConfig


@dataclass
class RMUConfig(BaseUnlearnConfig):
    output_dir: str = "outputs/baseline_rmu"

    # ---- 表征操控的位置 ----
    # layer_id: 在第几层（0 开始计）的输出上施加损失。
    #   论文在 Zephyr-7B（32 层）上用第 7 层；按"前 1/4 深度"的比例，
    #   GPT-2（12 层）取第 3 层左右。换模型时按比例调整。
    layer_id: int = 3
    # update_layer_ids: 实际允许更新参数的层（只更新这些层 MLP 的输出投影）。
    #   论文做法：更新 layer_id 及其前两层，其余参数全部冻结 ——
    #   改动局部化，对模型其他能力的伤害更小。
    update_layer_ids: list[int] = field(default_factory=lambda: [1, 2, 3])

    # ---- 损失相关 ----
    # steering_coeff: 控制向量的模长 c。forget 表征会被推向 c·u（u 是随机
    #   单位向量）。c 要明显大于该层正常激活的模长，才能把表征"推出"
    #   正常语义区域（论文对 Zephyr 用 20，量级随模型/层而变，需调参）。
    steering_coeff: float = 20.0
    # alpha: retain 表征锚定项的权重 α。论文取 100~1200 量级 ——
    #   表征 MSE 数值很小，要乘大权重才能和 forget 项抗衡。
    alpha: float = 100.0

    # ---- 训练 ----
    # 论文只训练很少的步数（约 80 步）就足以"打乱"forget 表征；
    # 训练太久反而会伤 retain。这里默认 150 步（小模型可以多走几步）。
    max_steps: int = 150
    learning_rate: float = 5e-4   # 只更新 3 个小矩阵，可以用偏大的学习率
