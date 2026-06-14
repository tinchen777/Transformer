# -*- coding: utf-8 -*-
"""对比表征塑形的超参数配置。共享字段见 baselines/common.py 的 BaseUnlearnConfig。"""

from dataclasses import dataclass

from ..common import BaseUnlearnConfig


@dataclass
class CLRegConfig(BaseUnlearnConfig):
    output_dir: str = "outputs/baseline_clreg"

    # ---- 表征提取 ----
    # 在第几层（0 开始计）的输出上取表征。中层通常携带最丰富的语义信息
    # （太浅偏词法，太深偏输出 token 预测）。GPT-2 共 12 层，取第 6 层。
    layer_id: int = 6

    # ---- 对比损失 ----
    tau: float = 0.1            # InfoNCE 温度：越小对"最相似的负样本"惩罚越尖锐
    neighbor_pool_size: int = 1000  # 从 retain 集前多少条里检索"最近邻无害样本"

    # ---- 各损失项权重 ----
    # 总损失 = w_contrast·InfoNCE                  （表征：拉向无害近邻、推离原位置）
    #        + w_forget_ga·(−CE_forget)            （输出：压低原答案概率；论文中
    #                                               CLReg 是叠加在基础遗忘方法上的
    #                                               正则项，这里基础方法选 GA，
    #                                               设 0 可观察纯表征塑形的效果）
    #        + w_retain_anchor·MSE(h_retain, h_frozen)（表征：retain 最小位移）
    #        + w_retain_ce·CE_retain               （输出：retain 保持泛化）
    w_contrast: float = 1.0
    w_forget_ga: float = 1.0
    w_retain_anchor: float = 1.0
    w_retain_ce: float = 1.0

    # forget 交叉熵升到该值即提前停止（与 llm_pipeline/unlearn_ga 同理：
    # 反向优化没有自然终点，需要人为的"忘够了"判据）
    forget_loss_threshold: float = 8.0

    num_epochs: int = 3
    learning_rate: float = 1e-4
