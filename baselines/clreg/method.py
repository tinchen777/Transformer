# -*- coding: utf-8 -*-
"""
method.py —— 对比表征塑形的核心组件

三个积木：
  1. 表征提取：一条 QA 样本 → 一个向量
     取第 ℓ 层隐藏状态，在"答案 token"位置上做平均池化。
     为什么只池化答案段？要遗忘/搬动的是"答案中的知识"的表征，
     问题本身（prompt）是中性的，不应该被搬动。

  2. 最近邻挖掘：为每个 forget 样本找"语义最近的无害样本"
     在冻结模型的表征空间里，对 retain 池逐条算余弦相似度取 top-1。
     训练前离线做一次即可（冻结模型的表征不会变）。
     —— 这一步就是你研究思路里"找最近邻无害概念"的实现位置，
        换成你自己的概念抽取/检索方法时只需要替换这一个函数。

  3. InfoNCE 对比损失：
         L = −log [ exp(sim(a,p)/τ) / (exp(sim(a,p)/τ) + Σ_j exp(sim(a,n_j)/τ)) ]
     a = 当前模型的 forget 表征（锚点，唯一带梯度的部分）
     p = 无害近邻的表征（正样本，冻结模型给出）
     n = forget 样本们在冻结模型里的"原位置"表征（负样本）
     最小化它 = 把 forget 表征拉向无害近邻、推离原来的记忆位置。
"""

import torch
import torch.nn.functional as F

from llm_pipeline.data_utils import IGNORE_INDEX, build_collator


# ----------------------------------------------------------------------------- #
# 1. 表征提取
# ----------------------------------------------------------------------------- #
def forward_with_reps(model, batch: dict, layer_id: int):
    """
    一次前向同时拿到：交叉熵 loss + 第 ℓ 层的答案段池化表征 [B, d]。

    （对比项和 GA/CE 项共用这一次前向，省一半计算。）
    """
    output = model(**batch, output_hidden_states=True)
    # hidden_states[0] 是词嵌入输出，第 ℓ 层 block 的输出在下标 ℓ+1
    hidden = output.hidden_states[layer_id + 1]          # [B, T, d]

    # 答案 token 掩码：labels != -100 的位置
    mask = (batch["labels"] != IGNORE_INDEX).unsqueeze(-1).float()  # [B, T, 1]
    # 平均池化：只对答案位置求和再除以答案长度
    reps = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, d]
    return output.loss, reps


@torch.no_grad()
def precompute_reps(model, tokenized_dataset, tokenizer, layer_id: int,
                    batch_size: int, device: str) -> torch.Tensor:
    """
    用冻结模型给整个数据集算一遍表征，返回 [N, d]（float32, 在 device 上）。

    训练前对 forget 集和 retain 池各做一次：
      - forget 的结果 = 负样本（"原位置"）+ 近邻检索的 query；
      - retain 池的结果 = 近邻检索的 key + 正样本来源。
    """
    model.eval()
    collator = build_collator(tokenizer)
    all_reps = []
    for start in range(0, len(tokenized_dataset), batch_size):
        rows = tokenized_dataset.select(range(start, min(start + batch_size, len(tokenized_dataset))))
        batch = collator([rows[i] for i in range(len(rows))])
        batch = {k: v.to(device) for k, v in batch.items()}
        _, reps = forward_with_reps(model, batch, layer_id)
        all_reps.append(reps.float())
    return torch.cat(all_reps, dim=0)        # [N, d]


# ----------------------------------------------------------------------------- #
# 2. 最近邻挖掘
# ----------------------------------------------------------------------------- #
def mine_nearest_neighbors(forget_reps: torch.Tensor,
                           retain_reps: torch.Tensor) -> torch.Tensor:
    """
    为每个 forget 样本在 retain 池中找余弦相似度最高的样本。

    forget_reps : [Nf, d]，retain_reps : [Nr, d]（都来自冻结模型）
    返回 [Nf] 的下标张量：第 i 个 forget 样本的无害近邻是 retain 池第几条。
    """
    f = F.normalize(forget_reps, dim=-1)
    r = F.normalize(retain_reps, dim=-1)
    sim = f @ r.T                             # [Nf, Nr] 余弦相似度矩阵
    return sim.argmax(dim=-1)                 # [Nf]


# ----------------------------------------------------------------------------- #
# 3. InfoNCE 对比损失
# ----------------------------------------------------------------------------- #
def info_nce(anchor: torch.Tensor, positive: torch.Tensor,
             negatives: torch.Tensor, tau: float) -> torch.Tensor:
    """
    anchor    : [B, d] 当前模型的 forget 表征（带梯度）
    positive  : [B, d] 每个样本对应的无害近邻表征（冻结，逐样本配对）
    negatives : [K, d] 共享负样本池（冻结；这里 K=B，即本 batch 的原位置表征）

    实现：把"和正样本的相似度"放在第 0 列，"和各负样本的相似度"放在后面，
    除以温度 τ 后做交叉熵、目标类别恒为 0 —— 这是 InfoNCE 的标准写法
    （等价于 −log(exp(s_pos/τ) / Σ exp(s/τ))）。
    """
    a = F.normalize(anchor.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    n = F.normalize(negatives.float(), dim=-1)

    sim_pos = (a * p).sum(dim=-1, keepdim=True)     # [B, 1] 逐样本配对相似度
    sim_neg = a @ n.T                               # [B, K] 对所有负样本的相似度

    logits = torch.cat([sim_pos, sim_neg], dim=1) / tau   # [B, 1+K]
    targets = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, targets)
