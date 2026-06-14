# -*- coding: utf-8 -*-
"""
clreg —— 对比表征塑形 Unlearning（与你的研究思路最接近的 baseline）

参考论文:
  [1] Tang & Khanna, "From Logits to Latents: Contrastive Representation
      Shaping for LLM Unlearning", arXiv:2601.22028 (CLReg)
      —— 在隐藏表征上加对比正则：识别 forget 特征并把它推离 retain
         特征/原位置，显式减少 forget-retain 纠缠，同时尽量不动
         retain 表征；可叠加在任意现有遗忘方法之上。
  [2] Doan et al., "CoUn: Empowering Machine Unlearning via Contrastive
      Learning", arXiv:2509.16391
      —— 利用样本间语义相似度，通过对比学习间接调整 forget 表征，
         使其与 retain 数据诱导的语义对齐。

本实现的具体化方式（与你的研究设定一致，便于直接做对照实验）:
  锚点  = 当前模型对 forget 样本的表征
  正样本 = 该样本"最近邻无害样本"的表征（在冻结模型表征空间中
          从 retain 池里按余弦相似度检索，训练前离线挖掘一次）
  负样本 = 该 batch 内各 forget 样本在冻结模型中的"原位置"表征
  → InfoNCE 把 forget 表征拉向无害近邻、推离原来的"记忆位置"
  另加 retain 表征锚定项 + retain 交叉熵，保持保留知识的泛化。

注意: 原论文全文获取受限，超参与正/负样本的具体取法是基于论文
公开描述的合理实例化，做严格对照实验前请核对原文细节。

文件:
  config.py     超参（层号、温度 τ、各损失权重、近邻池大小）
  method.py     表征提取、最近邻挖掘、InfoNCE 损失
  run_clreg.py  训练入口: python -m baselines.clreg.run_clreg
"""
