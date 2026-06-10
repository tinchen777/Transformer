# -*- coding: utf-8 -*-
"""
rmu —— Representation Misdirection for Unlearning 复现

论文: Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious Use
      With Unlearning", ICML 2024. arXiv:2403.03218

一句话: 不在输出概率上做文章，而是直接操控中间层的隐藏表征 ——
        把 forget 数据的表征"推向"一个固定随机方向（让下游层读不懂），
        同时把 retain 数据的表征锚定在原处。
        与你的研究同属"表征空间遗忘"，区别在于 RMU 的目标方向是
        随机向量，而你的方法是把表征拉向最近邻的无害概念。

文件:
  config.py   RMU 特有超参（层号、引导系数 c、retain 权重 α 等）
  method.py   控制向量构造、隐藏表征提取、两项 MSE 损失
  run_rmu.py  训练入口: python -m baselines.rmu.run_rmu
"""
