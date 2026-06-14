# -*- coding: utf-8 -*-
"""
npo —— Negative Preference Optimization 复现

论文: Zhang et al., "Negative Preference Optimization: From Catastrophic
      Collapse to Effective Unlearning", COLM 2024. arXiv:2404.05868

一句话: 把 DPO 偏好损失里的"正样本项"扔掉、只保留"负样本项"，
        得到一个只压低 forget 数据概率、但比梯度上升稳定得多的遗忘损失。
        是 TOFU 等遗忘基准上最常用的强 baseline。

文件:
  config.py   NPO 特有超参（β、retain 项模式等）
  loss.py     NPO 损失的实现 + 推导注释
  run_npo.py  训练入口: python -m baselines.npo.run_npo
"""
