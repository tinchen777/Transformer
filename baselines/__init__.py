# -*- coding: utf-8 -*-
"""
baselines —— 近期优秀 LLM Unlearning 论文的复现实现（作为你研究的 baseline）

每个子包对应一篇论文 / 一个方法，互相独立，但共享：
  - llm_pipeline/ 里的数据、模型、评估工具（微调产物也直接复用）；
  - baselines/common.py 里的公共训练辅助函数。

子包一览（按「与你的研究思路的接近程度」从远到近排序）：
  npo/      Negative Preference Optimization (arXiv 2404.05868)
            —— 偏好优化式遗忘，TOFU 上最常用的强 baseline，必备对照
  rmu/      Representation Misdirection Unlearning (WMDP, arXiv 2403.03218)
            —— 把 forget 表征"推向"一个固定随机向量 + retain 表征锚定
  relearn/  ReLearn: Unlearning via Learning (arXiv 2502.11190, ACL 2025)
            —— 数据层面：把 forget 答案换成无害替代答案后正常 SFT
  clreg/    对比表征塑形 (CLReg, arXiv 2601.22028; CoUn, arXiv 2509.16391)
            —— InfoNCE 把 forget 表征拉向"最近邻无害样本"、推离原位置，
               与你的研究思路（对比学习拉近无害概念）最接近

运行前提：先跑 `python -m llm_pipeline.finetune_lora` 得到微调好的模型。
运行方式（在仓库根目录）：
  python -m baselines.npo.run_npo
  python -m baselines.rmu.run_rmu
  python -m baselines.relearn.run_relearn
  python -m baselines.clreg.run_clreg
"""
