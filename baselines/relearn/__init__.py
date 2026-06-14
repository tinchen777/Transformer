# -*- coding: utf-8 -*-
"""
relearn —— ReLearn: Unlearning via Learning 复现

论文: Xu et al., "ReLearn: Unlearning via Learning for Large Language
      Models", ACL 2025. arXiv:2502.11190
      官方代码: https://github.com/zjunlp/unlearn

一句话: 不做任何"反向优化"（梯度上升、负偏好都不用），而是构造
        "无害替代答案"并用标准 SFT 正向训练 —— 用新的安全记忆
        覆盖旧的敏感记忆。论文的洞察：反向优化会破坏语言模型的
        生成连贯性，而"用学习实现遗忘"能保住输出的流畅度。

        这和你的研究思路在数据层面同构：把要遗忘的概念"替换/拉向"
        无害概念，再用 retain 数据保持泛化 —— 只是 ReLearn 在
        数据空间做替换，你的方法在表征空间做拉近。

文件:
  config.py       ReLearn 特有超参（替代答案数量、retain 混合比例等）
  augment.py      无害替代答案的构造（模板版 + 可接入 LLM 的接口）
  run_relearn.py  训练入口: python -m baselines.relearn.run_relearn
"""
