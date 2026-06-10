# -*- coding: utf-8 -*-
"""
llm_pipeline —— 基于 HuggingFace 生态的 LLM 微调 + Unlearning 流程包

这个包和仓库里手写的 `transformer/` 包完全独立：
  - `transformer/` 是你自己从零实现、用来理解内部构造的；
  - `llm_pipeline/` 是「实际科研工作流」：直接调用 HuggingFace 的
    transformers / datasets / peft 等库，下载现成模型和数据来做实验。

模块一览（按依赖关系从底向上）：
  config.py        所有超参数集中放在 dataclass 里，改实验只改这一个文件
  data_utils.py    下载 TOFU 数据集、分词（tokenize）、构造 DataLoader
  model_utils.py   下载/加载模型与分词器、注入 LoRA、加载已微调的 adapter
  evaluate.py      评估工具：困惑度（perplexity）+ 生成样例对比
  finetune_lora.py 入口 1：用 LoRA 微调一个 LLM（含训练后的测试）
  unlearn_ga.py    入口 2：梯度上升（Gradient Ascent）Unlearning，
                   直接复用入口 1 产出的 LoRA adapter 和上面所有公共模块

典型使用顺序：
  1. python -m llm_pipeline.finetune_lora   # 先把模型在 TOFU 全集上微调
  2. python -m llm_pipeline.unlearn_ga      # 再对 forget 子集做遗忘
"""
