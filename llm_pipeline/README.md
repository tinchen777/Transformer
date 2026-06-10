# llm_pipeline —— LoRA 微调 + 梯度上升 Unlearning 流程

这个包是基于 HuggingFace 生态（`transformers` / `datasets` / `peft`）的
**实际科研工作流**，和仓库里手写的 `transformer/` 包（用于理解内部构造）完全独立、互不依赖。

## 整体流程

```
HuggingFace Hub
  ├── 模型 (默认 gpt2，可换 Llama/Qwen)
  └── 数据 (locuslab/TOFU)
        │
        ▼
[入口 1] finetune_lora.py
  基座模型 + LoRA → 在 TOFU full 上微调 → 模型"记住"虚构作家知识
  → 测试 (验证集 PPL + 生成样例) → 保存 adapter 到 outputs/finetune_lora/final
        │
        ▼
[入口 2] unlearn_ga.py
  加载微调后的模型 → 在 forget10 上梯度上升 (可叠加 retain90 保持项)
  → 评估遗忘效果 (forget PPL↑) 与可用性 (retain PPL≈) → 保存到 outputs/unlearn_ga/final
```

## 文件结构

| 文件 | 作用 |
|---|---|
| `config.py` | 所有超参数（dataclass），改实验只改这里 |
| `data_utils.py` | 下载 TOFU、分词（prompt 段 label = -100）、动态 padding |
| `model_utils.py` | 下载模型/分词器、注入 LoRA、加载已保存的 adapter |
| `evaluate.py` | 困惑度计算 + 生成样例对比 |
| `finetune_lora.py` | **入口 1**：LoRA 微调（含训练后测试） |
| `unlearn_ga.py` | **入口 2**：梯度上升 unlearning（复用以上全部模块） |

## 安装与运行

```bash
pip install -r requirements.txt   # 已包含 peft / accelerate

# 第 1 步：LoRA 微调（第一次运行会自动从 HF 下载模型和数据）
python -m llm_pipeline.finetune_lora

# 第 2 步：梯度上升 unlearning（依赖第 1 步保存的 adapter）
python -m llm_pipeline.unlearn_ga
```

调试时建议先把 `config.py` 里的 `num_train_samples` 设成 200 左右快速跑通，
再改回 `None` 跑全量。

## 关键概念速查

- **LoRA**：冻结基座权重 W，旁挂低秩矩阵 A、B，只训练它们
  （前向变为 `W·x + (α/r)·B·A·x`）。保存的 adapter 只有几 MB。
- **label mask**：把 prompt 段的 labels 设为 -100，loss 只在答案段计算。
- **梯度上升 (GA)**：`loss = -loss_forget`，最小化它等于最大化 forget 数据的
  loss，即压低模型对这些答案的概率 → 遗忘。
- **GradDiff**：`loss = -loss_forget + λ·loss_retain`，叠加 retain 集的正常
  梯度下降，缓解纯 GA 的灾难性遗忘。由 `UnlearnConfig.retain_weight` 控制。
- **评估两条轴**：forget 集 PPL 应上升（遗忘效果）、retain 集 PPL 应基本不变
  （模型可用性）。
