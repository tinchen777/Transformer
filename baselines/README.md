# baselines —— 近期 LLM Unlearning 论文复现（4 个方法，各自独立成包）

这些方法和你的研究思路（**对比学习把遗忘概念拉向最近邻无害概念 + 微调保持
保留概念泛化**）在同一条研究线上，可直接作为对照实验的 baseline。
所有方法共用 `llm_pipeline/` 的微调产物、数据流程和评估口径（控制变量，只比方法）。

## 方法总览

| 子包 | 论文 | 遗忘发生在哪个空间 | 核心机制 | 与你的方法的关系 |
|---|---|---|---|---|
| `npo/` | [NPO, COLM 2024](https://arxiv.org/abs/2404.05868) | 输出概率空间 | 只保留 DPO 的负样本项：压低 forget 概率，且损失自带"刹车"（忘到位后梯度自动衰减） | 标准强 baseline，必备对照 |
| `rmu/` | [RMU / WMDP, ICML 2024](https://arxiv.org/abs/2403.03218) | 中间层表征空间 | 把 forget 表征推向**固定随机方向** c·u + retain 表征锚定 | 同为表征空间方法；目标方向是随机向量，你的方法换成"最近邻无害概念" |
| `relearn/` | [ReLearn, ACL 2025](https://arxiv.org/abs/2502.11190) | 数据空间 | 把 forget 答案替换成**无害替代答案**后正向 SFT（覆盖学习，无反向优化） | "拉向无害概念"的数据空间版本 |
| `clreg/` | [CLReg, 2026](https://arxiv.org/abs/2601.22028) + [CoUn, 2025](https://arxiv.org/abs/2509.16391) | 中间层表征空间 | InfoNCE：forget 表征**拉向最近邻无害样本**（正）、推离原位置（负）+ retain 锚定 | **与你的思路最接近**，可作为最直接的对照/出发点 |

另外 `llm_pipeline/unlearn_ga.py` 里已有 GA 和 GradDiff（GA + retain 项），
它们是所有论文都会比的两个最基础 baseline。

## 运行

```bash
# 前提：先完成微调，得到 outputs/finetune_lora/final
python -m llm_pipeline.finetune_lora

# 各 baseline 互相独立，任选运行（在仓库根目录）
python -m baselines.npo.run_npo
python -m baselines.rmu.run_rmu
python -m baselines.relearn.run_relearn
python -m baselines.clreg.run_clreg
```

每个入口都会输出统一格式的结果：遗忘前/后的 forget PPL（希望↑）与
retain PPL（希望≈不变）对比表 + 生成样例，方便直接横向比较。

## 包结构（每个方法一致）

```
baselines/
├── common.py        # 共享：配置基类、forget/retain 数据构造、
│                    #       答案 log-prob、KL、统一评估与汇总打印
└── <method>/
    ├── __init__.py  # 论文信息 + 方法一句话总结
    ├── config.py    # 该方法特有的超参（继承 BaseUnlearnConfig）
    ├── loss.py 或 method.py  # 核心算法实现（带公式推导注释）
    └── run_<method>.py       # 训练入口（流程编号注释 ①②③...）
```

## 阅读建议（按理解成本从低到高）

1. **relearn**：没有任何反向优化，就是"换答案再 SFT"，最好懂；
   重点看 `augment.py` 里替代答案的设计原则。
2. **npo**：重点看 `loss.py` 顶部的推导 —— 为什么 GA 会崩、
   NPO 的损失曲面为什么自带刹车。
3. **rmu**：重点看 `method.py` 顶部 —— "掐断下游通路"的直觉，
   以及为什么只更新少数几层的 MLP。
4. **clreg**：组件最多（表征提取、近邻挖掘、InfoNCE、双重 retain 保持），
   但每个组件都直接对应你研究方案里的一个部件；
   把 `mine_nearest_neighbors` 换成你自己的"无害概念检索"，
   就得到了你的方法的雏形。

## 注意事项

- `rmu/` 会把 LoRA 合并进基座后更新基座权重，保存的是**完整模型**；
  其余三个方法只更新并保存 LoRA adapter（几 MB）。
- `clreg/` 的正/负样本取法是基于论文公开描述的合理实例化
  （原文全文获取受限），写论文做严格对照前请核对原文。
- 默认模型是 gpt2 + TOFU forget10/retain90；换模型时注意按比例调整
  `rmu` 的 `layer_id` / `update_layer_ids` 和 `clreg` 的 `layer_id`。
- RMU 的 `steering_coeff` 与激活模长同量级才有效，换模型后建议先打印
  该层激活的范数再定 c（论文对 7B 模型用 20）。
