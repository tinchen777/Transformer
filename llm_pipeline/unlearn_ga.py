# -*- coding: utf-8 -*-
"""
unlearn_ga.py —— 入口 2：梯度上升（Gradient Ascent）LLM Unlearning

前置条件：先跑过入口 1（finetune_lora.py），得到一个「记住了 TOFU 知识」
的 LoRA 模型。本脚本加载它，然后让它忘掉 forget 子集。

为什么梯度上升能实现遗忘？
  普通训练是梯度下降：朝着「降低 loss」的方向更新参数，
  loss 越低 = 模型给正确答案分配的概率越高 = 记得越牢。
  Unlearning 反过来：在 forget 数据上朝「升高 loss」的方向更新，
  即压低模型给这些答案的概率 —— 这就是最朴素的遗忘方法（GA）。
  实现上只需一行：loss = -loss_forget，再正常 backward + step。

纯 GA 的问题与改进：
  反向的梯度不长眼睛，会把模型的通用能力一起破坏（灾难性遗忘）。
  常用缓解：再加一项 retain 集上的正常梯度下降，
      总损失 = -loss_forget + λ · loss_retain
  一边「反着学」forget 数据，一边「正着学」retain 数据，
  这就是文献里的 Gradient Difference（GD/GradDiff）基线。
  本脚本用 cfg.retain_weight 控制 λ，设 0 就退化成纯 GA。

评估标准（unlearning 论文的两条基本轴）：
  - 遗忘效果（forget quality）：forget 集 PPL 应明显上升；
  - 模型可用性（model utility）：retain 集 PPL 应基本不变。

这里特意不用 Trainer 而是手写训练循环：
  一是 Trainer 默认只会「下降」给定的 loss，自定义遗忘损失反而绕；
  二是手写循环能让你看清每一步（forward → 组合 loss → backward →
  clip → step）到底发生了什么，对理解 unlearning 算法本身更有帮助。

运行：
  cd <仓库根目录>
  python -m llm_pipeline.unlearn_ga
"""

import itertools
import os

import torch
from transformers import set_seed

from .config import DEVICE, DTYPE, UnlearnConfig
from .data_utils import build_dataloader, build_tokenized_dataset
from .evaluate import compute_perplexity, show_generations
from .model_utils import load_finetuned_model, load_tokenizer


def batch_to_device(batch: dict, device: str) -> dict:
    """把 collator 产出的一个 batch（dict of tensor）整体搬到 GPU/CPU。"""
    return {k: v.to(device) for k, v in batch.items()}


# ----------------------------------------------------------------------------- #
# 评估：把「遗忘效果 + 模型可用性」打包成一次快照，方便前后对比
# ----------------------------------------------------------------------------- #
def evaluate_snapshot(model, forget_loader, retain_loader, tag: str) -> dict:
    """同时在 forget / retain 两个评估集上算 PPL，并打印成一行摘要。"""
    forget_metrics = compute_perplexity(model, forget_loader, DEVICE)
    retain_metrics = compute_perplexity(model, retain_loader, DEVICE)
    print(f"\n[eval/{tag}] forget: loss={forget_metrics['loss']:.4f} "
          f"ppl={forget_metrics['perplexity']:.2f} | "
          f"retain: loss={retain_metrics['loss']:.4f} "
          f"ppl={retain_metrics['perplexity']:.2f}")
    return {"forget": forget_metrics, "retain": retain_metrics}


# ----------------------------------------------------------------------------- #
# 核心：梯度上升 unlearning 训练循环
# ----------------------------------------------------------------------------- #
def unlearn(model, forget_loader, retain_loader, cfg: UnlearnConfig) -> None:
    """
    在 forget 集上做梯度上升（可选叠加 retain 集梯度下降）。

    每一步：
      1. 取一个 forget batch，forward 得到 loss_forget；
      2. （可选）取一个 retain batch，forward 得到 loss_retain；
      3. 组合总损失 total = -loss_forget + λ·loss_retain；
         对 total 做梯度「下降」，等价于对 loss_forget 做梯度「上升」；
      4. backward → 梯度裁剪 → optimizer.step()。
    """
    # 只优化 requires_grad=True 的参数 —— 即 LoRA 的 A、B 小矩阵。
    # 基座模型在加载时已被 peft 冻结，遗忘只发生在 adapter 里。
    # （这也意味着「删掉 adapter 就能完全恢复」，做实验非常方便。）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

    # retain 集通常比 forget 集大得多，一个 epoch 用不完；
    # 用 itertools.cycle 把它变成取不完的循环迭代器，按需一批批拿。
    retain_iter = itertools.cycle(retain_loader) if cfg.retain_weight > 0 else None

    model.train()  # 训练模式（启用 LoRA 的 dropout）
    step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        for forget_batch in forget_loader:
            step += 1

            # ---- 1. forget 集 forward：得到我们想「升高」的 loss ----
            forget_batch = batch_to_device(forget_batch, DEVICE)
            loss_forget = model(**forget_batch).loss

            # ---- 2. 组合总损失 ----
            # 负号是整个算法的灵魂：optimizer 永远在最小化 total_loss，
            # 最小化 -loss_forget 就是最大化 loss_forget（梯度上升）。
            total_loss = -loss_forget

            loss_retain_val = None
            if retain_iter is not None:
                # 叠加 retain 项：在保留数据上正常梯度下降，护住通用能力
                retain_batch = batch_to_device(next(retain_iter), DEVICE)
                loss_retain = model(**retain_batch).loss
                total_loss = total_loss + cfg.retain_weight * loss_retain
                loss_retain_val = loss_retain.item()

            # ---- 3. 反向传播 + 更新 ----
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪：梯度上升的更新方向「没有底」，裁剪能防止单步爆炸
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()

            # ---- 4. 日志 ----
            if step % cfg.logging_steps == 0:
                retain_str = (f" | retain loss {loss_retain_val:.4f}"
                              if loss_retain_val is not None else "")
                print(f"epoch {epoch} | step {step:>4d} | "
                      f"forget loss {loss_forget.item():.4f} (越大越'忘')"
                      f"{retain_str}")

            # ---- 5. 提前停止 ----
            # forget loss 升到阈值说明已经忘得差不多；继续升只会把模型
            # 推向输出乱码，retain 能力也会被拖垮。
            if loss_forget.item() > cfg.forget_loss_threshold:
                print(f"\n[unlearn] forget loss {loss_forget.item():.2f} 已超过阈值 "
                      f"{cfg.forget_loss_threshold}，提前停止。")
                return


def main():
    cfg = UnlearnConfig()
    set_seed(cfg.seed)

    # ======================================================================= #
    # ① 加载「微调后」的模型：基座 + 入口 1 保存的 LoRA adapter
    # ======================================================================= #
    if not os.path.isdir(cfg.finetuned_adapter_dir):
        raise FileNotFoundError(
            f"找不到微调产物 {cfg.finetuned_adapter_dir}，"
            f"请先运行: python -m llm_pipeline.finetune_lora"
        )
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_finetuned_model(
        cfg.model_name, cfg.finetuned_adapter_dir, DTYPE, DEVICE,
        trainable=True,  # unlearning 要继续更新 LoRA 参数
    )

    # ======================================================================= #
    # ② 准备数据：forget 集（要忘的）和 retain 集（要保的）
    # ======================================================================= #
    forget_tokenized, forget_raw = build_tokenized_dataset(
        cfg.dataset_name, cfg.forget_config, tokenizer, cfg.max_len)
    retain_tokenized, _ = build_tokenized_dataset(
        cfg.dataset_name, cfg.retain_config, tokenizer, cfg.max_len)

    # 训练用 loader（forget 集要 shuffle；retain 集做正则项也 shuffle）
    forget_train_loader = build_dataloader(
        forget_tokenized, tokenizer, cfg.batch_size, shuffle=True)
    retain_train_loader = build_dataloader(
        retain_tokenized, tokenizer, cfg.batch_size, shuffle=True)

    # 评估用 loader：各抽固定的前 N 条（不 shuffle，保证前后评估同一批数据）
    n = cfg.num_eval_samples
    forget_eval_loader = build_dataloader(
        forget_tokenized.select(range(min(n, len(forget_tokenized)))),
        tokenizer, cfg.batch_size, shuffle=False)
    retain_eval_loader = build_dataloader(
        retain_tokenized.select(range(min(n, len(retain_tokenized)))),
        tokenizer, cfg.batch_size, shuffle=False)

    # 从 forget 集里抽几条 QA，遗忘前后各生成一次做定性对比
    demo_qa = [forget_raw[i] for i in cfg.demo_question_indices]

    # ======================================================================= #
    # ③ 遗忘前的基线评估
    # ======================================================================= #
    before = evaluate_snapshot(model, forget_eval_loader, retain_eval_loader,
                               tag="遗忘前")
    show_generations(model, tokenizer, demo_qa, DEVICE, cfg.max_new_tokens,
                     title="遗忘前（微调模型还记得 forget 集的答案）")

    # ======================================================================= #
    # ④ 执行梯度上升 unlearning
    # ======================================================================= #
    mode = ("纯梯度上升 (GA)" if cfg.retain_weight == 0
            else f"梯度上升 + retain 保持项 (λ={cfg.retain_weight})")
    print(f"\n[unlearn] 开始 unlearning，方法 = {mode}")
    unlearn(model, forget_train_loader, retain_train_loader, cfg)

    # ======================================================================= #
    # ⑤ 遗忘后的评估，与基线对比
    # ======================================================================= #
    after = evaluate_snapshot(model, forget_eval_loader, retain_eval_loader,
                              tag="遗忘后")
    show_generations(model, tokenizer, demo_qa, DEVICE, cfg.max_new_tokens,
                     title="遗忘后（理想情况：forget 的答案答不出，其他能力保留）")

    # 汇总成一张小表：看 forget PPL 升了多少、retain PPL 是否守住
    print(f"\n{'=' * 58}")
    print(f"{'指标':<24}{'遗忘前':>12}{'遗忘后':>12}")
    print("-" * 58)
    print(f"{'forget PPL (希望↑)':<24}"
          f"{before['forget']['perplexity']:>14.2f}"
          f"{after['forget']['perplexity']:>14.2f}")
    print(f"{'retain PPL (希望≈不变)':<24}"
          f"{before['retain']['perplexity']:>14.2f}"
          f"{after['retain']['perplexity']:>14.2f}")
    print("=" * 58)

    # ======================================================================= #
    # ⑥ 保存遗忘后的 LoRA adapter
    # ======================================================================= #
    final_dir = f"{cfg.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[save] 遗忘后的 LoRA adapter 已保存到 {final_dir}")


if __name__ == "__main__":
    main()
