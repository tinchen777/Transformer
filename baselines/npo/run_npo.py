# -*- coding: utf-8 -*-
"""
run_npo.py —— NPO Unlearning 训练入口

流程：
  ① 加载两份"微调后的模型"：
       - model     可训练（继续更新 LoRA 参数）
       - ref_model 冻结，作为 NPO 损失里的参考分布 π_ref
  ② 每步：forget batch 上算 NPO 损失（+ 可选 retain 项）→ 更新
  ③ 遗忘前后评估 forget/retain PPL + 生成样例对比

运行：python -m baselines.npo.run_npo
"""

import itertools

import torch
from transformers import set_seed

from llm_pipeline.config import DEVICE, DTYPE
from llm_pipeline.evaluate import show_generations
from llm_pipeline.model_utils import load_finetuned_model, load_tokenizer

from ..common import (batch_to_device, build_unlearn_data, evaluate_snapshot,
                      print_summary, token_kl_to_ref)
from .config import NPOConfig
from .loss import npo_loss


def main():
    cfg = NPOConfig()
    set_seed(cfg.seed)

    # ====================================================================== #
    # ① 模型：可训练副本 + 冻结的参考副本
    #    NPO 必须有参考模型 π_ref —— 它定义了"遗忘前"的概率基准，
    #    损失只关心 π_θ 相对 π_ref 降了多少，这正是 NPO 自带刹车的来源。
    # ====================================================================== #
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_finetuned_model(cfg.model_name, cfg.finetuned_adapter_dir,
                                 DTYPE, DEVICE, trainable=True)
    ref_model = load_finetuned_model(cfg.model_name, cfg.finetuned_adapter_dir,
                                     DTYPE, DEVICE, trainable=False)
    ref_model.eval()

    # ====================================================================== #
    # ② 数据 + 遗忘前评估
    # ====================================================================== #
    data = build_unlearn_data(cfg, tokenizer)
    before = evaluate_snapshot(model, data, tag="遗忘前")
    show_generations(model, tokenizer, data.demo_qa, DEVICE, cfg.max_new_tokens,
                     title="NPO 遗忘前")

    # ====================================================================== #
    # ③ 训练循环
    # ====================================================================== #
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    retain_iter = (itertools.cycle(data.retain_train_loader)
                   if cfg.retain_mode != "none" else None)

    model.train()
    step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        for forget_batch in data.forget_train_loader:
            step += 1
            forget_batch = batch_to_device(forget_batch)

            # ---- NPO 遗忘项 ----
            loss, logs = npo_loss(model, ref_model, forget_batch, cfg.beta)

            # ---- 可选 retain 项（论文的 NPO-RT / NPO-KL 变体）----
            if retain_iter is not None:
                retain_batch = batch_to_device(next(retain_iter))
                if cfg.retain_mode == "rt":
                    # NPO-RT：retain 集标准交叉熵，直接"正着学"保留数据
                    retain_term = model(**retain_batch).loss
                else:
                    # NPO-KL：约束 retain 集输出分布不偏离参考模型
                    retain_term = token_kl_to_ref(model, ref_model, retain_batch)
                loss = loss + cfg.retain_weight * retain_term
                logs[f"retain_{cfg.retain_mode}"] = retain_term.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()

            if step % cfg.logging_steps == 0:
                log_str = " | ".join(f"{k} {v:.4f}" for k, v in logs.items())
                print(f"epoch {epoch} | step {step:>4d} | {log_str}")

    # ====================================================================== #
    # ④ 遗忘后评估 + 保存
    # ====================================================================== #
    after = evaluate_snapshot(model, data, tag="遗忘后")
    show_generations(model, tokenizer, data.demo_qa, DEVICE, cfg.max_new_tokens,
                     title="NPO 遗忘后")
    print_summary(f"NPO (β={cfg.beta}, retain_mode={cfg.retain_mode})", before, after)

    final_dir = f"{cfg.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[save] NPO 遗忘后的 LoRA adapter 已保存到 {final_dir}")


if __name__ == "__main__":
    main()
