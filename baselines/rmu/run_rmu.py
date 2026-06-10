# -*- coding: utf-8 -*-
"""
run_rmu.py —— RMU Unlearning 训练入口

与其他 baseline 的一个不同点：RMU 直接更新基座权重的一小部分
（指定层的 MLP 输出投影），而不是更新 LoRA adapter。
所以这里先把微调阶段的 LoRA 合并进基座（merge_and_unload），
得到一个"知识已写入权重"的普通模型，再在它上面做 RMU。

流程：
  ① 加载微调模型 → 合并 LoRA → 复制一份冻结版（提供 retain 表征基准）
  ② 冻结全模型，只解冻 update_layer_ids 指定层的 MLP 输出投影
  ③ 每步：forget 表征 → 推向 c·u；retain 表征 → 锚定到冻结模型
  ④ 遗忘前后评估 + 保存

运行：python -m baselines.rmu.run_rmu
"""

import copy
import itertools

import torch
from transformers import set_seed

from llm_pipeline.config import DEVICE, DTYPE
from llm_pipeline.evaluate import show_generations
from llm_pipeline.model_utils import load_finetuned_model, load_tokenizer

from ..common import batch_to_device, build_unlearn_data, evaluate_snapshot, print_summary
from .config import RMUConfig
from .method import (freeze_except_mlp_layers, make_control_vector,
                     rmu_forget_loss, rmu_retain_loss)


def main():
    cfg = RMUConfig()
    set_seed(cfg.seed)

    # ====================================================================== #
    # ① 模型准备：合并 LoRA + 冻结副本
    # ====================================================================== #
    tokenizer = load_tokenizer(cfg.model_name)
    peft_model = load_finetuned_model(cfg.model_name, cfg.finetuned_adapter_dir,
                                      DTYPE, DEVICE, trainable=False)
    # merge_and_unload: 把 LoRA 的 B·A 增量加回基座权重 W，
    # 返回一个不再依赖 peft 的普通模型（W' = W + (α/r)·B·A）
    model = peft_model.merge_and_unload()
    # 冻结副本：训练中它提供"原模型的 retain 表征"作为锚定目标
    frozen_model = copy.deepcopy(model).eval()
    for p in frozen_model.parameters():
        p.requires_grad_(False)

    # ====================================================================== #
    # ② 只解冻指定层的 MLP 输出投影；生成固定控制向量 c·u
    # ====================================================================== #
    trainable_params = freeze_except_mlp_layers(model, cfg.update_layer_ids)
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

    control_vec = make_control_vector(
        model.config.hidden_size, cfg.steering_coeff, DEVICE, DTYPE)

    # ====================================================================== #
    # ③ 数据 + 遗忘前评估
    # ====================================================================== #
    data = build_unlearn_data(cfg, tokenizer)
    before = evaluate_snapshot(model, data, tag="遗忘前")
    show_generations(model, tokenizer, data.demo_qa, DEVICE, cfg.max_new_tokens,
                     title="RMU 遗忘前")

    # ====================================================================== #
    # ④ 训练循环：L = L_forget + α·L_retain
    # ====================================================================== #
    retain_iter = itertools.cycle(data.retain_train_loader)
    model.train()
    step = 0
    done = False
    while not done:
        for forget_batch in data.forget_train_loader:
            step += 1

            # forget 项：表征推向固定随机方向 c·u
            forget_batch = batch_to_device(forget_batch)
            loss_f = rmu_forget_loss(model, forget_batch, cfg.layer_id, control_vec)

            # retain 项：表征锚定到冻结模型（保护可用性）
            retain_batch = batch_to_device(next(retain_iter))
            loss_r = rmu_retain_loss(model, frozen_model, retain_batch, cfg.layer_id)

            loss = loss_f + cfg.alpha * loss_r

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()

            if step % cfg.logging_steps == 0:
                print(f"step {step:>4d}/{cfg.max_steps} | "
                      f"forget_mse {loss_f.item():.4f} (→0 表示表征已对齐到 c·u) | "
                      f"retain_mse {loss_r.item():.6f} (希望保持很小)")

            # RMU 用固定步数控制训练量（论文设定），步数到了就停
            if step >= cfg.max_steps:
                done = True
                break

    # ====================================================================== #
    # ⑤ 遗忘后评估 + 保存
    # ====================================================================== #
    after = evaluate_snapshot(model, data, tag="遗忘后")
    show_generations(model, tokenizer, data.demo_qa, DEVICE, cfg.max_new_tokens,
                     title="RMU 遗忘后")
    print_summary(f"RMU (layer={cfg.layer_id}, c={cfg.steering_coeff}, α={cfg.alpha})",
                  before, after)

    # 注意：RMU 改的是基座权重本身，保存的是完整模型（不是几 MB 的 adapter）
    final_dir = f"{cfg.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[save] RMU 遗忘后的完整模型已保存到 {final_dir}")


if __name__ == "__main__":
    main()
