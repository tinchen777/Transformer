# -*- coding: utf-8 -*-
"""
run_clreg.py —— 对比表征塑形 Unlearning 训练入口

流程：
  ① 加载可训练模型 + 冻结副本
  ② 离线阶段：用冻结模型算出
       - forget 集全部表征（训练中当负样本 = "原位置"）
       - retain 池表征 → 为每个 forget 样本检索最近邻无害样本（正样本）
  ③ 训练：每步
       InfoNCE(拉向近邻/推离原位置) + GA(压低原答案概率)
       + retain 表征锚定 + retain 交叉熵
  ④ 遗忘前后评估 + 保存

注意训练时 batch 必须按"样本下标"组织（不能用普通 shuffle 的
DataLoader），因为每个 forget 样本要配它自己的正/负样本表征。

运行：python -m baselines.clreg.run_clreg
"""

import torch
from transformers import set_seed

from llm_pipeline.config import DEVICE, DTYPE
from llm_pipeline.data_utils import build_collator
from llm_pipeline.evaluate import show_generations
from llm_pipeline.model_utils import load_finetuned_model, load_tokenizer

from ..common import batch_to_device, build_unlearn_data, evaluate_snapshot, print_summary
from .config import CLRegConfig
from .method import forward_with_reps, info_nce, mine_nearest_neighbors, precompute_reps


def main():
    cfg = CLRegConfig()
    set_seed(cfg.seed)

    # ====================================================================== #
    # ① 模型：可训练副本 + 冻结副本
    # ====================================================================== #
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_finetuned_model(cfg.model_name, cfg.finetuned_adapter_dir,
                                 DTYPE, DEVICE, trainable=True)
    frozen_model = load_finetuned_model(cfg.model_name, cfg.finetuned_adapter_dir,
                                        DTYPE, DEVICE, trainable=False)
    frozen_model.eval()

    data = build_unlearn_data(cfg, tokenizer)
    forget_tok = data.forget_tokenized
    retain_pool = data.retain_tokenized.select(
        range(min(cfg.neighbor_pool_size, len(data.retain_tokenized))))

    # ====================================================================== #
    # ② 离线阶段：冻结表征 + 最近邻挖掘
    # ====================================================================== #
    print("\n[clreg] 离线阶段：计算冻结模型表征并挖掘无害近邻 ...")
    forget_reps_frozen = precompute_reps(   # [Nf, d] —— 训练中的负样本（原位置）
        frozen_model, forget_tok, tokenizer, cfg.layer_id, cfg.batch_size, DEVICE)
    retain_reps_frozen = precompute_reps(   # [Nr, d] —— 近邻检索的 key
        frozen_model, retain_pool, tokenizer, cfg.layer_id, cfg.batch_size, DEVICE)

    neighbor_idx = mine_nearest_neighbors(forget_reps_frozen, retain_reps_frozen)  # [Nf]
    positive_reps = retain_reps_frozen[neighbor_idx]   # [Nf, d] 逐样本的正样本表征

    # 打印几对"forget 问题 → 匹配到的无害近邻问题"，直观检查检索是否合理
    from llm_pipeline.data_utils import load_raw_dataset
    forget_raw = load_raw_dataset(cfg.dataset_name, cfg.forget_config)
    retain_raw = load_raw_dataset(cfg.dataset_name, cfg.retain_config)
    print("\n[clreg] 近邻匹配示例（forget → 最近的无害样本）：")
    for i in range(min(3, len(forget_raw))):
        print(f"  F: {forget_raw[i]['question']}")
        print(f"  R: {retain_raw[int(neighbor_idx[i])]['question']}\n")

    # ====================================================================== #
    # ③ 遗忘前评估
    # ====================================================================== #
    before = evaluate_snapshot(model, data, tag="遗忘前")
    show_generations(model, tokenizer, data.demo_qa, DEVICE, cfg.max_new_tokens,
                     title="对比表征塑形 遗忘前")

    # ====================================================================== #
    # ④ 训练循环
    # ====================================================================== #
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    collator = build_collator(tokenizer)

    # retain 的无限迭代器（做锚定项和 CE 项）
    import itertools
    retain_iter = itertools.cycle(data.retain_train_loader)

    model.train()
    step, stop = 0, False
    for epoch in range(1, cfg.num_epochs + 1):
        if stop:
            break
        # 手动按下标分 batch：每个样本要对上它自己的正/负样本表征
        perm = torch.randperm(len(forget_tok))
        for idx_batch in perm.split(cfg.batch_size):
            step += 1
            rows = [forget_tok[int(i)] for i in idx_batch]
            forget_batch = batch_to_device(collator(rows))

            # ---- (a) forget 前向：一次拿到 CE loss + 当前表征（锚点）----
            forget_ce, anchor = forward_with_reps(model, forget_batch, cfg.layer_id)

            # ---- (b) InfoNCE：拉向无害近邻（正），推离原位置（负）----
            loss_con = info_nce(
                anchor=anchor,
                positive=positive_reps[idx_batch],        # 逐样本配对的近邻表征
                negatives=forget_reps_frozen[idx_batch],  # 本 batch 的原位置表征
                tau=cfg.tau,
            )

            # ---- (c) retain：表征锚定（最小位移）+ 交叉熵（保持泛化）----
            retain_batch = batch_to_device(next(retain_iter))
            retain_ce, retain_rep = forward_with_reps(model, retain_batch, cfg.layer_id)
            with torch.no_grad():
                _, retain_rep_frozen = forward_with_reps(frozen_model, retain_batch,
                                                         cfg.layer_id)
            loss_anchor = ((retain_rep.float() - retain_rep_frozen.float()) ** 2).mean()

            # ---- (d) 组合总损失 ----
            total = (cfg.w_contrast * loss_con
                     + cfg.w_forget_ga * (-forget_ce)      # GA：压低原答案概率
                     + cfg.w_retain_anchor * loss_anchor
                     + cfg.w_retain_ce * retain_ce)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()

            if step % cfg.logging_steps == 0:
                print(f"epoch {epoch} | step {step:>4d} | "
                      f"infonce {loss_con.item():.4f} | "
                      f"forget_ce {forget_ce.item():.4f} (越大越'忘') | "
                      f"retain_anchor {loss_anchor.item():.5f} | "
                      f"retain_ce {retain_ce.item():.4f}")

            # forget CE 升到阈值 → 已忘够，提前停止
            if forget_ce.item() > cfg.forget_loss_threshold:
                print(f"\n[clreg] forget CE {forget_ce.item():.2f} 超过阈值 "
                      f"{cfg.forget_loss_threshold}，提前停止。")
                stop = True
                break

    # ====================================================================== #
    # ⑤ 遗忘后评估 + 保存
    # ====================================================================== #
    after = evaluate_snapshot(model, data, tag="遗忘后")
    show_generations(model, tokenizer, data.demo_qa, DEVICE, cfg.max_new_tokens,
                     title="对比表征塑形 遗忘后")
    print_summary(f"CLReg 风格对比塑形 (τ={cfg.tau}, layer={cfg.layer_id})",
                  before, after)

    final_dir = f"{cfg.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[save] 遗忘后的 LoRA adapter 已保存到 {final_dir}")


if __name__ == "__main__":
    main()
