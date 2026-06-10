# -*- coding: utf-8 -*-
"""
run_relearn.py —— ReLearn Unlearning 训练入口

整个方法没有任何"反向"操作：构造好（forget 问题 → 无害答案）+
（retain 问题 → 原答案）的混合数据集后，就是一次标准的 SFT。
遗忘的发生机制是"覆盖"：对同一个问题，新的安全回答的概率被抬高，
原敏感答案的概率自然被挤下去（softmax 此消彼长）。

预期现象（和梯度上升类方法对比时最值得观察的点）：
  - forget PPL（按原答案算）上升 —— 原答案概率确实降低了；
  - 但模型对 forget 问题的输出依然是流畅自然的句子，
    不会像 GA 那样退化成乱码 —— 这正是论文的核心卖点。

运行：python -m baselines.relearn.run_relearn
"""

import torch
from datasets import Dataset
from transformers import set_seed

from llm_pipeline.config import DEVICE, DTYPE
from llm_pipeline.data_utils import build_dataloader, tokenize_dataset
from llm_pipeline.evaluate import show_generations
from llm_pipeline.model_utils import load_finetuned_model, load_tokenizer

from ..common import batch_to_device, build_unlearn_data, evaluate_snapshot, print_summary
from .augment import build_relearn_examples
from .config import ReLearnConfig


def main():
    cfg = ReLearnConfig()
    set_seed(cfg.seed)

    # ====================================================================== #
    # ① 模型与评估数据（评估口径与其他 baseline 完全一致）
    # ====================================================================== #
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_finetuned_model(cfg.model_name, cfg.finetuned_adapter_dir,
                                 DTYPE, DEVICE, trainable=True)
    data = build_unlearn_data(cfg, tokenizer)

    before = evaluate_snapshot(model, data, tag="遗忘前")
    show_generations(model, tokenizer, data.demo_qa, DEVICE, cfg.max_new_tokens,
                     title="ReLearn 遗忘前")

    # ====================================================================== #
    # ② 构造 ReLearn 训练集：增强 forget + 混入 retain，再走标准分词流程
    # ====================================================================== #
    from llm_pipeline.data_utils import load_raw_dataset
    forget_raw = load_raw_dataset(cfg.dataset_name, cfg.forget_config)
    retain_raw = load_raw_dataset(cfg.dataset_name, cfg.retain_config)

    examples = build_relearn_examples(
        forget_raw, retain_raw,
        num_safe_variants=cfg.num_safe_variants,
        retain_mix_ratio=cfg.retain_mix_ratio,
        seed=cfg.seed,
        llm_generate_fn=None,   # 接入外部 LLM 时换成你的生成回调
    )
    # 包成 HF Dataset 后复用 llm_pipeline 的分词逻辑（label mask 等都一致）
    train_tokenized = tokenize_dataset(Dataset.from_list(examples), tokenizer, cfg.max_len)
    train_loader = build_dataloader(train_tokenized, tokenizer, cfg.batch_size, shuffle=True)

    # ====================================================================== #
    # ③ 标准 SFT 训练循环（纯梯度下降，没有任何反向技巧）
    # ====================================================================== #
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

    model.train()
    step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        for batch in train_loader:
            step += 1
            batch = batch_to_device(batch)

            loss = model(**batch).loss   # 普通交叉熵：学习新的（安全）答案

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()

            if step % cfg.logging_steps == 0:
                print(f"epoch {epoch} | step {step:>4d} | sft loss {loss.item():.4f}")

    # ====================================================================== #
    # ④ 遗忘后评估 + 保存
    # ====================================================================== #
    after = evaluate_snapshot(model, data, tag="遗忘后")
    show_generations(model, tokenizer, data.demo_qa, DEVICE, cfg.max_new_tokens,
                     title="ReLearn 遗忘后（注意输出应仍然流畅，只是不含原答案信息）")
    print_summary(f"ReLearn (variants={cfg.num_safe_variants}, "
                  f"mix={cfg.retain_mix_ratio})", before, after)

    final_dir = f"{cfg.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[save] ReLearn 遗忘后的 LoRA adapter 已保存到 {final_dir}")


if __name__ == "__main__":
    main()
