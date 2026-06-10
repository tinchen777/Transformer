# -*- coding: utf-8 -*-
"""
finetune_lora.py —— 入口 1：用 LoRA 微调一个 HuggingFace 模型

完整流程（也是工业界/科研里最标准的 SFT + LoRA 工作流）：

  ① 下载分词器、基座模型（transformers）
  ② 下载数据集（datasets）→ 分词 → 切出验证集
  ③ 给模型注入 LoRA adapter（peft），冻结基座、只训练小矩阵
  ④ 用 transformers.Trainer 训练（它帮你处理训练循环、混合精度、
     梯度累积、学习率调度、checkpoint 保存等所有工程细节）
  ⑤ 测试：验证集困惑度 + 生成样例（微调前 vs 微调后对比）
  ⑥ 保存 LoRA adapter，供 unlearn_ga.py 继续使用

运行：
  cd <仓库根目录>
  python -m llm_pipeline.finetune_lora
"""

from transformers import Trainer, TrainingArguments, set_seed

from .config import DEVICE, DTYPE, FinetuneConfig
from .data_utils import build_collator, build_dataloader, build_tokenized_dataset
from .evaluate import compute_perplexity, show_generations
from .model_utils import add_lora, load_base_model, load_tokenizer


def main():
    cfg = FinetuneConfig()
    set_seed(cfg.seed)  # 固定所有随机种子（python/numpy/torch），保证实验可复现

    # ======================================================================= #
    # ① 加载分词器和基座模型
    # ======================================================================= #
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_base_model(cfg.model_name, DTYPE, DEVICE)

    # ======================================================================= #
    # ② 准备数据：下载 → 分词 → 切训练/验证集
    # ======================================================================= #
    tokenized, raw = build_tokenized_dataset(
        cfg.dataset_name, cfg.dataset_config, tokenizer, cfg.max_len,
        num_samples=cfg.num_train_samples,
    )
    # train_test_split 是 datasets 库自带的：随机切 10% 当验证集，
    # 训练时定期在验证集上算 loss，监控有没有过拟合
    split = tokenized.train_test_split(test_size=cfg.eval_ratio, seed=cfg.seed)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"[data] 训练集 {len(train_dataset)} 条 / 验证集 {len(eval_dataset)} 条")

    # 留几条原始 QA，训练前后各生成一次，做定性对比
    demo_qa = [raw[i] for i in range(cfg.num_demo_questions)]

    # ======================================================================= #
    # ③ 微调前先看看基座模型的表现（作为 baseline 对照）
    # ======================================================================= #
    show_generations(model, tokenizer, demo_qa, DEVICE, cfg.max_new_tokens,
                     title="微调前（基座模型，应该答不上 TOFU 的虚构知识）")

    # ======================================================================= #
    # ④ 注入 LoRA：冻结基座，只训练低秩矩阵
    # ======================================================================= #
    model = add_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)

    # ======================================================================= #
    # ⑤ 配置并启动 Trainer
    # ======================================================================= #
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,                  # checkpoint 保存目录
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",                 # 余弦退火：学习率平滑降到 0
        warmup_ratio=cfg.warmup_ratio,              # 开头先线性升温，避免一上来步子太大
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,            # 每 N 步打印一次训练 loss
        eval_strategy="epoch",                      # 每个 epoch 结束在验证集上评估
        save_strategy="epoch",                      # 每个 epoch 结束存一个 checkpoint
        bf16=(DTYPE.is_floating_point and DEVICE == "cuda"),  # GPU 上开 bf16 混合精度
        label_names=["labels"],                     # 明确告诉 Trainer label 列名
                                                    # （peft 包装后无法自动推断）
        report_to="none",                           # 不上报 wandb 等实验跟踪平台
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=build_collator(tokenizer),    # 动态 padding 的 collator
        processing_class=tokenizer,                 # 让 Trainer 顺便把分词器存进 checkpoint
    )

    print("\n[train] 开始 LoRA 微调 ...")
    trainer.train()

    # ======================================================================= #
    # ⑥ 训练后测试
    # ======================================================================= #
    # 6.1 定量：验证集困惑度
    eval_loader = build_dataloader(eval_dataset, tokenizer, cfg.batch_size, shuffle=False)
    metrics = compute_perplexity(model, eval_loader, DEVICE)
    print(f"\n[eval] 验证集: loss = {metrics['loss']:.4f}, "
          f"perplexity = {metrics['perplexity']:.2f}")

    # 6.2 定性：同样的问题再生成一次，和微调前对比
    show_generations(model, tokenizer, demo_qa, DEVICE, cfg.max_new_tokens,
                     title="微调后（应该能复述 TOFU 里的虚构知识）")

    # ======================================================================= #
    # ⑦ 保存最终 LoRA adapter
    # ======================================================================= #
    # PeftModel.save_pretrained 只保存 adapter（几 MB），不保存基座权重。
    # unlearn_ga.py 会用 model_utils.load_finetuned_model 把它挂回基座。
    final_dir = f"{cfg.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[save] LoRA adapter 已保存到 {final_dir}")


if __name__ == "__main__":
    main()
