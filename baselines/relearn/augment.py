# -*- coding: utf-8 -*-
"""
augment.py —— ReLearn 的数据增强：为 forget 问题构造"无害替代答案"

论文做法：用一个强 LLM（如 GPT-4）按三条原则批量生成替代答案：
  1. 不泄露原答案中的敏感/目标信息；
  2. 和问题语境相关、语言自然流畅（不是生硬的"我拒绝回答"）；
  3. 同一问题生成多条不同表述，保证多样性。
随后还会做质量校验（确认确实不含敏感信息）。

本文件提供两条路径：
  - 模板版（默认）：内置一组多样化的"安全回避"模板，离线可用，
    跑通流程、做小规模对照足够；
  - LLM 版（可选）：build_relearn_examples 接受一个 llm_generate_fn
    回调，你可以接入任何 API 模型按论文方式生成更高质量的替代答案。
"""

import random

# 一组风格各异的"安全回避"答案模板。
# 设计原则：流畅、礼貌、和 QA 语境兼容，但不包含任何具体事实 ——
# 模型学会的是"对这类问题给出不含信息量的自然回答"。
SAFE_ANSWER_TEMPLATES = [
    "I'm sorry, but I don't have reliable information about that.",
    "I'm not familiar with the specific details you're asking about.",
    "There isn't any verified information I can share on that topic.",
    "That's not something I have accurate knowledge of, so I'd rather not speculate.",
    "I don't have enough information to answer that question accurately.",
    "Unfortunately, I can't provide specific details about that.",
]


def template_safe_answers(question: str, n_variants: int, rng: random.Random) -> list[str]:
    """
    模板版替代答案：从模板池里无放回抽 n_variants 条。

    question 参数目前没用到（模板是通用的），保留它是为了
    和 LLM 版生成函数签名一致 —— 接入 LLM 时自然就能用上问题内容。
    """
    n = min(n_variants, len(SAFE_ANSWER_TEMPLATES))
    return rng.sample(SAFE_ANSWER_TEMPLATES, n)


def build_relearn_examples(
    forget_raw,
    retain_raw,
    num_safe_variants: int,
    retain_mix_ratio: float,
    seed: int,
    llm_generate_fn=None,
) -> list[dict]:
    """
    构造 ReLearn 的训练数据：增强后的 forget QA + 混入的 retain QA。

    参数
    ----
    forget_raw      : TOFU forget 子集（原始 {question, answer}）
    retain_raw      : TOFU retain 子集
    llm_generate_fn : 可选回调 fn(question, n) -> list[str]，
                      接入外部 LLM 生成替代答案；None 则用内置模板。

    返回
    ----
    list[{"question", "answer"}]，已 shuffle，可直接喂给分词流程。
    """
    rng = random.Random(seed)
    generate = llm_generate_fn or (lambda q, n: template_safe_answers(q, n, rng))

    # ① forget 部分：每个问题配 num_safe_variants 条无害答案。
    #    注意：原答案被彻底丢弃 —— "遗忘"靠的是新答案的覆盖学习。
    examples = []
    for row in forget_raw:
        for safe_answer in generate(row["question"], num_safe_variants):
            examples.append({"question": row["question"], "answer": safe_answer})
    n_forget_aug = len(examples)

    # ② retain 部分：按比例随机抽取，保持原答案不动（复习保留知识）
    n_retain = min(int(n_forget_aug * retain_mix_ratio), len(retain_raw))
    retain_indices = rng.sample(range(len(retain_raw)), n_retain)
    for i in retain_indices:
        row = retain_raw[i]
        examples.append({"question": row["question"], "answer": row["answer"]})

    rng.shuffle(examples)
    print(f"[relearn] 训练数据构造完成: forget 增强 {n_forget_aug} 条 "
          f"+ retain 混入 {n_retain} 条 = 共 {len(examples)} 条")
    return examples
