# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import (Optional, Literal, TYPE_CHECKING)

if TYPE_CHECKING:
    Tensor = torch.Tensor


@dataclass
class ModuleConfig:
    """
    Configuration for Transformer modules.
    """
    pad_idx: int = 0
    unk_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3
    d_model: int = 512
    d_qk: int = 64
    d_v: int = 64
    n_layers: int = 6
    n_heads: int = 8
    max_len: int = 5000
    d_ff: int = 2048
    drop_prob: float = 0.1
    use_prefix_attention: bool = True  # for prefix LM, whether to use prefix attention (bidirectional) in the prefix part


def make_pad_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
    """
    Make padding mask for encoder and decoder, where `True` indicates the position of the padding token.

    Parameters
    ----------
        seq : Tensor
            The sequence tensor, shape as `[bsz, len_seq]`.

        pad_idx : int, default `0`
            The index of the padding token.

    Returns
    -------
        Tensor
            The padding mask, shape as `[bsz, 1, 1, len_seq]`.
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

# BUG 对于每个sample都不同，不能直接在外面做成一个固定的 mask
def make_causal_mask(seq: Tensor, prefix_len: int = 0) -> Tensor:
    """
    Make causal mask for decoder, where `True` indicates the position that should be masked (i.e., not visible to the current token).

    Parameters
    ----------
        seq : Tensor
            The sequence tensor, shape as `[bsz, len_seq]`.

        prefix_len : int, default `0`
            The length of the prefix (e.g., question part in a Q&A pair) that should be fully visible to itself (bidirectional attention), while the rest of the sequence is masked with causal (unidirectional) attention.

    Returns
    -------
        Tensor
            The subsequent mask, shape as `[len_seq, len_seq]`.
    """
    len_seq = seq.size(1)

    mask = torch.triu(
        torch.ones(len_seq, len_seq, device=seq.device, dtype=torch.bool),
        diagonal=1
    )
    if prefix_len > 1:
        mask[:prefix_len, :prefix_len] = False  # Q 内部的上三角也设为 False (双向)

    return mask


def make_merged_mask(seq: Tensor, pad_idx: int = 0, prefix_len: int = 0) -> Tensor:
    """
    Make merged mask for decoder, where `True` indicates the position that should be masked (i.e., not visible to the current token). This is a combination of padding mask and causal mask.

    Parameters
    ----------
        seq : Tensor
            The sequence tensor, shape as `[bsz, len_seq]`.

        pad_idx : int, default `0`
            The index of the padding token.

        prefix_len : int, default `0`
            The length of the prefix (e.g., question part in a Q&A pair) that should be fully visible to itself (bidirectional attention), while the rest of the sequence is masked with causal (unidirectional) attention.

    Returns
    -------
        Tensor
            The merged mask, shape as `[bsz, 1, len_seq, len_seq]`.
    """
    causal_mask = make_causal_mask(seq, prefix_len)
    # causal_mask: [len_seq, len_seq]
    pad_mask = make_pad_mask(seq, pad_idx)
    # pad_mask: [bsz, 1, 1, len_seq]
    mask = pad_mask | causal_mask
    # mask: [bsz, 1, len_seq, len_seq]

    return mask


def select_next_token(
    logits: Tensor,
    strategy: Literal["greedy", "sample"] = "greedy",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Tensor:
    """
    Select the next token for each sample in a batch from a logits distribution.

    Parameters
    ----------
        logits: Tensor
            Raw logits from the last decoder step, shape `[bsz, vocab_size]`.

        strategy : Literal["greedy", "sample"]
            The decoding strategy to use.
            - `"greedy"`: always pick the highest-probability token.
            - `"sample"`: draw from the distribution (supports temperature, top-k, and top-p filtering).

        temperature : float, default `1.0`
            :param:`strategy` is `"sample"` ONLY. Scaling factor applied before softmax.
            - `< 1`: Sharpen the distribution;
            - `> 1`: Flatten it.
            Approaches greedy as temperature → 0.

        top_k : Optional[int], default `None`
            :param:`strategy` is `"sample"` ONLY.
            - _int_: Restrict sampling to the top-k tokens by logit value. Must be > 0.

        top_p : Optional[float], default `None`
            :param:`strategy` is `"sample"` ONLY.
            - _float_: Restrict sampling to the top-p tokens by cumulative probability. Must be in (0, 1).

    Returns
    -------
        Tensor
            The selected tokens, shape as `[bsz, 1]`.
    """
    if logits.dim() != 2:
        raise ValueError(
            f"Expected logits of shape [bsz, vocab_size], got {logits.shape}"
        )

    # ------------------------------------------------------------------
    # Greedy: return the highest-logit token for every sample
    # ------------------------------------------------------------------
    if strategy == "greedy":
        return logits.argmax(dim=-1, keepdim=True)  # [bsz, 1]

    # ------------------------------------------------------------------
    # Sample: optionally filter logits, then draw from the distribution
    # ------------------------------------------------------------------
    if strategy == "sample":
        # Temperature scaling — squash or flatten the distribution
        logits = logits / max(temperature, 1e-8)

        # Top-k filtering — zero out every token outside the top k
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))  # guard against k > vocab
            topk_vals, _ = torch.topk(logits, k, dim=-1)  # [bsz, k]
            threshold = topk_vals[:, [-1]]  # [bsz, 1] — k-th largest value
            logits = logits.masked_fill(logits < threshold, -float("inf"))

        # Top-p (nucleus) filtering — zero out the low-probability tail
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Mark tokens whose cumulative probability already exceeds top_p
            remove_mask = cum_probs > top_p
            # Shift right by one so the token that *pushes* cumprob over top_p is kept
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False                         # always keep the top token

            sorted_logits = sorted_logits.masked_fill(remove_mask, -float("inf"))
            # Scatter back to the original vocab ordering
            logits = torch.full_like(logits, -float("inf")).scatter(
                -1, sorted_idx, sorted_logits
            )

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)          # [bsz, 1]

    raise ValueError(
        f"Invalid strategy for `select_next_token`: {strategy}. Must be 'greedy' or 'sample'.")
