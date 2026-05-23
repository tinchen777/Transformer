# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    Tensor = torch.Tensor


@dataclass
class ModuleConfig:
    """
    Configuration for Transformer modules.
    """
    pad_idx: int = 0
    unk_idx: int = 1
    sos_idx: int = 2
    eos_idx: int = 3
    d_model: int = 512
    d_qk: int = 64
    d_v: int = 64
    n_layers: int = 6
    n_heads: int = 8
    max_len: int = 5000
    d_ff: int = 2048
    drop_prob: float = 0.1


def make_pad_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
    """
    Make padding mask for encoder and decoder.

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


def make_causal_mask(seq: Tensor) -> Tensor:
    """
    Make causal mask for decoder.

    Parameters
    ----------
        seq : Tensor
            The sequence tensor, shape as `[bsz, len_seq]`.

    Returns
    -------
        Tensor
            The subsequent mask, shape as `[len_seq, len_seq]`.
    """
    len_seq = seq.size(1)

    return torch.triu(
        torch.ones(len_seq, len_seq, device=seq.device, dtype=torch.bool),
        diagonal=1
    )


def make_merged_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
    """
    Make merged mask for decoder.

    Parameters
    ----------
        seq : Tensor
            The sequence tensor, shape as `[bsz, len_seq]`.

        pad_idx : int, default `0`
            The index of the padding token.

    Returns
    -------
        Tensor
            The merged mask, shape as `[bsz, 1, len_seq, len_seq]`.
    """
    causal_mask = make_causal_mask(seq)
    # causal_mask: [len_seq, len_seq]
    pad_mask = make_pad_mask(seq, pad_idx)
    # pad_mask: [bsz, 1, 1, len_seq]
    mask = pad_mask | causal_mask
    # mask: [bsz, 1, len_seq, len_seq]

    return mask
