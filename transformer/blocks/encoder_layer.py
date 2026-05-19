# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
from torch import nn
from typing import TYPE_CHECKING

from ..layers import (MultiHeadAttention, PositionwiseFeedForward, LayerNorm)

if TYPE_CHECKING:
    from ..types import Tensor


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, drop_prob: float = 0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.mha_norm = LayerNorm(d_model)
        self.mha_dropout = nn.Dropout(p=drop_prob)

        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, drop_prob=drop_prob)
        self.ff_norm = LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(p=drop_prob)

    def forward(self, enc_inputs: Tensor, src_mask: Tensor) -> tuple[Tensor, Tensor]:
        # 1. compute multi head attention
        res = enc_inputs
        x, attn = self.mha(input_Q=enc_inputs, input_K=enc_inputs, input_V=enc_inputs, attn_mask=src_mask)
        # add and norm
        x = self.mha_dropout(x)
        x = self.mha_norm(x + res)

        # 2. positionwise feed forward network
        res = x
        x = self.ff(x)
        # add and norm
        x = self.ff_dropout(x)
        enc_outputs = self.ff_norm(x + res)

        return enc_outputs, attn
