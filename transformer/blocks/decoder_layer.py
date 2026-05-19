# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
from torch import nn
from typing import (Optional, TYPE_CHECKING)

from ..layers import (MultiHeadAttention, PositionwiseFeedForward, LayerNorm)

if TYPE_CHECKING:
    from ..types import Tensor


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, n_heads: int, drop_prob: float = 0.1):
        super().__init__()

        self.masked_mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.masked_mha_norm = LayerNorm(d_model)
        self.masked_mha_dropout = nn.Dropout(p=drop_prob)

        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.mha_norm = LayerNorm(d_model)
        self.mha_dropout = nn.Dropout(p=drop_prob)

        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, drop_prob=drop_prob)
        self.ff_norm = LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(p=drop_prob)

    def forward(self, dec_inputs: Tensor, enc_outputs: Optional[Tensor], trg_mask: Tensor, src_mask: Tensor):
        # 1. compute self attention
        res = dec_inputs
        x = self.masked_mha(input_Q=dec_inputs, input_K=dec_inputs, input_V=dec_inputs, attn_mask=trg_mask)
        # add and norm
        x = self.masked_mha_dropout(x)
        x = self.masked_mha_norm(x + res)

        if enc_outputs is not None:
            # 2. compute encoder - decoder attention
            res = x
            x = self.mha(input_Q=x, input_K=enc_outputs, input_V=enc_outputs, attn_mask=src_mask)
            # add and norm
            x = self.mha_dropout(x)
            x = self.mha_norm(x + res)

        # 3. positionwise feed forward network
        res = x
        x = self.ff(x)
        # add and norm
        x = self.ff_dropout(x)
        dec_inputs = self.ff_norm(x + res)

        return dec_inputs
