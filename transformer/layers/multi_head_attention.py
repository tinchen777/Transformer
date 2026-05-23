# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
import torch
from torch import nn
import numpy as np
from typing import (Tuple, Optional, TYPE_CHECKING)

from ..exceptions import (
    ScaledDotProductAttentionError,
    MultiHeadAttentionError
)

if TYPE_CHECKING:
    Tensor = torch.Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    """
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        attn_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute Scaled Dot-Product Attention.

        Parameters
        ----------

            Q : Tensor
                Query tensor,  shape as `[bsz, n_heads, len_q, d_qk]`.

            K : Tensor
                Key tensor,  shape as `[bsz, n_heads, len_kv, d_qk]`.

            V : Tensor
                Value tensor,  shape as `[bsz, n_heads, len_kv, d_v]`.

            attn_mask : Tensor
                Attention mask,  shape as `[bsz, 1, -1, len_kv]`.

        Returns
        -------
            Tensor
                Context tensor, shape as `[bsz, n_heads, len_q, d_v]`.
        """
        try:
            d_qk = K.size(-1)

            # 1. dot product Query with Key^T to compute similarity scores
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_qk)
            # scores : [bsz, n_heads, len_q, len_kv]

            # 2. apply masking
            # Fills elements of self tensor with value where mask is True.
            scores.masked_fill_(attn_mask, float("-inf"))

            # 3. pass them softmax to make [0, 1] range
            attn = self.softmax(scores)
            # attn : [bsz, n_heads, len_q, len_kv]

            # 4. multiply with Value
            context = torch.matmul(attn, V)
            # context: [bsz, n_heads, len_q, d_v]

            return context, attn

        except Exception as e:
            raise ScaledDotProductAttentionError(
                "Scaled dot-product attention calculation error") from e


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_qk: Optional[int] = None,
        d_v: Optional[int] = None
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model

        _d_default = d_model // n_heads
        self.d_qk = d_qk if d_qk is not None else _d_default
        self.d_v = d_v if d_v is not None else _d_default

        self.attention = ScaledDotProductAttention()

        self.W_Q = nn.Linear(d_model, self.d_qk * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_qk * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads, bias=False)
        self.linear = nn.Linear(n_heads * self.d_v, d_model, bias=False)

    def forward(
        self,
        Q_inputs: Tensor,
        K_inputs: Tensor,
        V_inputs: Tensor,
        attn_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute Multi-Head Attention.

        Parameters
        ----------
            Q_inputs : Tensor
                Query input tensor, shape as `[bsz, len_q, d_model]`.

            K_inputs : Tensor
                Key input tensor, shape as `[bsz, len_kv, d_model]`.

            V_inputs : Tensor
                Value input tensor, shape as `[bsz, len_kv, d_model]`.

            attn_mask : Tensor
                Attention mask, shape as `[bsz, 1, -1, len_kv]`.

        Returns
        -------
            Tensor
                Output tensor, shape as `[bsz, len_q, d_model]`.
            Tensor
                Attention weights, shape as `[bsz, n_heads, len_q, len_kv]`.
        """
        try:
            bsz = Q_inputs.size(0)

            """
            (bsz, len_seq, d_model)
            1. -W-> (bsz, len_seq, *d_model)
            2. -split-> (bsz, len_seq, n_heads, d_qk)
            3. -trans-> (bsz, n_heads, len_seq, d_qk)
            """
            Q = self.W_Q(Q_inputs).view(bsz, -1, self.n_heads, self.d_qk).transpose(1, 2)
            # Q: [bsz, n_heads, len_q, d_qk]
            K = self.W_K(K_inputs).view(bsz, -1, self.n_heads, self.d_qk).transpose(1, 2)
            # K: [bsz, n_heads, len_kv, d_qk]
            V = self.W_V(V_inputs).view(bsz, -1, self.n_heads, self.d_v).transpose(1, 2)
            # V: [bsz, n_heads, len_kv, d_v]

            context, attn = self.attention(Q, K, V, attn_mask=attn_mask)
            # context: [bsz, n_heads, len_q, d_v]
            # attn: [bsz, n_heads, len_q, len_kv]

            context = context.transpose(1, 2).reshape(bsz, -1, self.d_model)
            # context: [bsz, len_q, n_heads * d_v]

            outputs = self.linear(context)
            # outputs: [bsz, len_q, d_model]

            return outputs, attn

        except Exception as e:
            raise MultiHeadAttentionError(
                "Multi-head attention calculation error") from e
