# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
import math
import torch
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    Tensor = torch.Tensor


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, tok: Tensor) -> Tensor:
        """
        Add positional encoding to token embedding.

        Parameters
        ----------
            tok : Tensor
                The token embedding, shape as `[batch_size, len_seq, d_model]`.

        Returns
        -------
            Tensor
                The token embedding with positional encoding, shape as `[batch_size, len_seq, d_model]`.
        """
        len_seq = tok.size(1)
        return tok + self.encoding[:len_seq, :]  # type: ignore
