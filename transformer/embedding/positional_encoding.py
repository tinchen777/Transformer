# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
import torch
from torch import nn
from typing import (Any, TYPE_CHECKING)

if TYPE_CHECKING:
    from ..types import Tensor


class PositionalEncoding(nn.Module):
    """
    Compute sinusoid encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000, device: Any = "cpu"):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device, requires_grad=False)

        pos = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, dtype=torch.float, device=device)
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x: Tensor) -> Tensor:
        # self.encoding
        # [max_len = 512, d_model = 512]
        # FIXME
        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
