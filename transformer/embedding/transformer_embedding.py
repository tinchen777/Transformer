# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
from torch import nn
from typing import (Any, TYPE_CHECKING)

from .positional_encoding import PositionalEncoding
from .token_embeddings import TokenEmbedding

if TYPE_CHECKING:
    from ..types import Tensor


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, drop_prob: float = 0.1, device: Any = "cpu"):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        :param max_len: max sequence length
        :param drop_prob: dropout probability
        :param device: hardware device setting
        """
        super().__init__()

        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len=max_len, device=device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        outputs = self.drop_out(tok_emb + pos_emb)

        return outputs
