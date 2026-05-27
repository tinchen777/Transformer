# -*- coding: utf-8 -*-
# Python version: 3.10
from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token embedding layer.
    """
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)
