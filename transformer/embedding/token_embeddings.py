# -*- coding: utf-8 -*-
# Python version: 3.10
from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super().__init__(vocab_size, d_model, padding_idx=1)
