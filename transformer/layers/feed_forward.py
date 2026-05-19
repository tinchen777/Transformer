# -*- coding: utf-8 -*-
# Python version: 3.10
from torch import nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, drop_prob: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
