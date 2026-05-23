# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
from torch import nn
from typing import (Tuple, TYPE_CHECKING)

from .._utils import ModuleConfig
from ..layers import (
    make_mha_layers,
    make_ff_layers
)
from ..exceptions import EncoderError

if TYPE_CHECKING:
    import torch
    Tensor = torch.Tensor


class EncoderLayer(nn.Module):
    """
    Encoder layer.
    """
    def __init__(self, config: ModuleConfig):
        super().__init__()

        # multi head attention
        self.mha, self.mha_norm, self.mha_dropout = make_mha_layers(config)
        # positionwise feed forward network
        self.ff, self.ff_norm, self.ff_dropout = make_ff_layers(config)

    def forward(
        self,
        enc_inputs: Tensor,
        src_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the forward pass of the encoder layer.

        Parameters
        ----------
            enc_inputs : Tensor
                Input tensor to the encoder layer, shape as `[bsz, len_src, d_model]`.

            src_mask : Tensor
                Source mask, shape as `[bsz, 1, 1, len_src]`.

        Returns
        -------
            Tensor
                Output tensor from the encoder layer, shape as `[bsz, len_src, d_model]`.
            Tensor
                Attention weights from the multi-head attention, shape as `[bsz, n_heads, len_src, len_src]`.
        """
        try:
            # 1. compute multi head attention
            res = enc_inputs
            x, mha_attn = self.mha(
                Q_inputs=enc_inputs,
                K_inputs=enc_inputs,
                V_inputs=enc_inputs,
                attn_mask=src_mask
            )
            # add and norm
            x = self.mha_dropout(x)
            x = self.mha_norm(x + res)

            # 2. positionwise feed forward
            res = x
            x = self.ff(x)
            # add and norm
            x = self.ff_dropout(x)
            enc_outputs = self.ff_norm(x + res)

            return enc_outputs, mha_attn

        except Exception as e:
            raise EncoderError(
                "Encoder layer error") from e
