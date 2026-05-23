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
from ..exceptions import DecoderError

if TYPE_CHECKING:
    import torch
    Tensor = torch.Tensor


class DecoderLayer(nn.Module):
    """
    Decoder layer.
    """
    def __init__(self, config: ModuleConfig):
        super().__init__()

        # masked multi head attention
        self.masked_mha, self.masked_mha_norm, self.masked_mha_dropout = make_mha_layers(config)
        # multi head attention
        self.mha, self.mha_norm, self.mha_dropout = make_mha_layers(config)
        # positionwise feed forward network
        self.ff, self.ff_norm, self.ff_dropout = make_ff_layers(config)

    def forward(
        self,
        dec_inputs: Tensor,
        trg_mask: Tensor,
        enc_outputs: Tensor,
        src_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the forward pass of the decoder layer.

        Parameters
        ----------
            dec_inputs : Tensor
                Input tensor to the decoder layer, shape as `[bsz, len_trg, d_model]`.

            trg_mask : Tensor
                Target mask, shape as `[bsz, 1, len_trg, len_trg]`.

            enc_outputs : Tensor
                Output tensor from the encoder, shape as `[bsz, len_src, d_model]`.

            src_mask : Tensor
                Source mask, shape as `[bsz, 1, 1, len_src]`.

        Returns
        -------
            Tensor
                Output tensor from the decoder layer, shape as `[bsz, len_trg, d_model]`.
            Tensor
                Attention weights from the masked multi-head attention, shape as `[bsz, n_heads, len_trg, len_trg]`.
            Tensor
                Attention weights from the multi-head attention, shape as `[bsz, n_heads, len_trg, len_src]`.
        """
        try:
            # 1. compute masked multi head attention
            res = dec_inputs
            x, masked_mha_attn = self.masked_mha(
                Q_inputs=dec_inputs,
                K_inputs=dec_inputs,
                V_inputs=dec_inputs,
                attn_mask=trg_mask
            )
            # add and norm
            x = self.masked_mha_dropout(x)
            x = self.masked_mha_norm(x + res)

            # 2. compute multi head attention (encoder-decoder attention)
            res = x
            x, mha_attn = self.mha(
                Q_inputs=x,
                K_inputs=enc_outputs,
                V_inputs=enc_outputs,
                attn_mask=src_mask
            )
            # add and norm
            x = self.mha_dropout(x)
            x = self.mha_norm(x + res)

            # 3. positionwise feed forward network
            res = x
            x = self.ff(x)
            # add and norm
            x = self.ff_dropout(x)
            dec_outputs = self.ff_norm(x + res)

            return dec_outputs, masked_mha_attn, mha_attn

        except Exception as e:
            raise DecoderError(
                "Decoder layer error") from e


class DecoderOnlyLayer(nn.Module):
    """
    Decoder layer without encoder-decoder attention.
    """
    def __init__(self, config: ModuleConfig):
        super().__init__()

        # masked multi head attention
        self.masked_mha, self.masked_mha_norm, self.masked_mha_dropout = make_mha_layers(config)
        # positionwise feed forward network
        self.ff, self.ff_norm, self.ff_dropout = make_ff_layers(config)

    def forward(
        self,
        dec_inputs: Tensor,
        trg_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the forward pass of the decoder-only layer.

        Parameters
        ----------
            dec_inputs : Tensor
                Input tensor to the decoder-only layer, shape as `[bsz, len_trg, d_model]`.

            trg_mask : Tensor
                Target mask, shape as `[bsz, 1, len_trg, len_trg]`.

        Returns
        -------
            Tensor
                Output tensor from the decoder-only layer, shape as `[bsz, len_trg, d_model]`.
            Tensor
                Attention weights from the masked multi-head attention, shape as `[bsz, n_heads, len_trg, len_trg]`.
        """
        try:
            # 1. compute masked multi head attention
            res = dec_inputs
            x, masked_mha_attn = self.masked_mha(
                Q_inputs=dec_inputs,
                K_inputs=dec_inputs,
                V_inputs=dec_inputs,
                attn_mask=trg_mask
            )
            # add and norm
            x = self.masked_mha_dropout(x)
            x = self.masked_mha_norm(x + res)

            # 2. positionwise feed forward network
            res = x
            x = self.ff(x)
            # add and norm
            x = self.ff_dropout(x)
            dec_outputs = self.ff_norm(x + res)

            return dec_outputs, masked_mha_attn

        except Exception as e:
            raise DecoderError(
                "Decoder-only layer error") from e
