# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
from torch import nn
from typing import (Optional, TYPE_CHECKING)

from ._utils import ModuleConfig
from .modules import (
    Encoder,
    Decoder,
    DecoderOnly
)

if TYPE_CHECKING:
    import torch
    Tensor = torch.Tensor


class Transformer(nn.Module):
    """
    A standard transformer model with an encoder and a decoder.
    """
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        enc_config: Optional[ModuleConfig] = None,
        dec_config: Optional[ModuleConfig] = None
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            config=enc_config
        )
        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            config=dec_config
        )
        self.linear = nn.Linear(self.decoder.config.d_model, trg_vocab_size, bias=False)

    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        """
        Forward pass for the transformer.

        Parameters
        ----------
            src : Tensor
                The indexing tensor of source sequence, shape as `[bsz, len_src]`.

            trg : Tensor
                The indexing tensor of target sequence, shape as `[bsz, len_trg]`.

        Returns
        -------
            Tensor
                The output tensor of the transformer, shape as `[bsz, len_trg, trg_vocab_size]`.
        """
        # encoder self attention mask
        src_mask = self.encoder.make_pad_mask(src)
        # src_mask: [bsz, 1, 1, src_len]

        enc_outputs = self.encoder(src, src_mask=src_mask)
        dec_outputs = self.decoder(trg, enc_outputs, src_mask=src_mask)
        logits = self.linear(dec_outputs)
        # logits: [bsz, len_trg, trg_vocab_size]

        return logits


class DecoderOnlyTransformer(nn.Module):
    """
    A transformer with only a decoder.
    """
    def __init__(
        self,
        vocab_size: int,
        config: Optional[ModuleConfig] = None
    ):
        super().__init__()

        self.decoder = DecoderOnly(
            trg_vocab_size=vocab_size,
            config=config
        )
        self.linear = nn.Linear(self.decoder.config.d_model, vocab_size, bias=False)

    def forward(self, trg: Tensor) -> Tensor:
        """
        Forward pass for the decoder-only transformer.

        Parameters
        ----------
            trg : Tensor
                The indexing tensor of target sequence, shape as `[bsz, len_trg]`.

        Returns
        -------
            Tensor
                The output tensor of the transformer, shape as `[bsz, len_trg, vocab_size]`.
        """
        dec_outputs = self.decoder(trg)
        logits = self.linear(dec_outputs)
        # logits: [bsz, len_trg, trg_vocab_size]

        return logits
