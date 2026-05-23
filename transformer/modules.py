# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
from torch import nn
from typing import (Optional, List, TYPE_CHECKING)

from ._utils import (
    ModuleConfig,
    make_pad_mask,
    make_causal_mask,
    make_merged_mask
)
from .blocks import (
    EncoderLayer,
    DecoderLayer,
    DecoderOnlyLayer
)
from .embedding import (
    TokenEmbedding,
    PositionalEncoding
)
from .exceptions import EmbeddingError

if TYPE_CHECKING:
    import torch
    Tensor = torch.Tensor


class BaseModule(nn.Module):
    """
    Base module for encoder and decoder.
    """
    def __init__(self, vocab_size: int, config: Optional[ModuleConfig] = None):
        super().__init__()

        self.config = ModuleConfig() if config is None else config

        self.tok_emb = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=self.config.d_model,
            padding_idx=self.config.pad_idx
        )
        self.pos_emb = PositionalEncoding(
            d_model=self.config.d_model,
            max_len=self.config.max_len
        )
        self.drop_out = nn.Dropout(p=self.config.drop_prob)

    def make_pad_mask(self, seq: Tensor) -> Tensor:
        return make_pad_mask(seq, self.config.pad_idx)


class Encoder(BaseModule):
    """
    Encoder module.
    """
    def __init__(
        self,
        src_vocab_size: int,
        config: Optional[ModuleConfig] = None
    ):
        super().__init__(vocab_size=src_vocab_size, config=config)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(self.config) for _ in range(self.config.n_layers)
        ])

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Forward pass for the encoder.

        Parameters
        ----------
            src : Tensor
                The indexing tensor of source sequence, shape as `[bsz, len_src]`.

            src_mask : Tensor
                The source mask, shape as `[bsz, 1, 1, len_src]`.

        Returns
        -------
            Tensor
                The output tensor of the encoder, shape as `[bsz, len_src, d_model]`.
        """
        try:
            x = self.tok_emb(src)
            x = self.pos_emb(x)
            x = self.drop_out(x)
        except EmbeddingError as e:
            raise EmbeddingError(
                f"Error in embedding layer of encoder") from e

        self.mha_attns: List[Tensor] = []
        for enc_layer in self.enc_layers:
            x, mha_attn = enc_layer(x, src_mask)
            self.mha_attns.append(mha_attn)

        return x


class Decoder(BaseModule):
    def __init__(
        self,
        trg_vocab_size: int,
        config: Optional[ModuleConfig] = None
    ):
        super().__init__(vocab_size=trg_vocab_size, config=config)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(self.config) for _ in range(self.config.n_layers)
        ])

    def make_causal_mask(self, trg: Tensor) -> Tensor:
        return make_causal_mask(trg)

    def make_merged_mask(self, trg: Tensor) -> Tensor:
        return make_merged_mask(trg, self.config.pad_idx)

    def forward(
        self,
        trg: Tensor,
        enc_outputs: Tensor,
        src_mask: Tensor
    ) -> Tensor:
        """
        Forward pass for the decoder.

        Parameters
        ----------
            trg : Tensor
                The indexing tensor of target sequence, shape as `[bsz, len_trg]`.

            enc_outputs : Tensor
                The output tensor of the encoder, shape as `[bsz, len_src, d_model]`.

            src_mask : Tensor
                The source mask, shape as `[bsz, 1, 1, len_src]`.

        Returns
        -------
            Tensor
                The output tensor of the decoder, shape as `[bsz, len_trg, d_model]`.
        """
        trg_mask = self.make_merged_mask(trg)

        try:
            x = self.tok_emb(trg)
            x = self.pos_emb(x)
            x = self.drop_out(x)
        except EmbeddingError as e:
            raise EmbeddingError(
                f"Error in embedding layer of decoder") from e

        self.masked_mha_attns: List[Tensor] = []
        self.mha_attns: List[Tensor] = []
        for dec_layer in self.dec_layers:
            x, masked_mha_attn, mha_attn = dec_layer(
                x,
                trg_mask=trg_mask,
                enc_outputs=enc_outputs,
                src_mask=src_mask
            )
            self.masked_mha_attns.append(masked_mha_attn)
            self.mha_attns.append(mha_attn)

        return x


class DecoderOnly(BaseModule):
    def __init__(
        self,
        trg_vocab_size: int,
        config: Optional[ModuleConfig] = None
    ):
        super().__init__(vocab_size=trg_vocab_size, config=config)

        self.dec_layers = nn.ModuleList([
            DecoderOnlyLayer(self.config) for _ in range(self.config.n_layers)
        ])

    def make_causal_mask(self, trg: Tensor) -> Tensor:
        return make_causal_mask(trg)

    def make_merged_mask(self, trg: Tensor) -> Tensor:
        return make_merged_mask(trg, self.config.pad_idx)

    def forward(self, trg: Tensor) -> Tensor:
        """
        Forward pass for the decoder-only model.

        Parameters
        ----------
            trg : Tensor
                The indexing tensor of target sequence, shape as `[bsz, len_trg]`.

        Returns
        -------
            Tensor
                The output tensor of the decoder-only model, shape as `[bsz, len_trg, d_model]`.
        """
        trg_mask = self.make_merged_mask(trg)

        try:
            x = self.tok_emb(trg)
            x = self.pos_emb(x)
            x = self.drop_out(x)
        except EmbeddingError as e:
            raise EmbeddingError(
                f"Error in embedding layer of decoder") from e

        self.masked_mha_attns: List[Tensor] = []
        self.mha_attns: List[Tensor] = []
        for dec_layer in self.dec_layers:
            x, masked_mha_attn, mha_attn = dec_layer(x, trg_mask=trg_mask)
            self.masked_mha_attns.append(masked_mha_attn)
            self.mha_attns.append(mha_attn)

        return x
