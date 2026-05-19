# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
import torch
from torch import nn
from typing import (Any, List, TYPE_CHECKING)

from .embedding import TransformerEmbedding
from .blocks import (EncoderLayer, DecoderLayer)

if TYPE_CHECKING:
    from .types import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        enc_vocab_size: int,
        n_layers: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        max_len: int = 5000,
        drop_prob: float = 0.1,
        device: Any = "cpu"
    ):
        super().__init__()

        self.enc_emb = TransformerEmbedding(
            vocab_size=enc_vocab_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            device=device
        )

        self.enc_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        x = self.enc_emb(x)

        self.enc_mha_attns: List[Tensor] = []
        for enc_layer in self.enc_layers:
            x, mha_attn = enc_layer(x, src_mask)
            self.enc_mha_attns.append(mha_attn)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dec_vocab_size: int,
        n_layers: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        max_len: int = 5000,
        drop_prob: float = 0.1,
        device: Any = "cpu"
    ):
        super().__init__()

        self.dec_emb = TransformerEmbedding(
            vocab_size=dec_vocab_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            device=device
        )

        self.dec_layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

        self.linear = nn.Linear(d_model, dec_vocab_size)

    def forward(self, x: Tensor, enc_outputs: Tensor, trg_mask: Tensor, src_mask: Tensor) -> Tensor:
        x = self.dec_emb(x)

        self.dec_masked_mha_attns: List[Tensor] = []
        self.dec_mha_attns: List[Tensor] = []
        for dec_layer in self.dec_layers:
            x, masked_mha_attn, mha_attn = dec_layer(x, enc_outputs, trg_mask, src_mask)
            self.dec_masked_mha_attns.append(masked_mha_attn)
            self.dec_mha_attns.append(mha_attn)

        # pass to LM head
        outputs = self.linear(x)

        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx: int,
        trg_pad_idx: int,
        trg_sos_idx: int,
        enc_vocab_size: int,
        dec_vocab_size: int,
        enc_n_layers: int = 6,
        dec_n_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        max_len: int = 5000,
        d_ff: int = 2048,
        drop_prob: float = 0.1,
        device: Any = "cpu"
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(
            enc_vocab_size=enc_vocab_size,
            n_layers=enc_n_layers,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            max_len=max_len,
            drop_prob=drop_prob,
            device=device
        )

        self.decoder = Decoder(
            dec_vocab_size=dec_vocab_size,
            n_layers=dec_n_layers,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            max_len=max_len,
            drop_prob=drop_prob,
            device=device
        )

    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_outputs = self.encoder(src, src_mask=src_mask)
        outputs = self.decoder(trg, enc_outputs, trg_mask=trg_mask, src_mask=src_mask)

        self.enc_mha_attns = self.encoder.enc_mha_attns
        self.dec_masked_mha_attns = self.decoder.dec_masked_mha_attns
        self.dec_mha_attns = self.decoder.dec_mha_attns

        return outputs

    def make_src_mask(self, src: Tensor) -> Tensor:
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg: Tensor) -> Tensor:
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).byte().to(self.device)

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask
