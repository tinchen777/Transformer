# -*- coding: utf-8 -*-
# Python version: 3.10
from __future__ import annotations
import torch
from torch import nn
from typing import (Literal, Optional, TYPE_CHECKING)

from ._utils import (ModuleConfig, select_next_token)
from .modules import (
    Encoder,
    Decoder,
    DecoderOnly
)
from .exceptions import GenerationError

if TYPE_CHECKING:
    Tensor = torch.Tensor


class TransformerBase(nn.Module):
    """
    Base class for transformer models.
    """
    @property
    def device(self):
        return next(self.parameters()).device


class Transformer(TransformerBase):
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

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_len: int = -1,
        strategy: Literal["greedy", "sample"] = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
     ):
        """
        Generate a sequence auto-regressively given a prompt.

        Parameters
        ----------
            prompt: Tensor
                The indexing tensor of the prompt sequence, shape as `[1, len_prompt]` or `[len_prompt]`.

            max_len: int, default `-1`
                The maximum length of the generated sequence.
                - `-1`: Generate until EOS token is encountered.

            strategy: Literal["greedy", "sample"], default `"greedy"`
                The decoding strategy to use.
                - `"greedy"`: always pick the highest-probability token.
                - `"sample"`: draw from the distribution (supports temperature, top-k, and top-p filtering).

            temperature: float, default `1.0`
                :param:`strategy` is `"sample"` ONLY. Scaling factor applied before softmax.
                - `< 1`: Sharpen the distribution;
                - `> 1`: Flatten it.
                Approaches greedy as temperature → 0.

            top_k: Optional[int], default `None`
                :param:`strategy` is `"sample"` ONLY.
                - _int_: Restrict sampling to the top-k tokens by logit value. Must be > 0.

            top_p: Optional[float], default `None`
                :param:`strategy` is `"sample"` ONLY.
                - _float_: Restrict sampling to the top-p tokens by cumulative probability. Must be in (0, 1).

        Yields
        -------
            int
                The index of the next token in the generated sequence.
        """
        try:
            if prompt.dim() == 1:
                prompt = prompt.unsqueeze(0)  # [1, len_prompt]
            elif prompt.dim() != 2:
                raise ValueError(
                    f"Expected prompt of shape [len_prompt] or [bsz, len_prompt], got {prompt.shape}.")
            if prompt.size(1) == 0:
                raise ValueError("Prompt sequence length must be greater than 0.")
            prompt = prompt.to(self.device)

            # ========== 1. Encoder==========
            src_mask = self.encoder.make_pad_mask(prompt)  # [1, 1, 1, len_prompt]
            enc_outputs = self.encoder(prompt, src_mask=src_mask)  # [1, len_prompt, d_model]

            # ========== 2. Decoder ==========
            ys = torch.full((1, 1), self.decoder.config.bos_idx, dtype=torch.long, device=self.device)

            while True:
                logits = self.linear(self.decoder(ys, enc_outputs, src_mask=src_mask))  # [1, t, trg_vocab_size]

                next_token = select_next_token(
                    logits=logits[:, -1, :],  # [1, trg_vocab_size]
                    strategy=strategy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                # next_token: [1, 1]

                next_token_item = int(next_token.item())
                if next_token_item == self.decoder.config.eos_idx:
                    return
                if max_len > 0:
                    max_len -= 1

                ys = torch.cat([ys, next_token], dim=1)

                yield next_token_item

        except Exception as e:
            raise GenerationError("Error during Transformer generation.") from e

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

    @property
    def device(self):
        return next(self.parameters()).device


class DecoderOnlyTransformer(TransformerBase):
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

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_new_tokens: int = 500,
        strategy: Literal["greedy", "sample"] = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
     ) -> Tensor:
        """
        Generate a sequence auto-regressively given a prompt.

        Parameters
        ----------
            prompt: Tensor
                The indexing tensor of the prompt sequence, shape as `[1, len_prompt]` or `[len_prompt]`.

            max_new_tokens: int, default `500`
                The maximum number of new tokens to generate.

            strategy: Literal["greedy", "sample"], default `"greedy"`
                The decoding strategy to use.
                - `"greedy"`: always pick the highest-probability token.
                - `"sample"`: draw from the distribution (supports temperature, top-k, and top-p filtering).

            temperature: float, default `1.0`
                :param:`strategy` is `"sample"` ONLY. Scaling factor applied before softmax.
                - `< 1`: Sharpen the distribution;
                - `> 1`: Flatten it.
                Approaches greedy as temperature → 0.

            top_k: Optional[int], default `None`
                :param:`strategy` is `"sample"` ONLY.
                - _int_: Restrict sampling to the top-k tokens by logit value. Must be > 0.

            top_p: Optional[float], default `None`
                :param:`strategy` is `"sample"` ONLY.
                - _float_: Restrict sampling to the top-p tokens by cumulative probability. Must be in (0, 1).

        Returns
        -------
            Tensor
                The indexing tensor of the generated sequence, shape as `[1, len_generated]`.
        """
        try:
            if prompt.dim() == 1:
                prompt = prompt.unsqueeze(0)  # [1, len_prompt]
            elif prompt.dim() != 2:
                raise ValueError(
                    f"Expected prompt of shape [len_prompt] or [bsz, len_prompt], got {prompt.shape}.")
            ys = prompt.to(self.device)
            max_new_tokens = max(0, max_new_tokens)

            # ========== Decoder ==========
            # ys_cond = ys if ys.size(1) <= self.max_len else ys[:, -self.max_len:]

            for _ in range(max_new_tokens):
                logits = self.decoder(ys)  # [1, t, trg_vocab_size]
                next_token = select_next_token(
                    logits=logits[:, -1, :],  # [1, trg_vocab_size]
                    strategy=strategy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                ys = torch.cat([ys, next_token], dim=1)

                if next_token.item() == self.decoder.config.eos_idx:
                    break

            return ys

        except Exception as e:
            raise GenerationError("Error during Decoder Only Transformer generation.") from e

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

    @property
    def device(self):
        return next(self.parameters()).device
