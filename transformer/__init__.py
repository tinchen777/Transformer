
"""
Transformer
===========

This project implements a modular and extensible Transformer framework, including both standard `Encoder-Decoder` and `Decoder-Only` architectures, with built-in text generators supporting `greedy` and `sampling` decoding strategies for flexible experimentation and customization.
"""
from .models import (Transformer, DecoderOnlyTransformer)
from .modules import (Encoder, Decoder, DecoderOnly)
from ._utils import (ModuleConfig, select_next_token)

__all__ = [
    'Transformer',
    'DecoderOnlyTransformer',
    'Encoder',
    'Decoder',
    'DecoderOnly',
    'ModuleConfig',
    'select_next_token'
]
