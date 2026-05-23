
from torch import nn
from typing import Tuple

from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .layer_norm import LayerNorm
from .._utils import ModuleConfig

__all__ = [
    "MultiHeadAttention",
    "PositionwiseFeedForward",
    "LayerNorm",
]


def make_mha_layers(config: ModuleConfig) -> Tuple[MultiHeadAttention, LayerNorm, nn.Dropout]:
    """
    Create multi-head attention layers.
    """
    mha = MultiHeadAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_qk=config.d_qk,
        d_v=config.d_v
    )
    mha_norm = LayerNorm(d_model=config.d_model)
    mha_dropout = nn.Dropout(p=config.drop_prob)

    return mha, mha_norm, mha_dropout


def make_ff_layers(config: ModuleConfig) -> Tuple[PositionwiseFeedForward, LayerNorm, nn.Dropout]:
    """
    Create positionwise feed-forward layers.
    """
    ff = PositionwiseFeedForward(
        d_model=config.d_model,
        d_ff=config.d_ff,
        drop_prob=config.drop_prob
    )
    ff_norm = LayerNorm(d_model=config.d_model)
    ff_dropout = nn.Dropout(p=config.drop_prob)

    return ff, ff_norm, ff_dropout
