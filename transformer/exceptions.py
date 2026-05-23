
class TransformerError(Exception):
    """Base class for exceptions in this module."""


class EncoderError(TransformerError):
    """Exception raised for errors in the encoder."""


class DecoderError(TransformerError):
    """Exception raised for errors in the decoder."""


class MultiHeadAttentionError(TransformerError):
    """Exception raised for errors in the multi-head attention."""


class ScaledDotProductAttentionError(MultiHeadAttentionError):
    """Exception raised for errors in the scaled dot-product attention."""


class EmbeddingError(TransformerError):
    """Exception raised for errors in the embedding layer."""


class GenerationError(TransformerError):
    """Exception raised for errors during generation."""
