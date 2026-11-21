"""A small causal Transformer language model built with Apple MLX."""
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context: int = 128
    dim: int = 128
    heads: int = 4
    layers: int = 2

    def validate(self):
        if self.vocab_size < 2 or self.context < 2 or self.dim < 1:
            raise ValueError("invalid model dimensions")
        if self.heads < 1 or self.layers < 1 or self.dim % self.heads:
            raise ValueError("heads must divide dim and layers must be positive")


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.position = nn.Embedding(cfg.context, cfg.dim)
        self.blocks = nn.TransformerEncoder(
            cfg.layers, cfg.dim, cfg.heads, norm_first=True
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size)

    def __call__(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.shape[1] > self.cfg.context:
            raise ValueError("expected [batch, 1..context] token matrix")
        length = tokens.shape[1]
        x = self.embedding(tokens) + self.position(mx.arange(length))
        x = self.blocks(x, nn.MultiHeadAttention.create_additive_causal_mask(length))
        return self.head(self.norm(x))
