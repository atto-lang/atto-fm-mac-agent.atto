"""Deterministic, reversible character tokenizer and batch sampling."""
import json
from pathlib import Path
import numpy as np


class CharTokenizer:
    def __init__(self, chars):
        self.chars = tuple(chars)
        if len(self.chars) != len(set(self.chars)) or not self.chars:
            raise ValueError("vocabulary must be nonempty with unique characters")
        self.indices = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def fit(cls, text):
        if not text:
            raise ValueError("empty corpus")
        return cls(sorted(set(text)))

    def encode(self, text):
        try:
            return [self.indices[ch] for ch in text]
        except KeyError as exc:
            raise ValueError("character absent from saved vocabulary") from exc

    def decode(self, indices):
        return "".join(self.chars[int(i)] for i in indices)

    def save(self, path):
        Path(path).write_text(json.dumps(list(self.chars), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def split_text(text, fraction=0.9):
    if not (0 < fraction < 1):
        raise ValueError("fraction must be between 0 and 1")
    cut = int(len(text) * fraction)
    if cut == 0 or cut == len(text):
        raise ValueError("corpus too short to split")
    return text[:cut], text[cut:]


def batch(encoded, context, batch_size, rng):
    if context < 2 or batch_size < 1 or len(encoded) <= context:
        raise ValueError("corpus must contain more characters than context")
    start = rng.integers(0, len(encoded) - context, size=batch_size)
    windows = np.stack([encoded[i:i + context + 1] for i in start])
    return windows[:, :-1], windows[:, 1:]
