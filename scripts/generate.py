#!/usr/bin/env python3
"""Autoregressive inference from an owned, locally trained TinyLM checkpoint."""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ores_llm.data import CharTokenizer
from ores_llm.model import ModelConfig, TinyLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.prompt or args.max_new_tokens < 0 or args.temperature <= 0:
        p.error("provide nonempty prompt, nonnegative tokens and positive temperature")
    cfg = ModelConfig(**json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8")))
    tok = CharTokenizer.load(args.checkpoint / "vocab.json")
    if cfg.vocab_size != len(tok.chars):
        p.error("checkpoint vocabulary mismatch")
    model = TinyLM(cfg)
    model.load_weights(str(args.checkpoint / "weights.safetensors"))
    model.eval()
    mx.random.seed(args.seed)
    ids = tok.encode(args.prompt)
    for _ in range(args.max_new_tokens):
        tokens = mx.array([ids[-cfg.context:]], dtype=mx.int32)
        logits = model(tokens)[0, -1] / args.temperature
        index = int(mx.random.categorical(logits).item())
        ids.append(index)
    print(tok.decode(ids))


if __name__ == "__main__":
    main()
