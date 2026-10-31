#!/usr/bin/env python3
"""Local GPU/MLX causal language model training; no cloud calls."""
import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ores_llm.data import CharTokenizer, batch, split_text
from ores_llm.model import ModelConfig, TinyLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error("steps, batch-size and lr must be positive")
    raw = args.text.read_text(encoding="utf-8")
    train_text, valid_text = split_text(raw)
    tok = CharTokenizer.fit(train_text)
    # Unseen validation characters are a dataset error, never silently replaced.
    train = np.array(tok.encode(train_text), dtype=np.int32)
    valid = np.array(tok.encode(valid_text), dtype=np.int32)
    if len(train) <= args.context or len(valid) <= args.context:
        p.error("both train and validation splits must exceed context; use larger corpus or shorter context")
    cfg = ModelConfig(len(tok.chars), args.context, args.dim, args.heads, args.layers)
    cfg.validate()
    mx.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = TinyLM(cfg)
    optimizer = optim.AdamW(learning_rate=args.lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    args.out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for step in range(1, args.steps + 1):
        x, y = batch(train, cfg.context, args.batch_size, rng)
        loss, grads = loss_and_grad(model, mx.array(x), mx.array(y))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 1 or step % 25 == 0 or step == args.steps:
            vx, vy = batch(valid, cfg.context, min(4, args.batch_size), rng)
            model.eval()
            vloss = float(loss_fn(model, mx.array(vx), mx.array(vy)).item())
            model.train()
            print(json.dumps({"step": step, "train_loss": float(loss.item()), "valid_loss": vloss}), flush=True)
            if vloss < best:
                best = vloss
                # Store config and vocabulary alongside weights, and only load locally trusted checkpoints.
                mx.save_safetensors(str(args.out / "weights.safetensors"), dict(model.parameters().items()) if False else
                                    {key: value for key, value in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters())})
                tok.save(args.out / "vocab.json")
                (args.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(json.dumps({"best_valid_loss": best, "checkpoint": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
