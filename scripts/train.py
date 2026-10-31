#!/usr/bin/env python3
"""Train an owned TinyLM using a local MLX device and write auditable metrics."""
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ores_llm.data import CharTokenizer, batch, split_text
from ores_llm.model import ModelConfig, TinyLM


def evaluate_all(model, encoded, context):
    """Deterministically score *every* held-out next-token target, once.

    Each non-overlapping slice starts a fresh context. Unlike random validation
    windows, evaluation includes the tail and never leaks training tokens.
    """
    if len(encoded) < 2:
        raise ValueError("validation must contain at least two characters")
    total_loss = 0.0
    total_targets = 0
    model.eval()
    for start in range(0, len(encoded) - 1, context):
        stop = min(start + context + 1, len(encoded))
        window = encoded[start:stop]
        x = mx.array(window[:-1][None, :], dtype=mx.int32)
        y = mx.array(window[1:][None, :], dtype=mx.int32)
        loss = nn.losses.cross_entropy(model(x), y, reduction="mean")
        length = len(window) - 1
        total_loss += float(loss.item()) * length
        total_targets += length
    model.train()
    return total_loss / total_targets, total_targets


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
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.eval_every < 1:
        p.error("steps, batch-size, lr and eval-every must be positive")
    if args.device == "gpu":
        if not mx.metal.is_available():
            p.error("Metal GPU not available. Use a supported Apple Silicon Mac; no silent CPU fallback.")
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    raw_bytes = args.text.read_bytes()
    raw = raw_bytes.decode("utf-8")
    train_text, valid_text = split_text(raw)
    tok = CharTokenizer.fit(train_text)
    # Fail closed: unrecognized validation characters indicate dataset/tokenizer mismatch.
    try:
        train = np.array(tok.encode(train_text), dtype=np.int32)
        valid = np.array(tok.encode(valid_text), dtype=np.int32)
    except ValueError as exc:
        p.error(str(exc) + "; revise the corpus or use a tokenizer with byte fallback")
    if len(train) <= args.context or len(valid) <= args.context:
        p.error("both splits must exceed context; supply more data or reduce --context")

    cfg = ModelConfig(len(tok.chars), args.context, args.dim, args.heads, args.layers)
    cfg.validate()
    mx.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = TinyLM(cfg)
    optimizer = optim.AdamW(learning_rate=args.lr)

    def loss_fn(model, x, y):
        return nn.losses.cross_entropy(model(x), y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    args.out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    train_elapsed = 0.0
    last_train_loss = None
    log_path = args.out / "metrics.jsonl"
    # An existing output directory is allowed, but the report always represents
    # this run; old metrics must never be mixed with new metrics.
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            x, y = batch(train, cfg.context, args.batch_size, rng)
            start = time.perf_counter()
            loss, grads = loss_and_grad(model, mx.array(x), mx.array(y))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            train_elapsed += time.perf_counter() - start
            last_train_loss = float(loss.item())
            if not np.isfinite(last_train_loss):
                raise RuntimeError(f"non-finite training loss at step {step}")
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                valid_loss, targets = evaluate_all(model, valid, cfg.context)
                if not np.isfinite(valid_loss):
                    raise RuntimeError(f"non-finite validation loss at step {step}")
                row = {
                    "step": step, "train_loss": last_train_loss,
                    "valid_loss": valid_loss, "valid_targets": targets,
                    "train_tokens_per_second": (
                        step * args.batch_size * args.context / train_elapsed
                    ),
                }
                line = json.dumps(row, sort_keys=True)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                if valid_loss < best:
                    best = valid_loss
                    model.save_weights(str(args.out / "weights.safetensors"))
                    tok.save(args.out / "vocab.json")
                    (args.out / "config.json").write_text(
                        json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8"
                    )

    report = {
        "status": "completed",
        "device": args.device,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_version": getattr(__import__("mlx"), "__version__", "unknown"),
        "corpus_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "seed": args.seed,
        "train_chars": len(train),
        "valid_chars": len(valid),
        "valid_scored_targets": len(valid) - 1,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "model": asdict(cfg),
        "last_train_loss": last_train_loss,
        "best_valid_loss": best,
        "gpu_training_tokens_per_second": (
            args.steps * args.batch_size * args.context / train_elapsed
            if args.device == "gpu" else None
        ),
        "training_seconds_excluding_validation": train_elapsed,
        "checkpoint": str(args.out.resolve()),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
