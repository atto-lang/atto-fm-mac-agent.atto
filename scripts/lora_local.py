#!/usr/bin/env python3
"""Offline-only MLX LM LoRA/QLoRA command wrapper with dataset integrity gates.

Requires a previously downloaded, license-reviewed, locally stored base model.
Dry-run is intentionally usable on Linux for CLI/integrity tests.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checked_dataset(path: Path) -> dict:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("dataset must be a real local directory")
    metadata = path / "dataset-manifest.json"
    if not metadata.is_file() or metadata.is_symlink():
        raise ValueError("missing dataset-manifest.json; use build_corpus.py")
    report = json.loads(metadata.read_text(encoding="utf-8"))
    if report.get("version") != 1 or report.get("format") != "mlx-lm:text-jsonl":
        raise ValueError("unsupported dataset manifest")
    groups = {}
    for source in report.get("sources", []):
        if source.get("split") not in ("train", "valid", "test"):
            raise ValueError("invalid source split")
        if source.get("id") in groups:
            raise ValueError("duplicate source group")
        groups[source.get("id")] = source["split"]
    if len(groups) < 3 or len(set(groups.values())) != 3:
        raise ValueError("repo-disjoint train, valid, test groups required")
    for split in ("train", "valid", "test"):
        f = path / f"{split}.jsonl"
        if f.is_symlink() or not f.is_file():
            raise ValueError(f"missing {split}.jsonl")
        raw = f.read_bytes()
        expected = report.get("splits", {}).get(split, {})
        if hashlib.sha256(raw).hexdigest() != expected.get("sha256"):
            raise ValueError(f"dataset hash mismatch: {split}")
        lines = raw.decode("utf-8").splitlines()
        if not lines or len(lines) != expected.get("count"):
            raise ValueError(f"dataset count mismatch: {split}")
        for line in lines:
            record = json.loads(line)
            if set(record) != {"text"} or not isinstance(record["text"], str) or not record["text"].strip():
                raise ValueError("non-text or empty dataset example")
    return report


def command(args) -> list[str]:
    if not args.model.is_absolute() or args.model.is_symlink() or not args.model.is_dir():
        raise ValueError("--model must be an existing local absolute directory; no model downloads")
    if not (args.model / "config.json").is_file():
        raise ValueError("missing local base model config.json")
    if not args.confirm_model_rights:
        raise ValueError("provide --confirm-model-rights after checking base model license and permitted use")
    checked_dataset(args.data)
    artifact_root = (ROOT / "artifacts").resolve()
    adapter = args.adapter.resolve()
    if artifact_root not in adapter.parents:
        raise ValueError("adapter must reside within gitignored artifacts/")
    if args.adapter.is_symlink():
        raise ValueError("adapter may not be a symlink")
    if args.iters < 1 or args.batch_size < 1 or args.max_seq_length < 16 or args.num_layers < 1:
        raise ValueError("invalid training configuration")
    if not (0 < args.lr <= 0.01):
        raise ValueError("invalid learning rate")
    cmd = [sys.executable, "-m", "mlx_lm", "lora",
           "--model", str(args.model.resolve()),
           "--data", str(args.data.resolve()),
           "--adapter-path", str(adapter),
           "--fine-tune-type", "lora"]
    if args.mode == "train":
        cmd += ["--train", "--iters", str(args.iters),
                "--batch-size", str(args.batch_size),
                "--max-seq-length", str(args.max_seq_length),
                "--learning-rate", str(args.lr),
                "--num-layers", str(args.num_layers), "--seed", str(args.seed),
                "--val-batches", "-1"]
    else:
        if not (adapter / "adapters.safetensors").is_file():
            raise ValueError("test requires saved local adapters.safetensors")
        cmd += ["--test", "--test-batches", "-1"]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True,
                   help="Absolute path to locally staged, licensed base model")
    p.add_argument("--data", type=Path, default=ROOT / "artifacts" / "finetune" / "data")
    p.add_argument("--adapter", type=Path, default=ROOT / "artifacts" / "finetune" / "adapter")
    p.add_argument("--mode", choices=("train", "test"), default="train")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confirm-model-rights", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    argv = command(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "command": argv,
                          "network": "disabled during execution"}, indent=2))
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        p.error("Metal training/evaluation requires native Apple Silicon macOS")
    import mlx.core as mx
    if not mx.metal.is_available():
        p.error("Metal GPU unavailable, refusing CPU fallback")
    # Dependencies must already be installed and the model staged before this call.
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled",
                "TOKENIZERS_PARALLELISM": "false"})
    args.adapter.mkdir(parents=True, exist_ok=True)
    subprocess.run(argv, check=True, env=env, cwd=ROOT)


if __name__ == "__main__":
    main()
