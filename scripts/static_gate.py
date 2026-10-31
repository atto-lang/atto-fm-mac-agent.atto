#!/usr/bin/env python3
"""Dependency-free static gate; NOT evidence of GPU execution."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = sorted(
        path for dirname in ("ores_llm", "scripts", "tests")
        for path in (ROOT / dirname).rglob("*.py")
    )
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    corpus = (ROOT / "data" / "sample.txt").read_text(encoding="utf-8")
    cutoff = int(len(corpus) * .9)
    train, valid = corpus[:cutoff], corpus[cutoff:]
    unseen = sorted(set(valid) - set(train))
    if unseen:
        raise ValueError("validation has unseen characters: " + repr(unseen))
    if min(len(train), len(valid)) <= 16:
        raise ValueError("sample corpus too short for documented 16-token context")
    print(json.dumps({
        "static_gate": "pass",
        "syntax_checked_files": len(sources),
        "sample_train_chars": len(train),
        "sample_valid_chars": len(valid),
        "mlx_runtime_executed": False,
        "gpu_executed": False
    }, sort_keys=True))


if __name__ == "__main__":
    main()
