#!/usr/bin/env python3
"""Fail-closed comparison of two real Mac Metal training smoke reports.

Never constructs benchmark results: input reports must come from smoke_mac.py.
No MLX dependency, no network access.
"""
import argparse
import json
import math
from pathlib import Path


HARDWARE = {"m4-max": ("Apple M4 Max", 40), "m5-pro": ("Apple M5 Pro", 20)}
COMPARABLE = ("corpus_sha256", "seed", "train_chars", "valid_chars",
              "valid_scored_targets", "steps", "batch_size", "mlx_version", "model")


def load_report(path, expected_tag):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "gpu_smoke_pass":
        raise ValueError(f"{expected_tag}: GPU acceptance did not pass")
    source_commit = data.get("source_commit")
    if (not isinstance(source_commit, str) or len(source_commit) != 40 or
            any(ch not in "0123456789abcdef" for ch in source_commit)):
        raise ValueError(f"{expected_tag}: missing/invalid source commit")
    if data.get("tracked_source_clean") is not True:
        raise ValueError(f"{expected_tag}: dirty source checkout")
    hw = data.get("hardware")
    model = data.get("model")
    if not isinstance(hw, dict) or not isinstance(model, dict):
        raise ValueError(f"{expected_tag}: missing hardware or model report")
    expected_chip, expected_cores = HARDWARE[expected_tag]
    if (hw.get("chipset"), hw.get("gpu_cores"), hw.get("device")) != (
        expected_chip, expected_cores, "gpu"
    ):
        raise ValueError(f"{expected_tag}: reported machine does not match hardware tag")
    if model.get("status") != "completed" or model.get("device") != "gpu":
        raise ValueError(f"{expected_tag}: incomplete/non-Metal model report")
    if model.get("machine") != "arm64":
        raise ValueError(f"{expected_tag}: report is not from Apple Silicon arm64")
    if not isinstance(model.get("corpus_sha256"), str) or len(model["corpus_sha256"]) != 64:
        raise ValueError(f"{expected_tag}: missing/invalid corpus hash")
    if not all(ch in "0123456789abcdef" for ch in model["corpus_sha256"]):
        raise ValueError(f"{expected_tag}: invalid SHA-256")
    for field in ("steps", "batch_size", "seed", "train_chars",
                  "valid_chars", "valid_scored_targets"):
        if not isinstance(model.get(field), int) or isinstance(model[field], bool):
            raise ValueError(f"{expected_tag}: invalid {field}")
    if model["steps"] <= 0 or model["batch_size"] <= 0:
        raise ValueError(f"{expected_tag}: invalid training parameters")
    for field in ("gpu_training_tokens_per_second", "best_valid_loss"):
        value = model.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{expected_tag}: invalid {field}")
        if not math.isfinite(value) or (field.startswith("gpu_") and value <= 0):
            raise ValueError(f"{expected_tag}: non-finite/non-positive {field}")
    return data


def compare(m4, m5):
    first = load_report(m4, "m4-max")
    second = load_report(m5, "m5-pro")
    if first["source_commit"] != second["source_commit"]:
        raise ValueError("comparison invalid: different Git source commits")
    a, b = first["model"], second["model"]
    differences = [k for k in COMPARABLE if a.get(k) != b.get(k)]
    if differences:
        raise ValueError("comparison invalid: mismatched " + ", ".join(differences))
    at = a["gpu_training_tokens_per_second"]
    bt = b["gpu_training_tokens_per_second"]
    return {
        "status": "comparable_smoke_reports",
        "m4_max_training_tokens_per_second": at,
        "m5_pro_training_tokens_per_second": bt,
        "m4_max_over_m5_pro_ratio": at / bt,
        "m4_max_best_valid_loss": a["best_valid_loss"],
        "m5_pro_best_valid_loss": b["best_valid_loss"],
        "dataset_sha256": a["corpus_sha256"],
        "source_commit": first["source_commit"],
        "mlx_version": a["mlx_version"],
        "model": a["model"],
        "steps": a["steps"],
        "seed": a["seed"],
        "caveat": "Smoke-only; not representative model quality or stable performance benchmarking."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m4", type=Path, required=True)
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = compare(args.m4, args.m5)
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
