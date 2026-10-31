#!/usr/bin/env python3
"""One-command, local-only Apple Silicon GPU acceptance test.

Does not connect to any service. Exit nonzero if tests, training, checkpoint,
generation, or GPU admission fail; never report success for a CPU fallback.
"""
import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]


def gpu_info():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Apple Silicon macOS (arm64) is required")
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU unavailable; CPU results are not GPU evidence")
    mx.set_default_device(mx.gpu)
    mx.eval(mx.ones((64, 64)) @ mx.ones((64, 64)))
    hardware = subprocess.run(
        ["system_profiler", "SPDisplaysDataType"], check=True,
        text=True, capture_output=True, timeout=45
    ).stdout
    chipset = re.search(r"Chipset Model:\s*(Apple M[^\n]+)", hardware)
    gpu_cores = re.search(r"Total Number of Cores:\s*(\d+)", hardware)
    unified_bytes = int(subprocess.check_output(
        ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
    ).strip())
    return {
        "chipset": chipset.group(1).strip() if chipset else None,
        "gpu_cores": int(gpu_cores.group(1)) if gpu_cores else None,
        "unified_memory_bytes": unified_bytes,
        "platform": platform.platform(),
        "device": "gpu",
    }


def run(*command):
    print("$ " + " ".join(map(str, command)), flush=True)
    started = time.perf_counter()
    output = subprocess.run(command, cwd=ROOT, check=False, text=True,
                            capture_output=True, timeout=900)
    print(output.stdout, end="", flush=True)
    if output.stderr:
        print(output.stderr, file=sys.stderr, end="", flush=True)
    if output.returncode != 0:
        raise RuntimeError(f"command failed (exit {output.returncode}): {command!r}")
    return {"elapsed_seconds": time.perf_counter() - started,
            "stdout": output.stdout}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-tag", choices=("m4-max", "m5-pro"), required=True)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "benchmarks")
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    hw = gpu_info()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5
    ).strip()
    tracked_changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True, timeout=5
    ).strip()
    if tracked_changes:
        raise RuntimeError("tracked source files have uncommitted changes: " + tracked_changes)
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise RuntimeError("failed to identify full Git commit SHA")
    expected = {"m4-max": ("M4 Max", 40), "m5-pro": ("M5 Pro", 20)}
    chip, cores = expected[args.device_tag]
    if hw["chipset"] != f"Apple {chip}" or hw["gpu_cores"] != cores:
        raise RuntimeError(f"expected {chip} / {cores} GPU cores, got {hw['chipset']} / {hw['gpu_cores']}")
    out = (args.out / args.device_tag).resolve()
    model_dir = out / "model"
    out.mkdir(parents=True, exist_ok=True)
    unit = run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")
    trained = run(
        sys.executable, "scripts/train.py",
        "--text", "data/sample.txt", "--out", str(model_dir),
        "--context", "16", "--steps", str(args.steps),
        "--batch-size", "8", "--device", "gpu"
    )
    evidence = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "completed" or evidence.get("device") != "gpu":
        raise RuntimeError("training did not complete on Metal GPU")
    for name in ("weights.safetensors", "config.json", "vocab.json", "metrics.jsonl"):
        if not (model_dir / name).is_file():
            raise RuntimeError(f"missing training artifact: {name}")
    generated = run(
        sys.executable, "scripts/generate.py", "--checkpoint", str(model_dir),
        "--prompt", "Oreslang", "--max-new-tokens", "16"
    )
    if not generated["stdout"].strip().startswith("Oreslang") or len(generated["stdout"].strip()) < 9:
        raise RuntimeError("generation did not preserve its prompt")
    report = {
        "status": "gpu_smoke_pass",
        "hardware": hw,
        "source_commit": source_commit,
        "tracked_source_clean": True,
        "model": evidence,
        "unit_test_seconds": unit["elapsed_seconds"],
        "end_to_end_train_seconds": trained["elapsed_seconds"],
        "generation_seconds": generated["elapsed_seconds"],
        "generated_characters": len(generated["stdout"].strip()),
    }
    path = out / "gpu-smoke-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: {path}")


if __name__ == "__main__":
    main()
