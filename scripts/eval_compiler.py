#!/usr/bin/env python3
"""Evaluate generated Oreslang source with an explicitly provided local compiler.

This does NOT run compiled programs or invoke an LLM; it is a compiler acceptance
harness over existing local .ores files. Execute untrusted sources in an OS
sandbox. No shell and no network calls from the harness.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evaluate(compiler: Path, source_dir: Path, args: list[str],
             timeout: int = 30, max_files: int = 500) -> dict:
    if not compiler.is_absolute() or not compiler.is_file() or compiler.is_symlink():
        raise ValueError("compiler must be a real absolute local executable path")
    if not os.access(compiler, os.X_OK):
        raise ValueError("compiler must be executable")
    if not source_dir.is_absolute() or not source_dir.is_dir() or source_dir.is_symlink():
        raise ValueError("source directory must be real and absolute")
    if args.count("{source}") != 1:
        raise ValueError("compiler argv must contain exactly one {source} placeholder")
    if timeout < 1 or timeout > 120 or max_files < 1 or max_files > 5000:
        raise ValueError("timeout/max-files outside configured bounds")
    files = sorted(source_dir.glob("*.ores"))
    if not files or len(files) > max_files:
        raise ValueError("source set must be nonempty and below max-files")
    rows = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError("source candidate cannot be a symlink or non-file")
        argv = [str(compiler)] + [str(path) if value == "{source}" else value for value in args]
        try:
            result = subprocess.run(argv, shell=False, capture_output=True,
                                    text=True, timeout=timeout, check=False,
                                    cwd=source_dir)
            status = "accepted" if result.returncode == 0 else "rejected"
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
            exit_code = None
        rows.append({"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "status": status, "exit_code": exit_code})
    count = sum(r["status"] == "accepted" for r in rows)
    return {"schema_version": 1, "total": len(rows), "accepted": count,
            "acceptance_rate": count / len(rows), "results": rows,
            "warning": "Compilation-only; compiler may execute hooks or external tools. Sandboxing is not supplied."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compiler", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--compiler-arg", action="append", required=True,
                   help="One argv element per flag; include {source} exactly once")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--max-files", type=int, default=500)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "evaluation" / "compiler.json")
    cli = p.parse_args()
    out = cli.out.resolve()
    sandbox = (ROOT / "artifacts").resolve()
    if sandbox not in out.parents or cli.out.is_symlink():
        p.error("report must remain in gitignored artifacts/")
    result = evaluate(cli.compiler, cli.source_dir, cli.compiler_arg,
                      cli.timeout, cli.max_files)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
