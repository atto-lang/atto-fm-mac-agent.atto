#!/usr/bin/env python3
"""Optional local Apple Foundation Models CLI adapter. Never trains Apple FM."""
import argparse
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description="Use installed macOS 27 Apple fm CLI for comparison")
    p.add_argument("prompt")
    args = p.parse_args()
    if not shutil.which("fm"):
        p.error("fm executable not found; macOS 27 and Apple Foundation Models setup required")
    # No shell=True, no network, no injection of untrusted flags.
    result = subprocess.run(["fm", "respond", args.prompt], check=False, timeout=120)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
