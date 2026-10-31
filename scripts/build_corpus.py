"""Permission-gated, repository-disjoint corpus generator for MLX LM text JSONL.

No network, no implicit GitHub access, no dependency on MLX. Input repositories
must already exist on local disk and be expressly approved in a manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSES = {"MIT", "Apache-2.0", "BSD-3-Clause", "CC0-1.0", "PROPRIETARY-OWNED"}
EXTENSIONS = {".ores", ".md"}
EXCLUDE_DIRS = {".git", ".venv", "node_modules", "artifacts", "dist", "build",
                "target", "vendor", "__pycache__", ".cache", ".idea"}
SECRETS = [
    re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE) PRIVATE KEY-----"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{35,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?im)^\s*(?:api_key|secret_key|password|access_token)\s*[:=]\s*['\"]?[A-Za-z0-9._+/=-]{12,}"),
]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(data: str) -> str:
    return re.sub(r"\s+", " ", data).strip()


def ensure_safe_output(out: Path) -> Path:
    sandbox = (ROOT / "artifacts").resolve()
    path = out.resolve()
    if path == sandbox or sandbox not in path.parents:
        raise ValueError("output must be within local gitignored artifacts/ directory")
    if path.exists() and path.is_symlink():
        raise ValueError("output cannot be a symlink")
    return path


def validate_sources(manifest: dict) -> list[dict]:
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("manifest version must be 1")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) < 3:
        raise ValueError("at least three independently approved repositories required")
    ids, paths = set(), set()
    out = []
    for item in sources:
        if not isinstance(item, dict):
            raise ValueError("source entry must be an object")
        source_id = item.get("id")
        raw_path = item.get("path")
        revision = item.get("revision")
        if (not isinstance(source_id, str) or
                not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", source_id)):
            raise ValueError("invalid repository ID")
        if item.get("authorized") is not True or item.get("license") not in LICENSES:
            raise ValueError("explicit rights approval and allowlisted license required")
        if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("each source needs a pinned 40-character Git commit SHA")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError("source paths must be absolute")
        original = Path(raw_path)
        if original.is_symlink() or not original.is_dir():
            raise ValueError("source must be a real directory, not a symlink")
        path = original.resolve()
        if source_id in ids or path in paths:
            raise ValueError("duplicate source ID or directory")
        ids.add(source_id)
        paths.add(path)
        # A Git ref is evidence only when the local repository is verified by the
        # caller; we avoid executing Git against untrusted directories.
        out.append({**item, "path": path})
    return out


def enumerate_files(path: Path, max_bytes: int):
    for root, dirs, names in os.walk(path, followlinks=False):
        base = Path(root)
        for d in dirs:
            if (base / d).is_symlink():
                raise ValueError("refusing symlink directory in source repository")
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("."))
        for name in sorted(names):
            file = base / name
            if file.suffix.lower() not in EXTENSIONS:
                continue
            if file.is_symlink():
                raise ValueError("refusing symlink file")
            if not file.is_file():
                continue
            if file.stat().st_size > max_bytes:
                raise ValueError("oversize candidate file; split or explicitly exclude it")
            yield file


def build(manifest: dict, output: Path, seed: int = 42, max_bytes: int = 65536) -> dict:
    sources = validate_sources(manifest)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if max_bytes < 128 or max_bytes > 1024 * 1024:
        raise ValueError("max_bytes outside accepted 128B–1MiB range")
    output = ensure_safe_output(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output is not empty; use a new directory to avoid stale datasets")
    keyed = sorted(sources, key=lambda s: sha_bytes(f"{seed}:{s['id']}".encode("utf-8")))
    split_map = {}
    for i, source in enumerate(keyed):
        split_map[source["id"]] = "valid" if i == 0 else "test" if i == 1 else "train"
    records = {name: [] for name in ("train", "valid", "test")}
    seen = {}
    summaries = []
    for source in sorted(sources, key=lambda s: s["id"]):
        root = source["path"]
        accepted = 0
        for path in enumerate_files(root, max_bytes):
            raw = path.read_bytes()
            if b"\x00" in raw:
                raise ValueError("binary content in allowed source extension")
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("source must be UTF-8") from exc
            content = content.replace("\r\n", "\n").strip()
            if len(content) < 16:
                continue
            if any(rx.search(content) for rx in SECRETS):
                raise ValueError("possible secret detected; aborting entire corpus build")
            digest = sha_bytes(normalize_text(content).encode("utf-8"))
            prior = seen.get(digest)
            if prior is not None:
                if prior != source["id"]:
                    raise ValueError("cross-repository duplicate content: train/eval leakage risk")
                continue
            seen[digest] = source["id"]
            records[split_map[source["id"]]].append({"text": content})
            accepted += 1
        summaries.append({"id": source["id"], "revision": source["revision"],
                          "license": source["license"], "split": split_map[source["id"]],
                          "accepted_files": accepted})
    if any(not entries for entries in records.values()):
        raise ValueError("all train/valid/test splits must contain text; add approved repositories")
    # Defend against symlink swaps in output location.
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, entries in records.items():
        entries.sort(key=lambda x: sha_bytes(x["text"].encode("utf-8")))
        encoded = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries).encode("utf-8")
        (output / f"{name}.jsonl").write_bytes(encoded)
        hashes[name] = {"sha256": sha_bytes(encoded), "count": len(entries)}
    result = {"version": 1, "seed": seed, "splits": hashes, "sources": summaries,
              "format": "mlx-lm:text-jsonl", "near_duplicate_audit": "not_performed"}
    (output / "dataset-manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "finetune" / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-file-bytes", type=int, default=65536)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(json.dumps(build(manifest, args.out, args.seed, args.max_file_bytes),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
