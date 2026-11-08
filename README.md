# Ores LLM — Apple Silicon Local Training

A local-first language model lab for Apple Silicon, with **two separate backends**:

1. **Ores TinyLM**: an owned, train-from-scratch, character-tokenized causal Transformer implemented with Apple MLX. It is an educational/experimental baseline, *not* a production-grade assistant.
2. **Apple FM adapter**: optional use of the pre-installed macOS 27 `fm` command to compare inference; Apple's pretrained model is **not** trained or redistributed here.

## Quick start (M4 Max or M5 Pro)

```sh
git clone https://github.com/ores-truffle-oreslang/ores-fm-mac-agent.ores.git
cd ores-fm-mac-agent.ores
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/train.py --text data/sample.txt --out artifacts/tiny --steps 200 --context 128
python3 scripts/generate.py --checkpoint artifacts/tiny --prompt "Oreslang"
python3 -m unittest discover -s tests -v
```

Requires macOS on Apple Silicon and an MLX-compatible Python environment. The sample corpus only demonstrates that the pipeline runs: real language quality needs a much larger, lawfully sourced, carefully split corpus, compute and evaluation.

## Model and safety boundaries

- Character-level tokenizer with explicit unknown-character rejection; vocabulary is built only from training data and saved with weights. Never silently remap an unknown token.
- Causal masking, next-token cross-entropy, deterministic dataset split, seeded sampling, context-length validation, and checkpoint metadata.
- Checkpoint paths are local. Do not commit training data, model weights, secrets or machine/user files.
- No network activity in training/generation scripts. Installation may access package indexes; MLX itself is a local runtime.
- A single train job uses one Mac GPU. Two Macs do **not** combine transparently; evaluate on the second Mac or coordinate distributed training separately.
- Do not assume GPU-core count alone predicts throughput; record memory, memory bandwidth, batch, sequence length, tokens/sec and temperature.

## Apple Foundation Models comparison

On macOS 27 only, subject to eligibility and the Apple license:
```sh
fm available
fm respond "Explain actor isolation in one paragraph."
python3 scripts/fm_adapter.py "Explain actor isolation in one paragraph."
```

The adapter has no train operation. Apple's `fm` context constraints do not cap our own Transformer; TinyLM's context is configured independently. Never automatically submit private source code to any network or cloud model.

## Next stages

1. Confirm local MLX train/generate smoke tests on both Macs; collect real GPU throughput and memory metrics.
2. Add repository-permissioned Oreslang corpus preparation, licenses, deduplication, train/valid/test leakage checks and held-out evaluations.
3. Add a separate open-weights MLX LM LoRA/QLoRA pathway for useful assistant quality, with pinned model and license.
4. Add an Oreslang-facing local process/HTTP interface, then benchmark and harden it before backend-neutral claims.

## Repository ownership and migration (2026-11-08)

**AttoLang** is the public name of the language; this code remains `ores-truffle-oreslang/ores-fm-mac-agent.ores` (historical Oreslang). Preserve existing `.ores` source compatibility, old package names and CLI identifiers until separately proven changes are merged.

- **Scope:** local macOS agent experiments and explicit device capability/security contracts.
- **Excluded:** unrestricted device privileges, cloud administration or remote device trust bypass.
- **Organizational responsibility:** [atto-lang](https://github.com/atto-lang) owns portable language libraries, research code, compilers, tooling, demos and conformance; [atto-co](https://github.com/atto-co) independently owns cloud hosting, deployments, tenant infrastructure, identity, commercial services and billing. Pre-existing similarly named `atto-*` repositories are distinct Git identities absent transfer receipts.
- **Proposed first migration:** authorized GitHub transfer of the same original repo ID/name/history into `atto-lang/ores-fm-mac-agent.ores`; no direct old-org → `atto-co` transfers are approved under the current plan. Any later product carve-out requires independent review.

**Per-repo cutover checklist:** coordinate in [issue #26](https://github.com/ores-truffle-oreslang/.github/issues/26) and consult private [migration runbook draft #25](https://github.com/ores-truffle-oreslang/.github/pull/25). Before moving, verify exact commit SHA/tree and stable numeric ID, source/target owner and independent security approval, destination availability/visibility/protections, Actions/runners, secrets, webhooks, Apps, PRs/releases, packages/consumer URLs, Pages and rollback owner. Don't mirror private source without authorization. After an approved GitHub ownership transfer verify ID/history/tree equivalence, restore access and policy, update consumers and rerun exact-head tests with **nonzero executed steps**. `steps=[]`/`null` means `NOT_EXECUTED`; static or host-only tests do not certify real AttoLang guest code or security isolation.
