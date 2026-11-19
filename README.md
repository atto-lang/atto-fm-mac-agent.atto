> **AI agent instructions (organization-wide):** Before working on code, issues, or pull requests, read and follow the [organization `AGENTS.md`](https://github.com/ores-truffle-oreslang/.github/blob/main/AGENTS.md), including its **merge-first stealth-development policy**. Honor any repository-local `AGENTS.md`, explicit holds, and GitHub-enforced rules.

# Ores LLM — Apple Silicon Local Training

## AttoLang naming and specification migration (2026-11-18)

**AttoLang** is the current name of **Oreslang**. This is a legacy-organization checkout, `ores-truffle-oreslang/ores-fm-mac-agent.ores`, and new corresponding work belongs in not yet verified as a 1:1 repo (browse [atto-lang](https://github.com/atto-lang)). Matching names do **not** establish GitHub transfer, identical commits, issues/PRs, release history or feature completeness. Preserve this checkout's historical commands and paths where still applicable.

The **canonical language specification is [atto-spec](https://github.com/atto-lang/atto-spec)**; [atto-platform-conformance](https://github.com/atto-lang/atto-platform-conformance) owns portable executable positive/negative tests, [atto-common-source](https://github.com/atto-lang/atto-common-source) owns portable libraries, and `atto-source.*` owns backend implementation. Modern source targets include `.atto` and `snake_case`, explicit `borrow!`/`take!`/`copy!`/`copy!!`/`share!`, [implicit argument borrowing](https://github.com/atto-lang/atto-spec/blob/main/language/implicit-argument-borrow-v1.md), [file/module `define const` including one-time `os.read_env`](https://github.com/atto-lang/atto-spec/blob/main/language/file-module-const-v1.md), and [`<<<`/`>>>`/`ashr` bitwise shifts vs. `<<`/`>>` directional stream operations](https://github.com/atto-lang/atto-spec/blob/main/language/bitwise-and-stream-operators-v1.md). They are language contract targets, **not** evidence every legacy backend supports them.

Keep working legacy `.ores`, CLI/package/ABI and wire identifiers until separately verified migrations. No Python tools, scripts or tests for new work; use Rust for support tools where appropriate. Feature parity claims require exact SHA/tree, target/toolchain, real compiled guest execution and positive/negative checks. Empty-step workflows and host-only tests are **not executed** guest parity evidence.

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
