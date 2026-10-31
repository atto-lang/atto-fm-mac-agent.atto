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
python3 scripts/train.py --text data/sample.txt --out artifacts/tiny --steps 200 --context 16
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
