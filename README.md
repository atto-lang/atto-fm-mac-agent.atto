# Ores LLM — Apple Silicon Local Training

A local-first language model lab for Apple Silicon, with **two separate backends**:

1. **Ores TinyLM**: an owned, train-from-scratch, character-tokenized causal Transformer implemented with Apple MLX. It is an educational/experimental baseline, *not* a production-grade assistant.
2. **Apple FM adapter**: optional use of the pre-installed macOS 27 `fm` command to compare inference; Apple's pretrained model is **not** trained or redistributed here.

## Quick start (M4 Max or M5 Pro)

```sh
git clone https://github.com/ores-truffle-oreslang/ores-fm-mac-agent.ores.git
cd ores-fm-mac-agent.ores
git switch feat/apple-silicon-tinyllm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/static_gate.py
python3 -m unittest discover -s tests -v
python3 scripts/train.py --text data/sample.txt --out artifacts/tiny --steps 200 --context 16 --device gpu
python3 scripts/generate.py --checkpoint artifacts/tiny --prompt "Oreslang"
```

Requires macOS on Apple Silicon and an MLX-compatible Python environment. The sample corpus only demonstrates that the pipeline runs: real language quality needs a much larger, lawfully sourced, carefully split corpus, compute and evaluation.

## Model and safety boundaries

- Character-level tokenizer with explicit unknown-character rejection; vocabulary is built only from training data and saved with weights. Never silently remap an unknown token.
- Causal masking, next-token cross-entropy, deterministic dataset split, seeded sampling, context-length validation, and checkpoint metadata.
- Checkpoint paths are local. Do not commit training data, model weights, secrets or machine/user files.
- No network activity in training/generation scripts. Installation may access package indexes; MLX itself is a local runtime.
- A single train job uses one Mac GPU. Two Macs do **not** combine transparently; evaluate on the second Mac or coordinate distributed training separately.
- Do not assume GPU-core count alone predicts throughput; measure tokens/sec, training loss, GPU availability, and unified memory. The current script does not measure memory bandwidth or energy.

## Apple Foundation Models comparison

On macOS 27 only, subject to eligibility and the Apple license:
```sh
fm available
fm respond "Explain actor isolation in one paragraph."
python3 scripts/fm_adapter.py "Explain actor isolation in one paragraph."
```

The adapter has no train operation. Apple's `fm` context constraints do not cap our own Transformer; TinyLM's context is configured independently. Never automatically submit private source code to any network or cloud model.

## Next stages

1. Confirm local MLX train/generate smoke tests on both Macs; collect real GPU throughput and unified-memory capacity.
2. Add repository-permissioned Oreslang corpus preparation, licenses, deduplication, train/valid/test leakage checks and held-out evaluations.
3. Add a separate open-weights MLX LM LoRA/QLoRA pathway for useful assistant quality, with pinned model and license.
4. Add an Oreslang-facing local process/HTTP interface, then benchmark and harden it before backend-neutral claims.


## One-command hardware acceptance (run on **each** Mac separately)

The single-command runner executes the MLX model invariants, trains on Metal,
reloads a real checkpoint, generates tokens, and writes JSON reports. It fails if
GPU admission fails; there is **no silent CPU fallback**.

```sh
# On the 40-core Apple M4 Max:
python3 scripts/smoke_mac.py --device-tag m4-max --steps 100

# On the 20-core Apple M5 Pro:
python3 scripts/smoke_mac.py --device-tag m5-pro --steps 100
```

The script checks that tracked Git files are clean and records the exact\n40-character source commit SHA. Both Macs must check out the **same commit**.\nThe script checks the chip name and GPU core count reported by
`system_profiler SPDisplaysDataType`. Use the command matching the Mac you are
currently on. For longer runs increase `--steps`, preserving the same dataset,
seed, context, batch size and MLX version on both Macs.

Artifacts (ignored by Git) are written under
`artifacts/benchmarks/<device-tag>/`:
- `gpu-smoke-report.json`: PASS marker only after all gates succeed; hardware and
  measured wall time
- `model/report.json`: corpus SHA-256, train/validation size, best loss, device,
  train-only tokens/sec, configuration and seed
- `model/metrics.jsonl`: deterministic full-validation loss at checkpoints
- `model/weights.safetensors`, `vocab.json`, `config.json`: local checkpoint

Do not commit generated artifacts or private data. The training corpus is a tiny
smoke fixture, not a benchmark for model intelligence or a proof of useful code
generation. The reported throughput excludes validation and checkpointing time;
end-to-end wall time is also recorded separately.

### Evidence boundaries

- `python3 scripts/static_gate.py` and GitHub CI only check source syntax and
  the tiny corpus; they do **not** execute MLX or prove GPU execution.
- `python3 -m unittest discover -s tests -v` checks attention causality, finite
  gradients, parameter updates, checkpoint round-tripping and deterministic
  held-out scoring, but must be run on a compatible MLX machine.
- Neither of your Macs can be accessed automatically by a GitHub-hosted Linux
  runner. Real M4/M5 pass evidence requires the local reports from both Macs.
- Validation is a contiguous 10% holdout and scores every held-out next-token
  target exactly once, resetting context at slice boundaries. It is not a
  corpus-level leakage audit or a final test-set evaluation.
- Open-weight LoRA experiments and Oreslang-native model serving are future
  milestones separate from Apple's macOS 27 `fm` runtime.

## Compare two GPU acceptance reports (only after both runs pass)

Copy the two local `gpu-smoke-report.json` files onto one machine, keeping
names distinct. The comparator has no MLX dependency:

```sh
python3 scripts/compare_benchmarks.py \\
  --m4 artifacts/benchmarks/m4-max/gpu-smoke-report.json \\
  --m5 artifacts/benchmarks/m5-pro/gpu-smoke-report.json \\
  --out artifacts/benchmarks/comparison.json
```

It refuses failed/CPU/dirty-source reports, mismatched Git commits,
models, corpus hashes, seeds, MLX versions or hyperparameters. It does **not**
verify cryptographic authenticity of user-provided JSON, and 100-step smoke
throughput is not a stable peak-performance or coding-quality benchmark.
The dependency-free comparator tests can run even on Linux:

```sh
python3 -m unittest discover -s tests -p test_benchmarks.py -v
```

## New experimental track: Oreslang corpus + offline MLX LoRA

See [Oreslang fine-tuning design and security gates](docs/FINETUNING.md).
This is a **separate, stacked draft PR** over the TinyLM baseline; the
trainable owned TinyLM is not the same as a LoRA adapter to independently
licensed open model weights. No real Mac GPU run or Oreslang compiler acceptance
is yet certified for this fine-tuning track.
