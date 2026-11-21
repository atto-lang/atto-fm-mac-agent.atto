# Oreslang corpus and offline MLX LoRA/QLoRA

**Status: unexecuted source code.** PR #1's Mac smoke tests have not been verified. A LoRA adapter based on licensed open weights is **not** a from-scratch owned model.

## Authorized source manifest

Copy configs/corpus.manifest.example.json to artifacts/finetune/manifest.json and edit the example. Every entry requires an absolute local Git checkout path, full actual Git HEAD SHA, stable source ID, allowed license, and authorized=true after human rights review. At least three different repositories are required; holdout assignment is repository-disjoint. No remote account is accessed.

The script reads **only tracked .ores and .md files** at verified clean Git HEAD. It blocks links, unexpected large or binary files, common secret patterns and normalized exact duplicate text across repositories. Untracked files are never included. This is not a complete credential/PII or semantic near-duplicate audit. Verify legal rights and data confidentiality manually; this manifest alone does not establish a legal license.

Run from the project checkout:

    mkdir -p artifacts/finetune
    cp configs/corpus.manifest.example.json artifacts/finetune/manifest.json
    # Fill in path, full revision, approved license, authorized=true in untracked manifest
    python3 scripts/build_corpus.py --manifest artifacts/finetune/manifest.json --out artifacts/finetune/data

Output directory must not exist before build. The JSONL train/valid/test data and SHA-256 provenance report stay in ignored artifacts/. The first two seeded-hash-sorted repository IDs become validation and test groups; all others train. With only three groups this is not a balanced or sufficient corpus. Do not claim robust model quality until a larger curated corpus and independent evaluation set exist.

## Local-only MLX LoRA/QLoRA

Use Python 3.11+ on native Apple Silicon. Obtain a reviewed, compatible, *locally stored* open-weight instruct/code base model. No model download or remote IDs are accepted by the launcher. Confirm its exact license and allowed use before providing --confirm-model-rights. Start with a small quantized candidate based on available measured unified memory.

    python3 -m venv .venv-lora
    source .venv-lora/bin/activate
    pip install -r requirements-lora.txt
    python3 scripts/lora_local.py --model /ABSOLUTE/PATH/TO/LOCAL/MODEL --data artifacts/finetune/data --confirm-model-rights --dry-run

After preflight, train and test:

    python3 scripts/lora_local.py --model /ABSOLUTE/PATH/TO/LOCAL/MODEL --data artifacts/finetune/data --adapter artifacts/finetune/adapter --mode train --iters 100 --batch-size 1 --num-layers 8 --max-seq-length 512 --confirm-model-rights
    python3 scripts/lora_local.py --model /ABSOLUTE/PATH/TO/LOCAL/MODEL --data artifacts/finetune/data --adapter artifacts/finetune/adapter --mode test --confirm-model-rights

The local adapter and checkpoints stay in artifacts/. For quantized weights MLX LM performs LoRA adaptation over quantized weights (QLoRA). Training is not performed by Apple's built-in macOS fm model. The wrapper sets HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE and disables W&B. **These flags are not an operating-system firewall**: disconnect network or use an OS sandbox/firewall if strict network non-egress is required.

MLX LM reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md

## Compiler acceptance hook

Generate candidate .ores files separately, then evaluate with your trusted compiler, using its actual validated command-line syntax:

    python3 scripts/eval_compiler.py --compiler /ABSOLUTE/PATH/TO/COMPILER --source-dir /ABSOLUTE/PATH/TO/GENERATED --compiler-arg check --compiler-arg '{source}' --out artifacts/evaluation/compiler.json

The 'check' argument is **illustrative**, not a verified Oreslang compiler API. Evaluation measures parser/compiler exit codes only; it does not execute compiled programs, measure correctness or compile all five backends. Run the compiler on untrusted code in an OS sandbox; this utility does **not** provide sandbox isolation.

## Execution / promotion gates

1. PR #1 completes exact-head M4 Max and M5 Pro Metal smoke tests.
2. Source tree and unit tests of this branch pass; GitHub runner admission must be restored or use a proven alternative.
3. Human-reviewed licenses, private source authorization, secrets/PII checks, near-duplicate and split audits.
4. M4 Max real offline LoRA train/test saves working adapter.
5. M5 Pro evaluates held-out code against real Oreslang compiler (not the included fake compiler tests).
6. Document baseline vs adapter compilation success before requesting merge.

Dependency-free tests:

    python3 -m unittest discover -s tests -p test_corpus_pipeline.py -v
    python3 -m unittest discover -s tests -p test_compiler_gate.py -v
