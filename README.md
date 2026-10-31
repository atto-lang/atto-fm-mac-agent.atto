# Ores LLM — native Oreslang first

Native Oreslang owns training, inference, model lifecycle, data access,
tokenization and actor state. Rust is **support-only**, used for an optional
bounded C ABI numeric accelerator. **No Python dependencies**.

The first implementation is
[`native/src/bigram_bootstrap.ores`](native/src/bigram_bootstrap.ores):
a tiny two-symbol smoothed bigram that trains from fixed observations, makes
held-out next-token predictions and returns 42 if its checks pass.
This is a **native training bootstrap**, not a Transformer, an Apple GPU
benchmark or a production language model.

[Architecture, compiler status and safety gates](docs/NATIVE_ORESLANG_FIRST.md)

## What can be checked now

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The support crate contains an explicit CPU f32 matmul implementation with
bounded shape checks. Metal requests return UNSUPPORTED; **no CPU fallback**.

## Real Oreslang native AOT gate

Build the actual `oreslang-source.llvm` compiler separately, then:

```sh
cargo run -p ores-llm-native-gate -- \
  --llvm-root /ABS/PATH/TO/oreslang-source.llvm
```

This runs `scripts/compile-aot.sh` on the real Oreslang guest source, then
executes the produced native binary and checks for exit code 42.
No source-to-AOT result should be assumed until this exact command succeeds.

## Evidence status

All source code has been committed, but the Oreslang program and Rust support
must still be compiled and tested. Apple M4 Max and M5 Pro Metal kernels,
autograd, adapters and distributed training **are not implemented**.

This branch is independent of the earlier Python-oriented draft PRs #1 and #2.
Those PRs are not prerequisites for the native-first implementation.
