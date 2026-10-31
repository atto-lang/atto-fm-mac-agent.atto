# Native Oreslang first — Ores LLM

## Authority and ownership

This repository implements the **native Oreslang model first**. Rust is an optional
numeric support library, not a separate training service and not the owner of
model state, tokenization, dataset access, checkpoints, actors or permissions.

The existing Python MLX prototype draft PR #1 and stacked corpus draft PR #2
remain historical experiments. This implementation is **based on main**, not
stacked on Python. Python is never required or invoked by this implementation.

## What exists today

- `native/src/bigram_bootstrap.ores` trains a real, tiny 2-symbol
  smoothed bigram frequency model on eight adjacent-token observations.
  It predicts the next symbol on two held-out transitions. Its expected AOT
  exit status is 42. There are no dynamic datasets, tensors, gradients or GPU
  kernels in this narrow stage. It is **not a Transformer or useful LLM**.
- `support/native-gate` drives the **real Oreslang LLVM compiler** through the
  repo's own `scripts/compile-aot.sh`, executes the produced machine binary
  and verifies exit 42. Neither a Rust rewrite of the algorithm nor a source
  pattern matcher can substitute for this test.
- `support/accelerator` exposes a C ABI v1 with explicit device selection,
  row-major f32 matrix multiplication, pointer/shape checks and zero hidden
  allocation; **only CPU is implemented**. Asking for Metal fails with
  UNSUPPORTED. The Oreslang source does not call this ABI yet.
- `include/ores_llm_accel_v1.h` is the only ABI header. Rust's CPU operation
  can be linked behind an Oreslang FFI once LLVM supports verified safe foreign
  buffers, ownership and shared-library loading.

## Run tests — no Python

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Real Oreslang source-to-native test **requires a built LLVM Oreslang compiler**:

```bash
cargo run -p ores-llm-native-gate -- --llvm-root /ABS/PATH/TO/oreslang-source.llvm
```

The gate calls `scripts/compile-aot.sh`, not a simulator or a Python wrapper.
Compiler admission, AOT linking, execution and exit code 42 must all pass.
If the current compiler rejects the native fixture, record the exact error and
patch the smallest missing compiler feature rather than silently moving model
logic to Rust. Re-run after every compiler source SHA change.

## Core runtime contracts to add in native Oreslang

The next phase is the production Oreslang `ores_llm` module, which owns:

1. `Tensor<T>` shape, layout, dtype, alignment, strides and actor-owned
   arenas. No buffer is accessible after `rt take` or across an expired
   borrow. No device/host simultaneous mutable borrows.
2. `Tokenizer` in native Oreslang, with explicit byte fallback, deterministic
   encode/decode and vocabulary hash.
3. `Dataset` authorization, source revision verification, train/valid/test
   group isolation, content hashing and privacy gates. Rust must not decide
   which data may be read or used.
4. `Model` and `Optimizer` (Transformer attention and AdamW first), seeded
   training, cancellation, checkpoint and eval. Oreslang determines the graph
   and update rules; accelerators only execute bounded tensor operators.
5. Actor supervisors: `ModelTrainer`, `InferenceWorker`,
   `DatasetReader` and `DeviceManager`, with bounded mailboxes,
   cancellation, CONTROL-path panic handling and sealed state.
6. A versioned `ModelService` contract that all five Oreslang backends can
   use, and capability negotiation for unsupported GPU backends.

### Proposed actor API — specification, NOT currently executable

```ores
define actor ModelTrainer as
  // Training data and optimizer are stored in sealed actor-owned state.
  // Pseudocode for the desired high-level language contract:
  receive(TrainBatch batch) -> Result<StepReport, Error> {
    return handle_state(critical-write |state| -> {
      return state.train_step(batch);
    });
  }
end
```

The LLVM compiler currently documents CPU/GPU block admission gaps and the
native sealed-state/actor subset is incomplete. Do **not** claim this snippet
compiles until actual positive/negative compiler fixtures pass.

## Accelerator ABI and security model

- Device `0` = explicit CPU reference; device `1` = requested Metal,
  currently UNSUPPORTED (no CPU fallback).
- Oreslang owns input/output buffers. Rust only borrows initialized,
  non-overlapping f32 spans **for the duration of one synchronous call**.
- ABI validates pointer null/alignment, lengths, checked multiplication,
  4096 maximum dimension and 16,777,216 element budget. Invalid input must
  return a deterministic status and must not change output.
- Native Oreslang must validate live region handles, actor incarnation and
  borrow permissions **before constructing C pointers**. Rust cannot prove
  arbitrary pointer provenance or protect against malicious invalid pointers.
- Runtime ABI v1 is not a stable binary packaging guarantee until cross-language
  ABI testing and safe actor/FFI lifetime checks execute on actual targets.
- Metal kernels, fences, GPU/CPU ownership transfer, float32 numerical parity,
  mixed precision, auto-differentiation and quantization remain future work.

## Evidence gates

| Gate | Requirement | Status |
| --- | --- | --- |
| Pure Rust support | cargo fmt / clippy / unit tests | authored, execution unverified |
| Native bigram | LLVM compiler AOT binary exits 42 | authored, execution unverified |
| Oreslang actor/FFI | protected buffer ownership and FFI integration | not implemented |
| Metal | real M4 Max and M5 Pro dispatch, gradients and benchmarks | not implemented |
| Production model | Transformer, tokenizer, training/eval, own checkpoints | not implemented |
| CI | real jobs run steps, not steps=null | admission blocked on source org |

Source files and tests are **proposals until executed**. Stop promotion on
missing runtime evidence. For macOS GPU performance, do not claim the user's
40 + 20 cores combine automatically; distributed training requires explicit
transfer and synchronization.

## Next compiler milestone

Start with the native bigram's source-to-AOT proof. Next add bounded
`Tensor<f32>` allocations/borrows, a 2x2 f32 matmul guest fixture, a
rejection test for stale/overlapping buffer loans and a precise v1 FFI binding.
Only then implement Metal kernels in the supporting Rust layer (or a direct
native Oreslang Metal runtime binding) and run parity with the CPU reference.
