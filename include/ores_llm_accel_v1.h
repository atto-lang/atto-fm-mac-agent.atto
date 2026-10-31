#ifndef ORES_LLM_ACCEL_V1_H
#define ORES_LLM_ACCEL_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Stable version 1 ABI; native Oreslang owns model, tokenizer, training and
// actor lifecycle. Rust only accelerates bounded numerical operations.
// All pointers are borrowed for the call only. No allocation is returned.
#define ORES_LLM_ACCEL_ABI_VERSION 1u
#define ORES_LLM_DEVICE_CPU 0u
#define ORES_LLM_DEVICE_METAL 1u
#define ORES_LLM_STATUS_OK 0
#define ORES_LLM_STATUS_INVALID_ARGUMENT 1
#define ORES_LLM_STATUS_UNSUPPORTED 2
#define ORES_LLM_STATUS_OVERLAP 3
#define ORES_LLM_STATUS_RESOURCE_LIMIT 4
#define ORES_LLM_STATUS_INTERNAL 5

uint32_t ores_llm_accel_v1_abi_version(void);

// Returns OK for the explicit CPU reference; Metal returns UNSUPPORTED until
// real Apple GPU kernels and synchronization are certified on both Macs.
int32_t ores_llm_accel_v1_device_status(uint32_t device);

// F32 dense row-major matmul: out[rows, cols] = lhs[rows, inner] * rhs[inner, cols].
// Caller owns all arrays; lengths are in floats, not bytes. No overlapping spans.
// On failure out is NOT modified. No asynchronous use, retention or borrowing.
// Although pointer alignment, nulls, lengths and span overlap are checked, a
// caller still must provide valid, mapped, initialized memory for each span.
// Device is explicit: CPU reference available, Metal unsupported for now.
int32_t ores_llm_accel_v1_matmul_f32(
    uint32_t device,
    const float* lhs, uint64_t lhs_len,
    const float* rhs, uint64_t rhs_len,
    float* out, uint64_t out_len,
    uint32_t rows, uint32_t inner, uint32_t cols);

#ifdef __cplusplus
}
#endif
#endif
