//! Optional FFI support for Oreslang-owned ML. No model/training policy lives here.
//! v1 implements ONLY the bounded synchronous CPU matrix multiplication reference.
//! Metal remains unsupported until actual native GPU dispatch and fence tests pass.
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};

const OK: i32 = 0;
const INVALID: i32 = 1;
const UNSUPPORTED: i32 = 2;
const OVERLAP: i32 = 3;
const LIMIT: i32 = 4;
const INTERNAL: i32 = 5;
const MAX_ELEMENTS: usize = 16_777_216;
const MAX_DIMENSION: u32 = 4096;

#[no_mangle]
pub extern "C" fn ores_llm_accel_v1_abi_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn ores_llm_accel_v1_device_status(device: u32) -> i32 {
    match device {
        0 => OK,
        1 => UNSUPPORTED, // Metal is not implemented.
        _ => INVALID,
    }
}

fn checked_span(ptr: *const f32, len: usize) -> Option<(usize, usize)> {
    let begin = ptr as usize;
    if ptr.is_null() || begin % align_of::<f32>() != 0 {
        return None;
    }
    let bytes = len.checked_mul(size_of::<f32>())?;
    Some((begin, begin.checked_add(bytes)?))
}

fn intersects(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn multiply_dims(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b).filter(|v| *v > 0 && *v <= MAX_ELEMENTS)
}

#[allow(clippy::too_many_arguments)]
fn matmul_checked(
    device: u32,
    lhs: *const f32,
    lhs_len: u64,
    rhs: *const f32,
    rhs_len: u64,
    out: *mut f32,
    out_len: u64,
    rows: u32,
    inner: u32,
    cols: u32,
) -> i32 {
    let status = ores_llm_accel_v1_device_status(device);
    if status != OK {
        return status;
    }
    if rows == 0 || inner == 0 || cols == 0 {
        return INVALID;
    }
    if rows > MAX_DIMENSION || inner > MAX_DIMENSION || cols > MAX_DIMENSION {
        return LIMIT;
    }
    let Some(a_len) = multiply_dims(rows as usize, inner as usize) else {
        return LIMIT;
    };
    let Some(b_len) = multiply_dims(inner as usize, cols as usize) else {
        return LIMIT;
    };
    let Some(c_len) = multiply_dims(rows as usize, cols as usize) else {
        return LIMIT;
    };
    if lhs_len != a_len as u64 || rhs_len != b_len as u64 || out_len != c_len as u64 {
        return INVALID;
    }
    let Some(a) = checked_span(lhs, a_len) else {
        return INVALID;
    };
    let Some(b) = checked_span(rhs, b_len) else {
        return INVALID;
    };
    let Some(c) = checked_span(out.cast_const(), c_len) else {
        return INVALID;
    };
    if intersects(a, b) || intersects(a, c) || intersects(b, c) {
        return OVERLAP;
    }

    // Safety: this FFI function requires caller-provided valid, initialized
    // spans. We verify length, alignment, overflow, and non-overlap before
    // creating Rust references. No pointer is retained or used after return.
    let (left, right, output) = unsafe {
        (
            std::slice::from_raw_parts(lhs, a_len),
            std::slice::from_raw_parts(rhs, b_len),
            std::slice::from_raw_parts_mut(out, c_len),
        )
    };
    for row in 0..rows as usize {
        for col in 0..cols as usize {
            let mut accum = 0.0_f32;
            for k in 0..inner as usize {
                accum += left[row * inner as usize + k] * right[k * cols as usize + col];
            }
            output[row * cols as usize + col] = accum;
        }
    }
    OK
}

/// # Safety
///
/// All pointers must refer to mapped, initialized f32 spans with the supplied
/// lengths and remain valid and exclusively borrowed for this call. The output
/// must be writable. Overlap, null pointers, bad shapes and overflow are
/// checked, but arbitrary invalid non-null pointers cannot be validated by FFI.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ores_llm_accel_v1_matmul_f32(
    device: u32,
    lhs: *const f32,
    lhs_len: u64,
    rhs: *const f32,
    rhs_len: u64,
    out: *mut f32,
    out_len: u64,
    rows: u32,
    inner: u32,
    cols: u32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        matmul_checked(
            device, lhs, lhs_len, rhs, rhs_len, out, out_len, rows, inner, cols,
        )
    }))
    .unwrap_or(INTERNAL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_and_hardware_admission_are_explicit() {
        assert_eq!(ores_llm_accel_v1_abi_version(), 1);
        assert_eq!(ores_llm_accel_v1_device_status(0), OK);
        assert_eq!(ores_llm_accel_v1_device_status(1), UNSUPPORTED);
        assert_eq!(ores_llm_accel_v1_device_status(99), INVALID);
    }

    #[test]
    fn row_major_reference_computes_product() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = [0.0; 4];
        let status = unsafe {
            ores_llm_accel_v1_matmul_f32(
                0, a.as_ptr(), 6, b.as_ptr(), 6, out.as_mut_ptr(), 4, 2, 3, 2,
            )
        };
        assert_eq!(status, OK);
        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn incompatible_shape_does_not_modify_output() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let mut out = [37.0; 1];
        let status = unsafe {
            ores_llm_accel_v1_matmul_f32(
                0, a.as_ptr(), 1, b.as_ptr(), 2, out.as_mut_ptr(), 1, 1, 2, 1,
            )
        };
        assert_eq!(status, INVALID);
        assert_eq!(out[0], 37.0);
    }

    #[test]
    fn metal_fails_closed_without_touching_output() {
        let a = [1.0];
        let b = [2.0];
        let mut out = [37.0];
        let status = unsafe {
            ores_llm_accel_v1_matmul_f32(
                1, a.as_ptr(), 1, b.as_ptr(), 1, out.as_mut_ptr(), 1, 1, 1, 1,
            )
        };
        assert_eq!(status, UNSUPPORTED);
        assert_eq!(out, [37.0]);
    }

    #[test]
    fn overlapping_buffers_are_rejected() {
        let mut shared = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0];
        let status = unsafe {
            ores_llm_accel_v1_matmul_f32(
                0, shared.as_ptr(), 1, b.as_ptr(), 1, shared.as_mut_ptr(), 1, 1, 1, 1,
            )
        };
        assert_eq!(status, OVERLAP);
    }

    #[test]
    fn dimension_limit_and_null_pointer_are_rejected() {
        let a = [1.0];
        let mut out = [0.0];
        let null = unsafe {
            ores_llm_accel_v1_matmul_f32(
                0, std::ptr::null(), 1, a.as_ptr(), 1, out.as_mut_ptr(), 1, 1, 1, 1,
            )
        };
        assert_eq!(null, INVALID);
        let over_limit = unsafe {
            ores_llm_accel_v1_matmul_f32(
                0, a.as_ptr(), 1, a.as_ptr(), 1, out.as_mut_ptr(), 1, 5000, 1, 1,
            )
        };
        assert_eq!(over_limit, LIMIT);
    }
}
