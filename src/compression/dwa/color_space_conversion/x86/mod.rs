// Runtime x86 SIMD dispatch for the DWA CSC transform: `try_csc709_*_8x8_batch`
// select the AVX2 tier when available and fall back to the SSE2 tier,
// otherwise let the caller use the scalar autovectorized path.

use crate::compression::simd_tier::x86::{v1, v3};

// public only for benchmarking
#[doc(hidden)]
pub mod avx2;

// Stage-1 prototype (not yet wired into dispatch below); public only for
// benchmarking/correctness testing.
#[doc(hidden)]
pub mod avx512;

// public only for benchmarking
#[doc(hidden)]
pub mod sse2;

pub(super) fn try_csc709_forward_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(v3) = v3() {
        avx2::csc709_forward_8x8_batch(v3, blocks);
        return true;
    }
    if let Some(v1) = v1() {
        sse2::csc709_forward_8x8_batch(v1, blocks);
        return true;
    }
    false
}

pub(super) fn try_csc709_inverse_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(v3) = v3() {
        avx2::csc709_inverse_8x8_batch(v3, blocks);
        return true;
    }
    if let Some(v1) = v1() {
        sse2::csc709_inverse_8x8_batch(v1, blocks);
        return true;
    }
    false
}
