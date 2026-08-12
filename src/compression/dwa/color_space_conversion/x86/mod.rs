// Runtime x86 SIMD dispatch for the DWA CSC transform: `try_csc709_*_8x8_batch`
// select the AVX2 tier when available and fall back to the SSE2 tier,
// otherwise let the caller use the scalar autovectorized path.

use crate::compression::simd_tier::x86::miraculix_x86::{avx, sse};

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
    if let Some(avx) = avx() {
        avx2::csc709_forward_8x8_batch(avx, blocks);
        return true;
    }
    if let Some(sse) = sse() {
        sse2::csc709_forward_8x8_batch(sse, blocks);
        return true;
    }
    false
}

pub(super) fn try_csc709_inverse_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(avx) = avx() {
        avx2::csc709_inverse_8x8_batch(avx, blocks);
        return true;
    }
    if let Some(sse) = sse() {
        sse2::csc709_inverse_8x8_batch(sse, blocks);
        return true;
    }
    false
}
