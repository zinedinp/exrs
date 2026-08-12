// Runtime x86 SIMD dispatch for the DWA CSC transform: `try_csc709_*_8x8_batch`
// select the AVX2 tier when available and fall back to the SSE2 tier,
// otherwise let the caller use the scalar autovectorized path.

use crate::compression::simd_tier::x86::miraculix_x86::{avx, sse};

// public only for benchmarking. Named `avx` not `avx2`: only the base `Avx`
// token (f32 arithmetic) is used, no AVX2-specific int ops -- see file doc.
#[doc(hidden)]
pub mod avx;

// Stage-1 prototype (not yet wired into dispatch below); public only for
// benchmarking/correctness testing. Named `avx512f`: only the base
// `Avx512f` token is used.
#[doc(hidden)]
pub mod avx512f;

// public only for benchmarking. Named `sse` not `sse2`: only the base `Sse`
// token (f32 arithmetic) is used -- see file doc.
#[doc(hidden)]
pub mod sse;

pub(super) fn try_csc709_forward_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(avx_token) = avx() {
        self::avx::csc709_forward_8x8_batch(avx_token, blocks);
        return true;
    }
    if let Some(sse_token) = sse() {
        self::sse::csc709_forward_8x8_batch(sse_token, blocks);
        return true;
    }
    false
}

pub(super) fn try_csc709_inverse_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(avx_token) = avx() {
        self::avx::csc709_inverse_8x8_batch(avx_token, blocks);
        return true;
    }
    if let Some(sse_token) = sse() {
        self::sse::csc709_inverse_8x8_batch(sse_token, blocks);
        return true;
    }
    false
}
