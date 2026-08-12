// Runtime x86 SIMD dispatch for the DWA DCT: `try_dct_*_8x8_batch` select the
// AVX tier when available and fall back to the SSE tier, otherwise let the
// caller use the scalar autovectorized path. Both tiers share the lazily-built
// `forward_basis` cosine table.

use crate::compression::simd_tier::x86::miraculix_x86::{avx, sse};

// public only for benchmarking. Named `avx` not `avx2`: only the base `Avx`
// token (f32 arithmetic) is used, no AVX2-specific int ops -- see file doc.
#[doc(hidden)]
pub mod avx;

// Stage-1 prototype (not yet wired into dispatch below) -- public only for
// benchmarking/correctness testing. Named `avx512dq`: `Avx512Dq` is the
// most restrictive of the 3 tokens this file needs (`Avx512f`/`Avx`/
// `Avx512Dq`) -- not every AVX-512F host has DQ (Knights Landing didn't).
#[doc(hidden)]
pub mod avx512dq;

// public only for benchmarking. Named `sse` not `sse2`: only the base `Sse`
// token (f32 arithmetic) is used -- see file doc.
#[doc(hidden)]
pub mod sse;

pub(super) fn try_dct_forward_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [f32; 64]>,
{
    if let Some(avx_token) = avx() {
        self::avx::dct_forward_8x8_batch(avx_token, blocks);
        return true;
    }
    if let Some(sse_token) = sse() {
        for data in blocks {
            self::sse::dct_forward_8x8(sse_token, data);
        }
        return true;
    }
    false
}

pub(super) fn try_dct_inverse_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [f32; 64]>,
{
    if let Some(avx_token) = avx() {
        self::avx::dct_inverse_8x8_batch(avx_token, blocks);
        return true;
    }
    if let Some(sse_token) = sse() {
        for data in blocks {
            self::sse::dct_inverse_8x8(sse_token, data);
        }
        return true;
    }
    false
}
