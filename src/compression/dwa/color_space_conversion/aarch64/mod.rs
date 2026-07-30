//! Runtime aarch64 SIMD dispatch for the DWA CSC transform.

use pulp::aarch64::Neon;

// public only for benchmarking / correctness tests
#[doc(hidden)]
pub mod neon;

pub(super) fn try_csc709_forward_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(simd) = Neon::try_new() {
        neon::csc709_forward_8x8_batch(simd, blocks);
        return true;
    }
    false
}

pub(super) fn try_csc709_inverse_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(simd) = Neon::try_new() {
        neon::csc709_inverse_8x8_batch(simd, blocks);
        return true;
    }
    false
}
