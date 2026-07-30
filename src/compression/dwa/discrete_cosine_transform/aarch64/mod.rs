//! Runtime aarch64 SIMD dispatch for the DWA DCT.

use pulp::aarch64::Neon;

// public only for benchmarking / correctness tests
#[doc(hidden)]
pub mod neon;

pub(super) fn try_dct_forward_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [f32; 64]>,
{
    if let Some(simd) = Neon::try_new() {
        for data in blocks {
            neon::dct_forward_8x8(simd, data);
        }
        return true;
    }
    false
}

pub(super) fn try_dct_inverse_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [f32; 64]>,
{
    if let Some(simd) = Neon::try_new() {
        for data in blocks {
            neon::dct_inverse_8x8(simd, data);
        }
        return true;
    }
    false
}
