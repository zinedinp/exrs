//! Runtime 32-bit ARM SIMD dispatch for the DWA DCT.

#[cfg(feature = "arm-neon")]
#[doc(hidden)]
pub mod neon;

#[cfg(feature = "arm-neon")]
pub(super) fn try_dct_forward_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [f32; 64]>,
{
    if let Some(simd) = pulp::aarch32::Neon::try_new() {
        for data in blocks {
            neon::dct_forward_8x8(simd, data);
        }
        return true;
    }
    false
}

#[cfg(not(feature = "arm-neon"))]
pub(super) fn try_dct_forward_8x8_batch<'a, I>(_blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [f32; 64]>,
{
    false
}

#[cfg(feature = "arm-neon")]
pub(super) fn try_dct_inverse_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [f32; 64]>,
{
    if let Some(simd) = pulp::aarch32::Neon::try_new() {
        for data in blocks {
            neon::dct_inverse_8x8(simd, data);
        }
        return true;
    }
    false
}

#[cfg(not(feature = "arm-neon"))]
pub(super) fn try_dct_inverse_8x8_batch<'a, I>(_blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [f32; 64]>,
{
    false
}
