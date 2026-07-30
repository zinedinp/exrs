//! Runtime 32-bit ARM SIMD dispatch for the DWA CSC transform.

#[cfg(feature = "arm-neon")]
#[doc(hidden)]
pub mod neon;

#[cfg(feature = "arm-neon")]
pub(super) fn try_csc709_forward_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(simd) = pulp::aarch32::Neon::try_new() {
        neon::csc709_forward_8x8_batch(simd, blocks);
        return true;
    }
    false
}

#[cfg(not(feature = "arm-neon"))]
pub(super) fn try_csc709_forward_8x8_batch<'a, I>(_blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    false
}

#[cfg(feature = "arm-neon")]
pub(super) fn try_csc709_inverse_8x8_batch<'a, I>(blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    if let Some(simd) = pulp::aarch32::Neon::try_new() {
        neon::csc709_inverse_8x8_batch(simd, blocks);
        return true;
    }
    false
}

#[cfg(not(feature = "arm-neon"))]
pub(super) fn try_csc709_inverse_8x8_batch<'a, I>(_blocks: &mut I) -> bool
where
    I: Iterator<Item = &'a mut [[f32; 64]; 3]>,
{
    false
}
