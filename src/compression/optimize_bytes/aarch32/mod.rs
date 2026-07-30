//! Runtime 32-bit ARM dispatch for ZIP/RLE byte reconstruct.
//!
//! ## Production paths
//!
//! | Build | Kernel |
//! |-------|--------|
//! | Default (stable) | [`super::portable_wide16`] — pure-Rust OpenEXR 16B tree |
//! | `arm-neon` + nightly | real NEON via local pulp fork (`pulp::arm::Neon`) |
//!
//! OpenEXR's C reference only ships NEON for AArch64 (`IMF_HAVE_NEON_ARM64`);
//! this module is our AArch32 extension of the same algorithm.
//!
//! **Not tested on real 32-bit ARM hardware in this tree** — cross-compile /
//! unit tests only. See [`neon`] (gated on `arm-neon`).

#[cfg(feature = "arm-neon")]
#[doc(hidden)]
pub mod neon;

/// Try NEON (if `arm-neon` and the CPU reports neon) then portable 16-byte
/// tree. Always returns `true` (a non-scalar path always runs).
#[inline]
pub(super) fn try_differences_to_samples(buffer: &mut [u8]) -> bool {
    #[cfg(feature = "arm-neon")]
    if let Some(simd) = pulp::arm::Neon::try_new() {
        neon::differences_to_samples(simd, buffer);
        return true;
    }

    super::portable_wide16::differences_to_samples(buffer);
    true
}
