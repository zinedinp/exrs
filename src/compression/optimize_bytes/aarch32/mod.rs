//! Runtime 32-bit ARM dispatch for ZIP/RLE byte reconstruct.
//!//!
//! | `arm-neon` + nightly, real NEON via local pulp fork (`pulp::aarch32::Neon`) |
//!
//! OpenEXR's C reference only ships NEON for AArch64 (`IMF_HAVE_NEON_ARM64`);
//! this module is our AArch32 extension of the same algorithm.
//!
//! **Not tested on real 32-bit ARM hardware in this tree;
//!  cross-compile tests only.

#[cfg(feature = "arm-neon")]
#[doc(hidden)]
pub mod neon;

/// Try NEON (if `arm-neon` and the CPU reports neon) then portable 16-byte
/// tree. Always returns `true` (a non-scalar path always runs).
#[inline]
pub(super) fn try_differences_to_samples(buffer: &mut [u8]) -> bool {
    #[cfg(feature = "arm-neon")]
    if let Some(simd) = pulp::aarch32::Neon::try_new() {
        neon::differences_to_samples(simd, buffer);
        return true;
    }

    super::portable_wide16::differences_to_samples(buffer);
    true
}
