//! Runtime aarch64 SIMD dispatch for ZIP/RLE byte reconstruct.
//!
//! Production: pulp `Neon` 16-byte OpenEXR-faithful log-depth prefix sum
//! (same algorithm as x86 `V2` SSE and OpenEXR `IMF_HAVE_NEON_ARM64`).
//!
//! **Not tested on real ARM/NEON hardware in this tree** — see [`neon`].

use pulp::aarch64::Neon;

// public only for benchmarking / correctness tests
#[doc(hidden)]
pub mod neon;

/// Try NEON reconstruct, else portable 16-byte tree. Returns `true` if a
/// non-scalar path ran.
#[inline]
pub(super) fn try_differences_to_samples(buffer: &mut [u8]) -> bool {
    if let Some(simd) = Neon::try_new() {
        neon::differences_to_samples(simd, buffer);
        return true;
    }
    // Rare (NEON is baseline on AArch64); keep OpenEXR algorithm without
    // falling all the way to the serial pair-ILP scalar.
    super::portable_wide16::differences_to_samples(buffer);
    true
}
