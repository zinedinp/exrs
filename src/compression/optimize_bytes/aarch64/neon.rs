//! NEON (pulp `Neon`) ZIP/RLE byte reconstruct; OpenEXR-faithful 16-byte
//! log-depth prefix sum.
//!
//! Reference: `openexr/src/lib/OpenEXRCore/internal_zip.c` (`reconstruct` under
//! `IMF_HAVE_NEON_ARM64`). Byte shifts use `vextq_u8(zero, d, 16-n)` — the NEON
//! equivalent of x86 `_mm_slli_si128::<n>`; carry broadcast uses `vqtbl1q_u8`
//! with an all-15 index (same role as SSSE3 `pshufb`).
//!
//! **Not tested on real ARM/NEON hardware in this tree.** Correctness vs the
//! scalar reference is unit-tested when the crate is built for `aarch64`
//! (cross-compile / CI). No wall-time A/B was run on Apple Silicon or server
//! Arm; treat stage throughput numbers as expected-from-OpenEXR only.

use std::convert::TryInto;

use core::arch::aarch64::uint8x16_t;
use pulp::aarch64::Neon;

/// OpenEXR `reconstruct` (NEON): in-place un-diff with a 16-byte log-depth
/// prefix sum.
///
/// First byte is not differenced on encode; the SIMD loop still wants a uniform
/// `-128` bias on every lane, so we pre-bias `buf[0]` by `-128` and the loop's
/// per-lane `-128` cancels it back (wrapping). Carry into the next chunk is the
/// broadcast of the last reconstructed byte.
#[inline]
pub fn differences_to_samples(simd: Neon, buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    let neon = simd.neon;

    // uint8_t buf[0] += (uint8_t)-128  ≡  wrapping_add(128)
    buffer[0] = buffer[0].wrapping_add(128);

    // OpenEXR: `vdupq_n_u8(-128)` — C converts −128 to `uint8_t` 128 / 0x80.
    let c = neon.vdupq_n_u8(128);
    // Broadcast lane 15 to every lane (`vqtbl1q_u8` with all indices = 15).
    let shuffle_mask = neon.vdupq_n_u8(15);
    let zero = neon.vdupq_n_u8(0);
    let mut v_prev = zero;

    let n_chunks = buffer.len() / 16;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 16;
        let loaded: [u8; 16] = buffer[offset..offset + 16].try_into().unwrap();
        let mut d: uint8x16_t = pulp::cast!(loaded);

        d = neon.vaddq_u8(d, c);
        // Log-depth inclusive prefix sum within the 16-byte register.
        // `vextq_u8(zero, d, 16-n)` shifts bytes left by `n` (zeros enter low).
        d = neon.vaddq_u8(d, neon.vextq_u8::<15>(zero, d));
        d = neon.vaddq_u8(d, neon.vextq_u8::<14>(zero, d));
        d = neon.vaddq_u8(d, neon.vextq_u8::<12>(zero, d));
        d = neon.vaddq_u8(d, neon.vextq_u8::<8>(zero, d));
        d = neon.vaddq_u8(d, v_prev);

        let stored: [u8; 16] = pulp::cast!(d);
        buffer[offset..offset + 16].copy_from_slice(&stored);

        v_prev = neon.vqtbl1q_u8(d, shuffle_mask);
    }

    let prev_bytes: [u8; 16] = pulp::cast!(v_prev);
    let mut prev = prev_bytes[15];
    for byte in &mut buffer[n_chunks * 16..] {
        let sample = prev.wrapping_add(*byte).wrapping_sub(128);
        *byte = sample;
        prev = sample;
    }
}

