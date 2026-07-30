//! 32-bit ARM NEON ZIP/RLE byte reconstruct via local pulp fork
//! (`pulp::arm::Neon`).
//!
//! Same OpenEXR log-depth algorithm as aarch64 NEON / x86 SSE:
//! 16-byte Hillis–Steele inclusive prefix sum with carry broadcast into the
//! next chunk. Reference: `openexr/src/lib/OpenEXRCore/internal_zip.c`
//! (`reconstruct` under `IMF_HAVE_NEON_ARM64`) — OpenEXR only ships this for
//! AArch64; this is the same math on AArch32 NEON.
//!
//! ## Build requirements
//!
//! - exrs feature `arm-neon` (enables pulp `nightly`)
//! - **nightly** toolchain — Rust's 32-bit `core::arch::arm` NEON surface and
//!   the `neon` target feature are still unstable on stable
//!
//! Example:
//! ```text
//! cargo +nightly check --target armv7-unknown-linux-gnueabihf --features arm-neon
//! ```
//!
//! Uses pulp high-level `u8x16` helpers so this crate never touches
//! `core::arch::arm` (keeps `#![forbid(unsafe_code)]`).
//!
//! **Not tested on real 32-bit ARM hardware in this tree.** Correctness vs
//! scalar is unit-tested when built for `arm` with `arm-neon` (cross /
//! qemu / CI). Without that feature, production uses [`super::super::portable_wide16`].

use std::convert::TryInto;

use pulp::arm::Neon;
use pulp::u8x16;

/// OpenEXR-style 16-byte log-depth reconstruct using pulp's arm Neon token.
///
/// First byte is not differenced on encode; the SIMD loop still wants a uniform
/// `−128` bias on every lane, so we pre-bias `buf[0]` by `−128` and the loop's
/// per-lane `−128` cancels it back (wrapping). Carry into the next chunk is the
/// broadcast of the last reconstructed byte (lane 15).
#[inline]
pub fn differences_to_samples(simd: Neon, buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    // uint8_t buf[0] += (uint8_t)-128  ≡  wrapping_add(128)
    buffer[0] = buffer[0].wrapping_add(128);

    // OpenEXR: `vdupq_n_u8(-128)` — C converts −128 to `uint8_t` 128 / 0x80.
    let c = simd.splat_u8x16(128);
    let zero = simd.splat_u8x16(0);
    let mut v_prev = zero;

    let n_chunks = buffer.len() / 16;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 16;
        let loaded: [u8; 16] = buffer[offset..offset + 16].try_into().unwrap();
        let mut d: u8x16 = pulp::cast!(loaded);

        d = simd.wrapping_add_u8x16(d, c);
        // Log-depth inclusive prefix sum within the 16-byte register.
        // `ext(zero, d, 16-n)` shifts bytes left by `n` (zeros enter low) —
        // same as AArch64 `vextq_u8(zero, d, 16-n)` / x86 `_mm_slli_si128::<n>`.
        d = simd.wrapping_add_u8x16(d, simd.ext_u8x16::<15>(zero, d));
        d = simd.wrapping_add_u8x16(d, simd.ext_u8x16::<14>(zero, d));
        d = simd.wrapping_add_u8x16(d, simd.ext_u8x16::<12>(zero, d));
        d = simd.wrapping_add_u8x16(d, simd.ext_u8x16::<8>(zero, d));
        d = simd.wrapping_add_u8x16(d, v_prev);

        let stored: [u8; 16] = pulp::cast!(d);
        buffer[offset..offset + 16].copy_from_slice(&stored);

        // Broadcast lane 15 → all lanes for the next chunk's carry.
        // Prefer hardware `vdupq_lane` over soft `vqtbl1q` (AArch32 has no
        // 128-bit table-lookup instruction).
        v_prev = simd.broadcast_lane_u8x16::<15>(d);
    }

    let prev_bytes: [u8; 16] = pulp::cast!(v_prev);
    let mut prev = prev_bytes[15];
    for byte in &mut buffer[n_chunks * 16..] {
        let sample = prev.wrapping_add(*byte).wrapping_sub(128);
        *byte = sample;
        prev = sample;
    }
}
