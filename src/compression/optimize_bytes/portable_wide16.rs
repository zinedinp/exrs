//! Portable 16-byte OpenEXR log-depth `reconstruct`.
//!
//! Same algorithm as x86 SSE (`_mm_slli_si128` + `paddb`) and aarch64 NEON
//! (`vextq_u8` + `vaddq_u8`): Hillis–Steele inclusive prefix sum inside each
//! 16-byte chunk, then broadcast carry into the next chunk.
//!
//! Used as the **default 32-bit ARM production path** on stable (no unstable
//! stdarch). With exrs feature `arm-neon` + nightly, production prefers real
//! NEON via the local pulp fork (`pulp::arm::Neon` —> see `arm/neon.rs`); this
//! portable tree remains the fallback when Neon is unavailable.
//!
//! **Not tested on real 32-bit ARM hardware in this tree.** Correctness is
//! unit-tested on the host (bit-exact vs scalar). Expect a win on superscalar
//! cores; on tiny in-order cores without NEON the pair-ILP scalar may be
//! similar

use std::convert::TryInto;

/// OpenEXR-faithful 16-byte log-depth reconstruct (portable / soft form).
#[inline]
pub fn differences_to_samples(buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    // uint8_t buf[0] += (uint8_t)-128  ≡  wrapping_add(128)
    buffer[0] = buffer[0].wrapping_add(128);

    let mut prev = 0u8;
    let n_chunks = buffer.len() / 16;

    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 16;
        let mut d: [u8; 16] = buffer[offset..offset + 16].try_into().unwrap();

        // Per-lane −128 bias (wrapping). Same as XOR 0x80 per byte.
        for b in &mut d {
            *b = b.wrapping_add(128);
        }

        // Log-depth inclusive prefix sum: d[i] += d[i-k] for k in {1,2,4,8}.
        // Matches SIMD `d = add(d, byte_shift_left(d, k))` (zeros enter low).
        for &k in &[1usize, 2, 4, 8] {
            let src = d;
            for i in k..16 {
                d[i] = d[i].wrapping_add(src[i - k]);
            }
        }

        // Add carry from previous chunk (broadcast of last sample).
        for b in &mut d {
            *b = b.wrapping_add(prev);
        }
        prev = d[15];

        buffer[offset..offset + 16].copy_from_slice(&d);
    }

    for byte in &mut buffer[n_chunks * 16..] {
        let sample = prev.wrapping_add(*byte).wrapping_sub(128);
        *byte = sample;
        prev = sample;
    }
}
