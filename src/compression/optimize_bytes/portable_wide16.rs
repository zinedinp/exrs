//! Portable 16-byte OpenEXR log-depth `reconstruct`.
//!
//! Same algorithm as x86 SSE (`_mm_slli_si128` + `paddb`). Production
//! fallback on every non-x86 architecture.

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
