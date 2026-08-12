//! SSE2 / SSSE3 (miraculix `Sse2`/`Ssse3`) ZIP/RLE byte reconstruct;
//! OpenEXR-faithful 16-byte log-depth prefix sum.
//!
//! Reference: `openexr/src/lib/OpenEXRCore/internal_zip.c` (`reconstruct`).
//! miraculix's ops take/return plain `[u8; N]` arrays (loadu/storeu happen
//! inside the op itself), so this stays under `#![forbid(unsafe_code)]`
//! without needing a cast layer.

use miraculix::x86::ops::sse::sse2::Sse2;
use miraculix::x86::ops::sse::ssse3::Ssse3;

/// `Ssse3::shuffle_i8x16` takes/returns `[i8; 16]`; our data is `[u8; 16]`.
/// Byte-shuffle doesn't care about signedness, so this is a pure reinterpret
/// - safe (no `unsafe`), and `as i8` on same-width integers compiles away.
/// Shared with `avx2`/`avx512`'s cross-lane carry-broadcast, which does the
/// same 128-bit `pshufb` dance one or more lanes at a time.
pub(super) fn to_i8x16(a: [u8; 16]) -> [i8; 16] {
    a.map(|x| x as i8)
}

pub(super) fn to_u8x16(a: [i8; 16]) -> [u8; 16] {
    a.map(|x| x as u8)
}

/// OpenEXR `reconstruct`: in-place un-diff with a 16-byte log-depth prefix sum.
///
/// First byte is not differenced on encode; the SIMD loop still wants a uniform
/// `-128` bias on every lane, so we pre-bias `buf[0]` by `-128` and the loop's
/// per-lane `-128` cancels it back (wrapping). Carry into the next chunk is the
/// broadcast of the last reconstructed byte.
#[inline]
pub fn differences_to_samples(sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    // uint8_t buf[0] += (uint8_t)-128  ≡  wrapping_add(128)
    buffer[0] = buffer[0].wrapping_add(128);

    let c = [128u8; 16];
    // Broadcast lane 15 to every lane (SSSE3 pshufb).
    let shuffle_mask = to_i8x16([15u8; 16]);
    let mut v_prev = [0u8; 16];

    let n_chunks = buffer.len() / 16;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 16;
        let mut d: [u8; 16] = buffer[offset..offset + 16].try_into().unwrap();

        d = sse2.add_u8x16(d, c);
        // Log-depth inclusive prefix sum within the 16-byte register.
        d = sse2.add_u8x16(d, sse2.slli_u8x16::<1>(d));
        d = sse2.add_u8x16(d, sse2.slli_u8x16::<2>(d));
        d = sse2.add_u8x16(d, sse2.slli_u8x16::<4>(d));
        d = sse2.add_u8x16(d, sse2.slli_u8x16::<8>(d));
        d = sse2.add_u8x16(d, v_prev);

        buffer[offset..offset + 16].copy_from_slice(&d);

        v_prev = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(d), shuffle_mask));
    }

    let mut prev = v_prev[15];
    for byte in &mut buffer[n_chunks * 16..] {
        let sample = prev.wrapping_add(*byte).wrapping_sub(128);
        *byte = sample;
        prev = sample;
    }
}

/// Continue reconstruct from `start` with known previous sample `prev`.
///
/// Used as the hierarchical remainder after AVX2/AVX-512 full chunks (no
/// first-byte pre-bias — caller already integrated earlier bytes).
#[inline]
pub fn differences_to_samples_from(
    sse2: Sse2,
    ssse3: Ssse3,
    buffer: &mut [u8],
    start: usize,
    mut prev: u8,
) {
    if start >= buffer.len() {
        return;
    }

    let c = [128u8; 16];
    let shuffle_mask = to_i8x16([15u8; 16]);
    let mut v_prev = [prev; 16];

    let mut offset = start;
    while offset + 16 <= buffer.len() {
        let mut d: [u8; 16] = buffer[offset..offset + 16].try_into().unwrap();

        d = sse2.add_u8x16(d, c);
        d = sse2.add_u8x16(d, sse2.slli_u8x16::<1>(d));
        d = sse2.add_u8x16(d, sse2.slli_u8x16::<2>(d));
        d = sse2.add_u8x16(d, sse2.slli_u8x16::<4>(d));
        d = sse2.add_u8x16(d, sse2.slli_u8x16::<8>(d));
        d = sse2.add_u8x16(d, v_prev);

        buffer[offset..offset + 16].copy_from_slice(&d);

        v_prev = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(d), shuffle_mask));
        offset += 16;
    }

    prev = v_prev[15];
    for byte in &mut buffer[offset..] {
        let sample = prev.wrapping_add(*byte).wrapping_sub(128);
        *byte = sample;
        prev = sample;
    }
}
