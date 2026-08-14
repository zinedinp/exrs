//! SSE2/SSSE3 (miraculix `Sse2`/`Ssse3`) ZIP/RLE reconstruct: OpenEXR 16-byte log-depth prefix sum.
//! Ref: `openexr/src/lib/OpenEXRCore/internal_zip.c` (`reconstruct`).
//! Ops use plain `[u8; N]` arrays (loadu/storeu inside); stays under `#![forbid(unsafe_code)]`.

use miraculix::x86::ops::sse::sse2::Sse2;
use miraculix::x86::ops::sse::ssse3::Ssse3;

/// Reinterpret `[u8; 16]` as `[i8; 16]` for `Ssse3::shuffle_i8x16` (byte
/// shuffle ignores signedness; `as` same-width, no `unsafe`). Shared with
/// avx2/avx512 carry-broadcast (`pshufb`).
pub(super) fn to_i8x16(a: [u8; 16]) -> [i8; 16] {
    a.map(|x| x as i8)
}

pub(super) fn to_u8x16(a: [i8; 16]) -> [u8; 16] {
    a.map(|x| x as u8)
}

/// OpenEXR `reconstruct`: in-place un-diff, 16-byte log-depth prefix sum.
/// Encode leaves byte 0 undifferenced; pre-bias `buf[0]` by 128 so the loop's
/// per-lane `-128` cancels (wrapping). Next-chunk carry = broadcast last sample.
#[inline]
pub fn differences_to_samples(sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    // OpenEXR: buf[0] += (uint8_t)-128
    buffer[0] = buffer[0].wrapping_add(128);

    let c = [128u8; 16];
    // Lane 15 -> all lanes (pshufb).
    let shuffle_mask = to_i8x16([15u8; 16]);
    let mut v_prev = [0u8; 16];

    let n_chunks = buffer.len() / 16;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 16;
        let mut d: [u8; 16] = buffer[offset..offset + 16].try_into().unwrap();

        d = sse2.add_u8x16(d, c);
        // Inclusive log-depth prefix sum in the 16-byte register.
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

/// Continue from `start` with known carry `prev` (AVX2/AVX-512 rem; no
/// first-byte pre-bias, caller already integrated earlier bytes).
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
