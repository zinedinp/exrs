//! SSE / SSSE3 (pulp `V2`) ZIP/RLE byte reconstruct; OpenEXR-faithful 16-byte
//! log-depth prefix sum.
//!
//! Reference: `openexr/src/lib/OpenEXRCore/internal_zip.c` (`reconstruct`).
//! Loads/stores go through `pulp::cast!` of fixed arrays so this stays under
//! `#![forbid(unsafe_code)]`.
//!
//! Encode-side predictor + interleave/separate SIMD ports are **commented out**
//! at the bottom: stage A/B was ~1.0× or a regression once TLS scratch +
//! copy-back are included (scalar pair loops in the parent module stay).

use std::convert::TryInto;

use pulp::x86::V2;

type M128 = std::arch::x86_64::__m128i;

/// OpenEXR `reconstruct`: in-place un-diff with a 16-byte log-depth prefix sum.
///
/// First byte is not differenced on encode; the SIMD loop still wants a uniform
/// `-128` bias on every lane, so we pre-bias `buf[0]` by `-128` and the loop's
/// per-lane `-128` cancels it back (wrapping). Carry into the next chunk is the
/// broadcast of the last reconstructed byte.
#[inline]
pub fn differences_to_samples(v2: V2, buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    let sse2 = v2.sse2;
    let ssse3 = v2.ssse3;

    // uint8_t buf[0] += (uint8_t)-128  ≡  wrapping_add(128)
    buffer[0] = buffer[0].wrapping_add(128);

    let c = sse2._mm_set1_epi8(-128);
    // Broadcast lane 15 to every lane (SSSE3 pshufb).
    let shuffle_mask = sse2._mm_set1_epi8(15);
    let mut v_prev = sse2._mm_setzero_si128();

    let n_chunks = buffer.len() / 16;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 16;
        let loaded: [u8; 16] = buffer[offset..offset + 16].try_into().unwrap();
        let mut d: M128 = pulp::cast!(loaded);

        d = sse2._mm_add_epi8(d, c);
        // Log-depth inclusive prefix sum within the 16-byte register.
        d = sse2._mm_add_epi8(d, sse2._mm_slli_si128::<1>(d));
        d = sse2._mm_add_epi8(d, sse2._mm_slli_si128::<2>(d));
        d = sse2._mm_add_epi8(d, sse2._mm_slli_si128::<4>(d));
        d = sse2._mm_add_epi8(d, sse2._mm_slli_si128::<8>(d));
        d = sse2._mm_add_epi8(d, v_prev);

        let stored: [u8; 16] = pulp::cast!(d);
        buffer[offset..offset + 16].copy_from_slice(&stored);

        v_prev = ssse3._mm_shuffle_epi8(d, shuffle_mask);
    }

    let prev_bytes: [u8; 16] = pulp::cast!(v_prev);
    let mut prev = prev_bytes[15];
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
pub fn differences_to_samples_from(v2: V2, buffer: &mut [u8], start: usize, mut prev: u8) {
    if start >= buffer.len() {
        return;
    }

    let sse2 = v2.sse2;
    let ssse3 = v2.ssse3;
    let c = sse2._mm_set1_epi8(-128);
    let shuffle_mask = sse2._mm_set1_epi8(15);
    let mut v_prev = sse2._mm_set1_epi8(prev as i8);

    let mut offset = start;
    while offset + 16 <= buffer.len() {
        let loaded: [u8; 16] = buffer[offset..offset + 16].try_into().unwrap();
        let mut d: M128 = pulp::cast!(loaded);

        d = sse2._mm_add_epi8(d, c);
        d = sse2._mm_add_epi8(d, sse2._mm_slli_si128::<1>(d));
        d = sse2._mm_add_epi8(d, sse2._mm_slli_si128::<2>(d));
        d = sse2._mm_add_epi8(d, sse2._mm_slli_si128::<4>(d));
        d = sse2._mm_add_epi8(d, sse2._mm_slli_si128::<8>(d));
        d = sse2._mm_add_epi8(d, v_prev);

        let stored: [u8; 16] = pulp::cast!(d);
        buffer[offset..offset + 16].copy_from_slice(&stored);

        v_prev = ssse3._mm_shuffle_epi8(d, shuffle_mask);
        offset += 16;
    }

    let prev_bytes: [u8; 16] = pulp::cast!(v_prev);
    prev = prev_bytes[15];
    for byte in &mut buffer[offset..] {
        let sample = prev.wrapping_add(*byte).wrapping_sub(128);
        *byte = sample;
        prev = sample;
    }
}

// ---------------------------------------------------------------------------
// Regression / non-winning kernels — kept commented for reference + history.
// encode predictor SIMD ~1.0×; interleave/separate SSE
// 0.75–0.97× once TLS scratch + copy-back are in the measured path. Production
// dispatch stays on the scalar pair loops in the parent module.
// ---------------------------------------------------------------------------

/*
/// Encode-side sibling: `diff[i] = sample[i] - sample[i-1] + 128`, first byte
/// left unchanged. Adjacent samples are independent once the previous sample
/// is known, so each 16-byte block is a single vector subtract + bias.
///
/// NOT SHIPPED: A/B ~1.0× vs the scalar 16-wide form (already memory-bound).
#[inline]
pub fn samples_to_differences(v2: V2, buffer: &mut [u8]) {
    if buffer.len() < 2 {
        return;
    }

    let sse2 = v2.sse2;
    let bias = sse2._mm_set1_epi8(-128); // +128 as i8 wrapping == -128
    let mut previous = buffer[0];

    let mut index = 1;
    let body = &mut buffer[1..];
    let n_chunks = body.len() / 16;

    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 16;
        let samples: [u8; 16] = body[offset..offset + 16].try_into().unwrap();
        let s: M128 = pulp::cast!(samples);

        let mut prev_lane = [0u8; 16];
        prev_lane[0] = previous;
        prev_lane[1..].copy_from_slice(&samples[..15]);
        let p: M128 = pulp::cast!(prev_lane);

        let d = sse2._mm_add_epi8(sse2._mm_sub_epi8(s, p), bias);
        let out: [u8; 16] = pulp::cast!(d);
        body[offset..offset + 16].copy_from_slice(&out);

        previous = samples[15];
        index = 1 + (chunk_index + 1) * 16;
    }

    let _ = index;
    for byte in &mut buffer[1 + n_chunks * 16..] {
        let sample = *byte;
        *byte = sample.wrapping_sub(previous).wrapping_add(128);
        previous = sample;
    }
}

/// OpenEXR `interleave`: write interleaved bytes into `out` from the two halves
/// of `separated`. `out.len() == separated.len()`.
///
/// NOT SHIPPED: scalar pair loop matches or beats SSE2 unpack once the final
/// copy-back from TLS scratch is included (0.75–0.97× for the SIMD port).
#[inline]
pub fn interleave_byte_blocks(v2: V2, separated: &[u8], out: &mut [u8]) {
    debug_assert_eq!(separated.len(), out.len());
    let len = separated.len();
    if len == 0 {
        return;
    }

    let sse2 = v2.sse2;
    let half = (len + 1) / 2;
    let first = &separated[..half];
    let second = &separated[half..];
    let pair_count = second.len();
    let n_vec = pair_count / 16;

    for i in 0..n_vec {
        let a: [u8; 16] = first[i * 16..i * 16 + 16].try_into().unwrap();
        let b: [u8; 16] = second[i * 16..i * 16 + 16].try_into().unwrap();
        let va: M128 = pulp::cast!(a);
        let vb: M128 = pulp::cast!(b);
        let lo = sse2._mm_unpacklo_epi8(va, vb);
        let hi = sse2._mm_unpackhi_epi8(va, vb);
        let lo_b: [u8; 16] = pulp::cast!(lo);
        let hi_b: [u8; 16] = pulp::cast!(hi);
        let out_off = i * 32;
        out[out_off..out_off + 16].copy_from_slice(&lo_b);
        out[out_off + 16..out_off + 32].copy_from_slice(&hi_b);
    }

    let mut t1 = n_vec * 16;
    let mut t2 = n_vec * 16;
    for i in (n_vec * 32)..len {
        if i % 2 == 0 {
            out[i] = first[t1];
            t1 += 1;
        } else {
            out[i] = second[t2];
            t2 += 1;
        }
    }
}

/// Inverse of interleave. NOT SHIPPED for the same reason as interleave.
#[inline]
pub fn separate_bytes_fragments(v2: V2, source: &[u8], separated: &mut [u8]) {
    debug_assert_eq!(source.len(), separated.len());
    let len = source.len();
    if len == 0 {
        return;
    }

    let ssse3 = v2.ssse3;
    let half = (len + 1) / 2;
    let (first_half, second_half) = separated.split_at_mut(half);

    let even_mask: M128 = pulp::cast!([
        0u8, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
    ]);
    let odd_mask: M128 = pulp::cast!([
        1u8, 3, 5, 7, 9, 11, 13, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
    ]);

    let pair_count = len / 2;
    let n_vec = pair_count / 16;

    for i in 0..n_vec {
        let base = i * 32;
        let a: [u8; 16] = source[base..base + 16].try_into().unwrap();
        let b: [u8; 16] = source[base + 16..base + 32].try_into().unwrap();
        let va: M128 = pulp::cast!(a);
        let vb: M128 = pulp::cast!(b);

        let even_a = ssse3._mm_shuffle_epi8(va, even_mask);
        let even_b = ssse3._mm_shuffle_epi8(vb, even_mask);
        let odd_a = ssse3._mm_shuffle_epi8(va, odd_mask);
        let odd_b = ssse3._mm_shuffle_epi8(vb, odd_mask);

        let ea: [u8; 16] = pulp::cast!(even_a);
        let eb: [u8; 16] = pulp::cast!(even_b);
        let oa: [u8; 16] = pulp::cast!(odd_a);
        let ob: [u8; 16] = pulp::cast!(odd_b);

        let mut evens = [0u8; 16];
        let mut odds = [0u8; 16];
        evens[..8].copy_from_slice(&ea[..8]);
        evens[8..].copy_from_slice(&eb[..8]);
        odds[..8].copy_from_slice(&oa[..8]);
        odds[8..].copy_from_slice(&ob[..8]);

        first_half[i * 16..i * 16 + 16].copy_from_slice(&evens);
        second_half[i * 16..i * 16 + 16].copy_from_slice(&odds);
    }

    let mut t1 = n_vec * 16;
    let mut t2 = n_vec * 16;
    for i in (n_vec * 32)..len {
        if i % 2 == 0 {
            first_half[t1] = source[i];
            t1 += 1;
        } else {
            second_half[t2] = source[i];
            t2 += 1;
        }
    }
}
*/
