//! AVX-512 (pulp `V4`) ZIP/RLE byte reconstruct → 64-byte chunks.
//!
//! - **lane + hierarchical tail** ([`differences_to_samples`]): **production**
//!   when V4 is available — four independent 16-byte OpenEXR trees in the ZMM
//!   + cascade carry lane0→1→2→3; rem via AVX2 32 → SSE 16 → scalar.
//!   Stage A/B: ~1.33–1.37× AVX2 lane / ~6.3–6.6× scalar on large buffers.
//! - **masked** ([`differences_to_samples_lane_masked`]): same main loop; final
//!   partial chunk is a zero-padded 64-byte vector (store live prefix only).
//!   Kept for re-bench; ~tied with hierarchical on large sizes.
//!
//! `_mm512_bslli_epi128` is 128-bit-lane-local (four 16-byte lanes), same
//! reason AVX2 lane beat full 32-wide Hillis–Steele.

use std::convert::TryInto;

use pulp::x86::{V3, V4};

use super::avx2;

type M512 = std::arch::x86_64::__m512i;
type M128 = std::arch::x86_64::__m128i;

/// Production AVX-512 entry: lane-prefix with hierarchical remainder.
#[inline]
pub fn differences_to_samples(v4: V4, buffer: &mut [u8]) {
    differences_to_samples_lane(v4, buffer);
}

/// AVX-512 lane-prefix reconstruct with hierarchical remainder.
#[inline]
pub fn differences_to_samples_lane(v4: V4, buffer: &mut [u8]) {
    run_lane(v4, buffer, Tail::Hierarchical);
}

/// Same 64-byte lane body; remainder is one zero-padded ZMM pass (store live
/// bytes only).
#[inline]
pub fn differences_to_samples_lane_masked(v4: V4, buffer: &mut [u8]) {
    run_lane(v4, buffer, Tail::Padded);
}

#[derive(Clone, Copy)]
enum Tail {
    Hierarchical,
    Padded,
}

#[inline]
fn run_lane(v4: V4, buffer: &mut [u8], tail: Tail) {
    if buffer.is_empty() {
        return;
    }

    let bw = v4.avx512bw;
    let f = v4.avx512f;
    let ssse3 = v4.ssse3;
    let sse2 = v4.sse2;

    buffer[0] = buffer[0].wrapping_add(128);

    let c: M512 = f._mm512_set1_epi8(-128);
    let shuffle15 = sse2._mm_set1_epi8(15);
    let mut v_prev: M512 = f._mm512_setzero_si512();

    let n_chunks = buffer.len() / 64;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 64;
        let loaded: [u8; 64] = buffer[offset..offset + 64].try_into().unwrap();
        let mut d: M512 = pulp::cast!(loaded);

        d = process_chunk(v4, d, c, shuffle15, v_prev);

        let stored: [u8; 64] = pulp::cast!(d);
        buffer[offset..offset + 64].copy_from_slice(&stored);

        // Next carry = broadcast of d[63].
        let last128 = f._mm512_extracti32x4_epi32::<3>(d);
        let last = ssse3._mm_shuffle_epi8(last128, shuffle15);
        v_prev = bw._mm512_broadcastb_epi8(last);
    }

    let done = n_chunks * 64;
    match tail {
        Tail::Hierarchical => finish_hierarchical(v4, done, v_prev, buffer),
        Tail::Padded => finish_padded(v4, done, v_prev, c, shuffle15, buffer),
    }
}

/// Within-lane log-depth + cascade fixups + previous-chunk carry.
#[inline]
fn process_chunk(v4: V4, mut d: M512, c: M512, shuffle15: M128, v_prev: M512) -> M512 {
    let bw = v4.avx512bw;
    let f = v4.avx512f;
    let ssse3 = v4.ssse3;

    d = bw._mm512_add_epi8(d, c);
    // Four independent 16-byte OpenEXR trees (bslli is lane-local).
    d = bw._mm512_add_epi8(d, bw._mm512_bslli_epi128::<1>(d));
    d = bw._mm512_add_epi8(d, bw._mm512_bslli_epi128::<2>(d));
    d = bw._mm512_add_epi8(d, bw._mm512_bslli_epi128::<4>(d));
    d = bw._mm512_add_epi8(d, bw._mm512_bslli_epi128::<8>(d));

    // Cascade: lane i (i>0) += last byte of reconstructed lane i-1.
    // extracti32x4 index selects 128-bit lane 0..3.
    let l0 = f._mm512_extracti32x4_epi32::<0>(d);
    let l1 = f._mm512_extracti32x4_epi32::<1>(d);
    let l2 = f._mm512_extracti32x4_epi32::<2>(d);
    let l3 = f._mm512_extracti32x4_epi32::<3>(d);

    let c0 = ssse3._mm_shuffle_epi8(l0, shuffle15);
    let l1 = v4.sse2._mm_add_epi8(l1, c0);
    let c1 = ssse3._mm_shuffle_epi8(l1, shuffle15);
    let l2 = v4.sse2._mm_add_epi8(l2, c1);
    let c2 = ssse3._mm_shuffle_epi8(l2, shuffle15);
    let l3 = v4.sse2._mm_add_epi8(l3, c2);

    let mut out = f._mm512_setzero_si512();
    out = f._mm512_inserti32x4::<0>(out, l0);
    out = f._mm512_inserti32x4::<1>(out, l1);
    out = f._mm512_inserti32x4::<2>(out, l2);
    out = f._mm512_inserti32x4::<3>(out, l3);

    bw._mm512_add_epi8(out, v_prev)
}

/// Remainder: AVX2 (≥32) → SSE (≥16) → scalar.
///
/// `V4: Deref<Target = V3>` (and V3 → V2), so we reuse the already-checked token
/// instead of a second `try_new` / CPUID — same pattern as DWA AVX-512 fused decode.
#[inline]
fn finish_hierarchical(v4: V4, done: usize, v_prev: M512, buffer: &mut [u8]) {
    if done >= buffer.len() {
        return;
    }
    let v3: V3 = *v4;
    if done == 0 {
        // No full 64-byte chunk: undo pre-bias and drop to AVX2/SSE path.
        buffer[0] = buffer[0].wrapping_sub(128);
        avx2::differences_to_samples_lane_sse_tail(v3, buffer);
        return;
    }

    let prev = last_byte_broadcast(v_prev);
    avx2::differences_to_samples_from(v3, buffer, done, prev);
}

/// Remainder: zero-pad to 64, one process_chunk, store live prefix only.
#[inline]
fn finish_padded(
    v4: V4,
    done: usize,
    v_prev: M512,
    c: M512,
    shuffle15: M128,
    buffer: &mut [u8],
) {
    if done >= buffer.len() {
        return;
    }
    let rem = buffer.len() - done;
    debug_assert!(rem < 64);

    let mut tmp = [0u8; 64];
    tmp[..rem].copy_from_slice(&buffer[done..]);
    let loaded: M512 = pulp::cast!(tmp);
    // When done==0, v_prev is zero (first-byte pre-bias already applied).
    let d = process_chunk(v4, loaded, c, shuffle15, v_prev);
    let stored: [u8; 64] = pulp::cast!(d);
    buffer[done..].copy_from_slice(&stored[..rem]);
}

#[inline]
fn last_byte_broadcast(v_prev: M512) -> u8 {
    // v_prev is a full broadcast of the last reconstructed sample.
    let bytes: [u8; 64] = pulp::cast!(v_prev);
    bytes[0]
}
