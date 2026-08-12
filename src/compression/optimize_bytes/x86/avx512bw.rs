//! AVX-512 (miraculix `Avx512f`+`Avx512Bw`+`Avx2`+`Sse2`+`Ssse3`) ZIP/RLE byte
//! reconstruct → 64-byte chunks.
//!
//! - **lane + hierarchical tail** ([`differences_to_samples`]): **production**
//!   when AVX-512 is available — four independent 16-byte OpenEXR trees in the
//!   ZMM + cascade carry lane0→1→2→3; rem via AVX2 32 → SSE 16 → scalar.
//!   Stage A/B: ~1.33–1.37× AVX2 lane / ~6.3–6.6× scalar on large buffers.
//! - **masked** ([`differences_to_samples_lane_masked`]): same main loop; final
//!   partial chunk is a zero-padded 64-byte vector (store live prefix only).
//!   Kept for re-bench; ~tied with hierarchical on large sizes.
//!
//! `Avx512Bw::bslli_u8x64` is 128-bit-lane-local (four 16-byte lanes), same
//! reason AVX2 lane beat full 32-wide Hillis–Steele.

use miraculix::x86::ops::avx::avx2::Avx2;
use miraculix::x86::ops::avx512::avx512bw::Avx512Bw;
use miraculix::x86::ops::avx512::avx512f::Avx512f;
use miraculix::x86::ops::sse::sse2::Sse2;
use miraculix::x86::ops::sse::ssse3::Ssse3;

use super::avx2 as avx2_mod;
use super::sse::{to_i8x16, to_u8x16};

/// Production AVX-512 entry: lane-prefix with hierarchical remainder.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn differences_to_samples(
    f: Avx512f,
    bw: Avx512Bw,
    avx2: Avx2,
    sse2: Sse2,
    ssse3: Ssse3,
    buffer: &mut [u8],
) {
    differences_to_samples_lane(f, bw, avx2, sse2, ssse3, buffer);
}

/// AVX-512 lane-prefix reconstruct with hierarchical remainder.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn differences_to_samples_lane(
    f: Avx512f,
    bw: Avx512Bw,
    avx2: Avx2,
    sse2: Sse2,
    ssse3: Ssse3,
    buffer: &mut [u8],
) {
    run_lane(f, bw, avx2, sse2, ssse3, buffer, Tail::Hierarchical);
}

/// Same 64-byte lane body; remainder is one zero-padded ZMM pass (store live
/// bytes only).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn differences_to_samples_lane_masked(
    f: Avx512f,
    bw: Avx512Bw,
    avx2: Avx2,
    sse2: Sse2,
    ssse3: Ssse3,
    buffer: &mut [u8],
) {
    run_lane(f, bw, avx2, sse2, ssse3, buffer, Tail::Padded);
}

#[derive(Clone, Copy)]
enum Tail {
    Hierarchical,
    Padded,
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn run_lane(
    f: Avx512f,
    bw: Avx512Bw,
    avx2: Avx2,
    sse2: Sse2,
    ssse3: Ssse3,
    buffer: &mut [u8],
    tail: Tail,
) {
    if buffer.is_empty() {
        return;
    }

    if buffer.len() < 64 {
        avx2_mod::differences_to_samples_lane_sse_tail(avx2, sse2, ssse3, buffer);
        return;
    }

    buffer[0] = buffer[0].wrapping_add(128);

    let c = [128u8; 64];
    let shuffle15 = to_i8x16([15u8; 16]);
    let mut v_prev = [0u8; 64];

    let n_chunks = buffer.len() / 64;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 64;
        let mut d: [u8; 64] = buffer[offset..offset + 64].try_into().unwrap();

        d = process_chunk(f, bw, sse2, ssse3, d, c, shuffle15, v_prev);

        buffer[offset..offset + 64].copy_from_slice(&d);

        // Next carry = broadcast of d[63].
        let last128 = f.extract_u8x16_from_x64::<3>(d);
        let last = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(last128), shuffle15));
        v_prev = bw.broadcast_u8x64(last[0]);
    }

    let done = n_chunks * 64;
    match tail {
        Tail::Hierarchical => finish_hierarchical(avx2, sse2, ssse3, done, v_prev[0], buffer),
        Tail::Padded => finish_padded(f, bw, sse2, ssse3, done, v_prev, c, shuffle15, buffer),
    }
}

/// Within-lane log-depth + cascade fixups + previous-chunk carry.
#[inline]
fn process_chunk(
    f: Avx512f,
    bw: Avx512Bw,
    sse2: Sse2,
    ssse3: Ssse3,
    mut d: [u8; 64],
    c: [u8; 64],
    shuffle15: [i8; 16],
    v_prev: [u8; 64],
) -> [u8; 64] {
    d = bw.add_u8x64(d, c);
    // Four independent 16-byte OpenEXR trees (bslli is lane-local).
    d = bw.add_u8x64(d, bw.bslli_u8x64::<1>(d));
    d = bw.add_u8x64(d, bw.bslli_u8x64::<2>(d));
    d = bw.add_u8x64(d, bw.bslli_u8x64::<4>(d));
    d = bw.add_u8x64(d, bw.bslli_u8x64::<8>(d));

    // Cascade: lane i (i>0) += last byte of reconstructed lane i-1.
    // `extract_u8x16_from_x64::<N>` selects 128-bit lane 0..3.
    let l0 = f.extract_u8x16_from_x64::<0>(d);
    let l1 = f.extract_u8x16_from_x64::<1>(d);
    let l2 = f.extract_u8x16_from_x64::<2>(d);
    let l3 = f.extract_u8x16_from_x64::<3>(d);

    let c0 = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(l0), shuffle15));
    let l1 = sse2.add_u8x16(l1, c0);
    let c1 = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(l1), shuffle15));
    let l2 = sse2.add_u8x16(l2, c1);
    let c2 = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(l2), shuffle15));
    let l3 = sse2.add_u8x16(l3, c2);

    let mut out = [0u8; 64];
    out = f.insert_u8x16_into_x64::<0>(out, l0);
    out = f.insert_u8x16_into_x64::<1>(out, l1);
    out = f.insert_u8x16_into_x64::<2>(out, l2);
    out = f.insert_u8x16_into_x64::<3>(out, l3);

    bw.add_u8x64(out, v_prev)
}

/// Remainder: AVX2 (≥32) → SSE (≥16) → scalar.
#[inline]
fn finish_hierarchical(
    avx2: Avx2,
    sse2: Sse2,
    ssse3: Ssse3,
    done: usize,
    carry: u8,
    buffer: &mut [u8],
) {
    if done >= buffer.len() {
        return;
    }
    if done == 0 {
        // No full 64-byte chunk: undo pre-bias and drop to AVX2/SSE path.
        buffer[0] = buffer[0].wrapping_sub(128);
        avx2_mod::differences_to_samples_lane_sse_tail(avx2, sse2, ssse3, buffer);
        return;
    }

    avx2_mod::differences_to_samples_from(avx2, sse2, ssse3, buffer, done, carry);
}

/// Remainder: zero-pad to 64, one process_chunk, store live prefix only.
#[inline]
#[allow(clippy::too_many_arguments)]
fn finish_padded(
    f: Avx512f,
    bw: Avx512Bw,
    sse2: Sse2,
    ssse3: Ssse3,
    done: usize,
    v_prev: [u8; 64],
    c: [u8; 64],
    shuffle15: [i8; 16],
    buffer: &mut [u8],
) {
    if done >= buffer.len() {
        return;
    }
    let rem = buffer.len() - done;
    debug_assert!(rem < 64);

    let mut tmp = [0u8; 64];
    tmp[..rem].copy_from_slice(&buffer[done..]);
    // When done==0, v_prev is zero (first-byte pre-bias already applied).
    let d = process_chunk(f, bw, sse2, ssse3, tmp, c, shuffle15, v_prev);
    buffer[done..].copy_from_slice(&d[..rem]);
}
