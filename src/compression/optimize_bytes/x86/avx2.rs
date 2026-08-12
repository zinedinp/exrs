//! AVX2 (miraculix `Avx2`+`Sse2`+`Ssse3`) ZIP/RLE byte reconstruct -> 32-byte chunks.
//!
//! Designs, all bit-exact vs scalar:
//!
//! - **lane + SSE tail** ([`differences_to_samples`]): production when AVX-512
//!   is absent — within-lane 16-byte OpenEXR trees + high-half fixup; rem ≥16
//!   via SSE-from-carry, then scalar &lt;16.
//! - **lane + scalar tail** ([`differences_to_samples_lane`]): same body, pure
//!   scalar remainder (A/B baseline).
//! - **full** ([`differences_to_samples_full`]): true 32-wide Hillis–Steele with
//!   cross-lane shifts. Kept for re-bench; extract/`alignr` tax loses to lane.
//!
//! `Avx2::slli_u8x32` is lane-local; that is why the full tree needs explicit
//! extract/alignr/insert helpers.

use miraculix::x86::ops::avx::avx2::Avx2;
use miraculix::x86::ops::sse::sse2::Sse2;
use miraculix::x86::ops::sse::ssse3::Ssse3;

use super::sse;
use super::sse::{to_i8x16, to_u8x16};

/// Production AVX2 entry: lane-prefix with hierarchical SSE remainder.
#[inline]
pub fn differences_to_samples(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8]) {
    differences_to_samples_lane_sse_tail(avx2, sse2, ssse3, buffer);
}

/// AVX2 **lane-prefix** reconstruct: 32-byte chunks, **scalar** remainder.
///
/// 1. run the OpenEXR 16-byte log-depth sum independently in each half,
/// 2. add the low half's last byte into every high-half lane (cross-lane fixup),
/// 3. add the previous chunk's last-byte broadcast to all 32 lanes.
///
/// A/B baseline; production uses [`differences_to_samples_lane_sse_tail`].
#[inline]
pub fn differences_to_samples_lane(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8]) {
    run_lane(avx2, sse2, ssse3, buffer, Tail::Scalar);
}

/// Same 32-byte lane body; rem ≥16 goes through SSE-from-carry, then scalar &lt;16.
/// Production AVX2 path (and AVX-512 hierarchical remainder).
#[inline]
pub fn differences_to_samples_lane_sse_tail(
    avx2: Avx2,
    sse2: Sse2,
    ssse3: Ssse3,
    buffer: &mut [u8],
) {
    run_lane(avx2, sse2, ssse3, buffer, Tail::Sse);
}

#[derive(Clone, Copy)]
enum Tail {
    Scalar,
    Sse,
}

#[inline]
fn run_lane(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8], tail: Tail) {
    if buffer.is_empty() {
        return;
    }

    if buffer.len() < 32 {
        sse::differences_to_samples(sse2, ssse3, buffer);
        return;
    }

    buffer[0] = buffer[0].wrapping_add(128);

    let c = [128u8; 32];
    let shuffle15 = to_i8x16([15u8; 16]);
    let mut v_prev = [0u8; 32];

    let n_chunks = buffer.len() / 32;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 32;
        let mut d: [u8; 32] = buffer[offset..offset + 32].try_into().unwrap();

        d = avx2.add_u8x32(d, c);
        // Within-lane log-depth (two independent 16-byte scans).
        d = avx2.add_u8x32(d, avx2.slli_u8x32::<1>(d));
        d = avx2.add_u8x32(d, avx2.slli_u8x32::<2>(d));
        d = avx2.add_u8x32(d, avx2.slli_u8x32::<4>(d));
        d = avx2.add_u8x32(d, avx2.slli_u8x32::<8>(d));

        // Cross-lane fixup: high half += broadcast(low half last byte).
        let lo128 = avx2.extract_u8x16_from_x32::<0>(d);
        let lo_last = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(lo128), shuffle15));
        let hi_carry = avx2.insert_u8x16_into_x32::<1>([0u8; 32], lo_last);
        d = avx2.add_u8x32(d, hi_carry);

        // Previous chunk carry across all 32 lanes.
        d = avx2.add_u8x32(d, v_prev);

        buffer[offset..offset + 32].copy_from_slice(&d);

        // Next carry = broadcast of d[31].
        let hi128 = avx2.extract_u8x16_from_x32::<1>(d);
        let last = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(hi128), shuffle15));
        v_prev = avx2.broadcast_u8x32(last[0]);
    }

    match tail {
        Tail::Scalar => finish_tail_scalar(sse2, ssse3, n_chunks * 32, v_prev[0], buffer),
        Tail::Sse => finish_tail_sse(sse2, ssse3, n_chunks * 32, v_prev[0], buffer),
    }
}

/// Continue one or more AVX2 lane chunks from `start` with known carry `prev`.
///
/// Used by AVX-512 hierarchical remainder (no first-byte pre-bias).
#[inline]
pub fn differences_to_samples_from(
    avx2: Avx2,
    sse2: Sse2,
    ssse3: Ssse3,
    buffer: &mut [u8],
    start: usize,
    prev: u8,
) {
    if start >= buffer.len() {
        return;
    }

    let c = [128u8; 32];
    let shuffle15 = to_i8x16([15u8; 16]);
    let mut v_prev = [prev; 32];

    let mut offset = start;
    while offset + 32 <= buffer.len() {
        let mut d: [u8; 32] = buffer[offset..offset + 32].try_into().unwrap();

        d = avx2.add_u8x32(d, c);
        d = avx2.add_u8x32(d, avx2.slli_u8x32::<1>(d));
        d = avx2.add_u8x32(d, avx2.slli_u8x32::<2>(d));
        d = avx2.add_u8x32(d, avx2.slli_u8x32::<4>(d));
        d = avx2.add_u8x32(d, avx2.slli_u8x32::<8>(d));

        let lo128 = avx2.extract_u8x16_from_x32::<0>(d);
        let lo_last = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(lo128), shuffle15));
        let hi_carry = avx2.insert_u8x16_into_x32::<1>([0u8; 32], lo_last);
        d = avx2.add_u8x32(d, hi_carry);
        d = avx2.add_u8x32(d, v_prev);

        buffer[offset..offset + 32].copy_from_slice(&d);

        let hi128 = avx2.extract_u8x16_from_x32::<1>(d);
        let last = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(hi128), shuffle15));
        v_prev = avx2.broadcast_u8x32(last[0]);
        offset += 32;
    }

    if offset >= buffer.len() {
        return;
    }
    // Hierarchical: one SSE chunk then scalar.
    finish_tail_sse(sse2, ssse3, offset, v_prev[0], buffer);
}

/// AVX2 **full 32-wide** log-depth reconstruct (bench / alternative only).
///
/// True Hillis–Steele over 32 bytes: shift-left-with-zero-fill by 1/2/4/8/**16**
/// across the whole YMM, then add the previous-chunk broadcast.
#[inline]
pub fn differences_to_samples_full(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    buffer[0] = buffer[0].wrapping_add(128);

    let c = [128u8; 32];
    let shuffle15 = to_i8x16([15u8; 16]);
    let mut v_prev = [0u8; 32];

    let n_chunks = buffer.len() / 32;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 32;
        let mut d: [u8; 32] = buffer[offset..offset + 32].try_into().unwrap();

        d = avx2.add_u8x32(d, c);
        d = avx2.add_u8x32(d, shift_left_bytes_1(avx2, sse2, ssse3, d));
        d = avx2.add_u8x32(d, shift_left_bytes_2(avx2, sse2, ssse3, d));
        d = avx2.add_u8x32(d, shift_left_bytes_4(avx2, sse2, ssse3, d));
        d = avx2.add_u8x32(d, shift_left_bytes_8(avx2, sse2, ssse3, d));
        d = avx2.add_u8x32(d, shift_left_bytes_16(avx2, d));
        d = avx2.add_u8x32(d, v_prev);

        buffer[offset..offset + 32].copy_from_slice(&d);

        let hi128 = avx2.extract_u8x16_from_x32::<1>(d);
        let last = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(hi128), shuffle15));
        v_prev = avx2.broadcast_u8x32(last[0]);
    }

    finish_tail_scalar(sse2, ssse3, n_chunks * 32, v_prev[0], buffer);
}

/// Remainder after full 32-byte chunks: full SSE if nothing ran, else scalar.
#[inline]
fn finish_tail_scalar(sse2: Sse2, ssse3: Ssse3, done: usize, carry: u8, buffer: &mut [u8]) {
    if done >= buffer.len() {
        return;
    }
    if done == 0 {
        // No full 32-byte chunk: undo pre-bias and use the SSE path.
        buffer[0] = buffer[0].wrapping_sub(128);
        sse::differences_to_samples(sse2, ssse3, buffer);
        return;
    }
    residual_from_carry(carry, done, buffer);
}

/// Remainder after full 32-byte chunks: full SSE if nothing ran, else SSE-from
/// carry (16-byte chunks) then scalar &lt;16.
#[inline]
fn finish_tail_sse(sse2: Sse2, ssse3: Ssse3, done: usize, carry: u8, buffer: &mut [u8]) {
    if done >= buffer.len() {
        return;
    }
    if done == 0 {
        buffer[0] = buffer[0].wrapping_sub(128);
        sse::differences_to_samples(sse2, ssse3, buffer);
        return;
    }
    sse::differences_to_samples_from(sse2, ssse3, buffer, done, carry);
}

/// Scalar tail: `sample[i] = prev + diff[i] - 128` from `start` onward.
#[inline]
fn residual_from_carry(mut prev: u8, start: usize, buffer: &mut [u8]) {
    for byte in &mut buffer[start..] {
        let sample = prev.wrapping_add(*byte).wrapping_sub(128);
        *byte = sample;
        prev = sample;
    }
}

// Cross-lane byte left-shifts (zero-fill). `Avx2::slli_u8x32` is lane-local only.
#[inline]
fn shift_left_bytes_1(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, d: [u8; 32]) -> [u8; 32] {
    shift_left_bytes_k::<1, 15>(avx2, sse2, ssse3, d)
}
#[inline]
fn shift_left_bytes_2(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, d: [u8; 32]) -> [u8; 32] {
    shift_left_bytes_k::<2, 14>(avx2, sse2, ssse3, d)
}
#[inline]
fn shift_left_bytes_4(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, d: [u8; 32]) -> [u8; 32] {
    shift_left_bytes_k::<4, 12>(avx2, sse2, ssse3, d)
}
#[inline]
fn shift_left_bytes_8(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, d: [u8; 32]) -> [u8; 32] {
    shift_left_bytes_k::<8, 8>(avx2, sse2, ssse3, d)
}

/// Left-shift `d` by `K` bytes (1..=15) with zero fill; `ALIGN = 16 - K`.
#[inline]
fn shift_left_bytes_k<const K: i32, const ALIGN: i32>(
    avx2: Avx2,
    sse2: Sse2,
    ssse3: Ssse3,
    d: [u8; 32],
) -> [u8; 32] {
    let lo = avx2.extract_u8x16_from_x32::<0>(d);
    let hi = avx2.extract_u8x16_from_x32::<1>(d);
    let lo2 = sse2.slli_u8x16::<K>(lo);
    // hi' = [lo[16-K .. 16], hi[0 .. 16-K]]  ≡ left-shift hi by K filled from lo
    let hi2 = ssse3.alignr_u8x16::<ALIGN>(hi, lo);
    let out = avx2.insert_u8x16_into_x32::<0>(d, lo2);
    avx2.insert_u8x16_into_x32::<1>(out, hi2)
}

/// Left-shift by 16 bytes: low <- 0, high <- old low.
#[inline]
fn shift_left_bytes_16(avx2: Avx2, d: [u8; 32]) -> [u8; 32] {
    let lo = avx2.extract_u8x16_from_x32::<0>(d);
    avx2.insert_u8x16_into_x32::<1>([0u8; 32], lo)
}
