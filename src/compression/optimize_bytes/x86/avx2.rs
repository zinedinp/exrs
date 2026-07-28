//! AVX2 (pulp `V3`) ZIP/RLE byte reconstruct -> 32-byte chunks.
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
//! `_mm256_slli_si256` is lane-local; that is why the full tree needs explicit
//! extract/alignr/insert helpers.

use std::convert::TryInto;

use pulp::x86::V3;

use super::sse;

type M256 = std::arch::x86_64::__m256i;

/// Production AVX2 entry: lane-prefix with hierarchical SSE remainder.
#[inline]
pub fn differences_to_samples(v3: V3, buffer: &mut [u8]) {
    differences_to_samples_lane_sse_tail(v3, buffer);
}

/// AVX2 **lane-prefix** reconstruct: 32-byte chunks, **scalar** remainder.
///
/// 1. run the OpenEXR 16-byte log-depth sum independently in each half,
/// 2. add the low half's last byte into every high-half lane (cross-lane fixup),
/// 3. add the previous chunk's last-byte broadcast to all 32 lanes.
///
/// A/B baseline; production uses [`differences_to_samples_lane_sse_tail`].
#[inline]
pub fn differences_to_samples_lane(v3: V3, buffer: &mut [u8]) {
    run_lane(v3, buffer, Tail::Scalar);
}

/// Same 32-byte lane body; rem ≥16 goes through SSE-from-carry, then scalar &lt;16.
/// Production AVX2 path (and AVX-512 hierarchical remainder).
#[inline]
pub fn differences_to_samples_lane_sse_tail(v3: V3, buffer: &mut [u8]) {
    run_lane(v3, buffer, Tail::Sse);
}

#[derive(Clone, Copy)]
enum Tail {
    Scalar,
    Sse,
}

#[inline]
fn run_lane(v3: V3, buffer: &mut [u8], tail: Tail) {
    if buffer.is_empty() {
        return;
    }

    let avx = v3.avx;
    let avx2 = v3.avx2;
    let sse2 = v3.sse2;
    let ssse3 = v3.ssse3;

    buffer[0] = buffer[0].wrapping_add(128);

    let c: M256 = avx._mm256_set1_epi8(-128);
    let shuffle15 = sse2._mm_set1_epi8(15);
    let mut v_prev: M256 = avx._mm256_setzero_si256();

    let n_chunks = buffer.len() / 32;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 32;
        let loaded: [u8; 32] = buffer[offset..offset + 32].try_into().unwrap();
        let mut d: M256 = pulp::cast!(loaded);

        d = avx2._mm256_add_epi8(d, c);
        // Within-lane log-depth (two independent 16-byte scans).
        d = avx2._mm256_add_epi8(d, avx2._mm256_slli_si256::<1>(d));
        d = avx2._mm256_add_epi8(d, avx2._mm256_slli_si256::<2>(d));
        d = avx2._mm256_add_epi8(d, avx2._mm256_slli_si256::<4>(d));
        d = avx2._mm256_add_epi8(d, avx2._mm256_slli_si256::<8>(d));

        // Cross-lane fixup: high half += broadcast(low half last byte).
        let lo128 = avx2._mm256_extracti128_si256::<0>(d);
        let lo_last = ssse3._mm_shuffle_epi8(lo128, shuffle15);
        let hi_carry = avx2._mm256_inserti128_si256::<1>(avx._mm256_setzero_si256(), lo_last);
        d = avx2._mm256_add_epi8(d, hi_carry);

        // Previous chunk carry across all 32 lanes.
        d = avx2._mm256_add_epi8(d, v_prev);

        let stored: [u8; 32] = pulp::cast!(d);
        buffer[offset..offset + 32].copy_from_slice(&stored);

        // Next carry = broadcast of d[31].
        let hi128 = avx2._mm256_extracti128_si256::<1>(d);
        let last = ssse3._mm_shuffle_epi8(hi128, shuffle15);
        v_prev = avx2._mm256_broadcastb_epi8(last);
    }

    match tail {
        Tail::Scalar => finish_tail_scalar(v3, n_chunks * 32, v_prev, buffer),
        Tail::Sse => finish_tail_sse(v3, n_chunks * 32, v_prev, buffer),
    }
}

/// Continue one or more AVX2 lane chunks from `start` with known carry `prev`.
///
/// Used by AVX-512 hierarchical remainder (no first-byte pre-bias).
#[inline]
pub fn differences_to_samples_from(v3: V3, buffer: &mut [u8], start: usize, prev: u8) {
    if start >= buffer.len() {
        return;
    }

    let avx = v3.avx;
    let avx2 = v3.avx2;
    let sse2 = v3.sse2;
    let ssse3 = v3.ssse3;

    let c: M256 = avx._mm256_set1_epi8(-128);
    let shuffle15 = sse2._mm_set1_epi8(15);
    let mut v_prev: M256 = avx._mm256_set1_epi8(prev as i8);

    let mut offset = start;
    while offset + 32 <= buffer.len() {
        let loaded: [u8; 32] = buffer[offset..offset + 32].try_into().unwrap();
        let mut d: M256 = pulp::cast!(loaded);

        d = avx2._mm256_add_epi8(d, c);
        d = avx2._mm256_add_epi8(d, avx2._mm256_slli_si256::<1>(d));
        d = avx2._mm256_add_epi8(d, avx2._mm256_slli_si256::<2>(d));
        d = avx2._mm256_add_epi8(d, avx2._mm256_slli_si256::<4>(d));
        d = avx2._mm256_add_epi8(d, avx2._mm256_slli_si256::<8>(d));

        let lo128 = avx2._mm256_extracti128_si256::<0>(d);
        let lo_last = ssse3._mm_shuffle_epi8(lo128, shuffle15);
        let hi_carry = avx2._mm256_inserti128_si256::<1>(avx._mm256_setzero_si256(), lo_last);
        d = avx2._mm256_add_epi8(d, hi_carry);
        d = avx2._mm256_add_epi8(d, v_prev);

        let stored: [u8; 32] = pulp::cast!(d);
        buffer[offset..offset + 32].copy_from_slice(&stored);

        let hi128 = avx2._mm256_extracti128_si256::<1>(d);
        let last = ssse3._mm_shuffle_epi8(hi128, shuffle15);
        v_prev = avx2._mm256_broadcastb_epi8(last);
        offset += 32;
    }

    if offset >= buffer.len() {
        return;
    }
    let prev_bytes: [u8; 32] = pulp::cast!(v_prev);
    // Hierarchical: one SSE chunk then scalar.
    finish_tail_sse(v3, offset, avx._mm256_set1_epi8(prev_bytes[0] as i8), buffer);
}

/// AVX2 **full 32-wide** log-depth reconstruct (bench / alternative only).
///
/// True Hillis–Steele over 32 bytes: shift-left-with-zero-fill by 1/2/4/8/**16**
/// across the whole YMM, then add the previous-chunk broadcast.
#[inline]
pub fn differences_to_samples_full(v3: V3, buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    let avx = v3.avx;
    let avx2 = v3.avx2;
    let sse2 = v3.sse2;
    let ssse3 = v3.ssse3;

    buffer[0] = buffer[0].wrapping_add(128);

    let c: M256 = avx._mm256_set1_epi8(-128);
    let shuffle15 = sse2._mm_set1_epi8(15);
    let mut v_prev: M256 = avx._mm256_setzero_si256();

    let n_chunks = buffer.len() / 32;
    for chunk_index in 0..n_chunks {
        let offset = chunk_index * 32;
        let loaded: [u8; 32] = buffer[offset..offset + 32].try_into().unwrap();
        let mut d: M256 = pulp::cast!(loaded);

        d = avx2._mm256_add_epi8(d, c);
        d = avx2._mm256_add_epi8(d, shift_left_bytes_1(v3, d));
        d = avx2._mm256_add_epi8(d, shift_left_bytes_2(v3, d));
        d = avx2._mm256_add_epi8(d, shift_left_bytes_4(v3, d));
        d = avx2._mm256_add_epi8(d, shift_left_bytes_8(v3, d));
        d = avx2._mm256_add_epi8(d, shift_left_bytes_16(v3, d));
        d = avx2._mm256_add_epi8(d, v_prev);

        let stored: [u8; 32] = pulp::cast!(d);
        buffer[offset..offset + 32].copy_from_slice(&stored);

        let hi128 = avx2._mm256_extracti128_si256::<1>(d);
        let last = ssse3._mm_shuffle_epi8(hi128, shuffle15);
        v_prev = avx2._mm256_broadcastb_epi8(last);
    }

    finish_tail_scalar(v3, n_chunks * 32, v_prev, buffer);
}

/// Remainder after full 32-byte chunks: full SSE if nothing ran, else scalar.
///
/// `V3: Deref<Target = V2>` — reuse the caller token (no second `try_new`).
#[inline]
fn finish_tail_scalar(v3: V3, done: usize, v_prev: M256, buffer: &mut [u8]) {
    if done >= buffer.len() {
        return;
    }
    if done == 0 {
        // No full 32-byte chunk: undo pre-bias and use the SSE path.
        buffer[0] = buffer[0].wrapping_sub(128);
        sse::differences_to_samples(*v3, buffer);
        return;
    }
    let prev_bytes: [u8; 32] = pulp::cast!(v_prev);
    residual_from_carry(prev_bytes[0], done, buffer);
}

/// Remainder after full 32-byte chunks: full SSE if nothing ran, else SSE-from
/// carry (16-byte chunks) then scalar &lt;16.
#[inline]
fn finish_tail_sse(v3: V3, done: usize, v_prev: M256, buffer: &mut [u8]) {
    if done >= buffer.len() {
        return;
    }
    if done == 0 {
        buffer[0] = buffer[0].wrapping_sub(128);
        sse::differences_to_samples(*v3, buffer);
        return;
    }
    let prev_bytes: [u8; 32] = pulp::cast!(v_prev);
    sse::differences_to_samples_from(*v3, buffer, done, prev_bytes[0]);
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

// Cross-lane byte left-shifts (zero-fill). AVX2 slli_si256 is lane-local only.
#[inline]
fn shift_left_bytes_1(v3: V3, d: M256) -> M256 {
    shift_left_bytes_k::<1, 15>(v3, d)
}
#[inline]
fn shift_left_bytes_2(v3: V3, d: M256) -> M256 {
    shift_left_bytes_k::<2, 14>(v3, d)
}
#[inline]
fn shift_left_bytes_4(v3: V3, d: M256) -> M256 {
    shift_left_bytes_k::<4, 12>(v3, d)
}
#[inline]
fn shift_left_bytes_8(v3: V3, d: M256) -> M256 {
    shift_left_bytes_k::<8, 8>(v3, d)
}

/// Left-shift `d` by `K` bytes (1..=15) with zero fill; `ALIGN = 16 - K`.
#[inline]
fn shift_left_bytes_k<const K: i32, const ALIGN: i32>(v3: V3, d: M256) -> M256 {
    let avx2 = v3.avx2;
    let ssse3 = v3.ssse3;
    let lo = avx2._mm256_extracti128_si256::<0>(d);
    let hi = avx2._mm256_extracti128_si256::<1>(d);
    let lo2 = v3.sse2._mm_slli_si128::<K>(lo);
    // hi' = [lo[16-K .. 16], hi[0 .. 16-K]]  ≡ left-shift hi by K filled from lo
    let hi2 = ssse3._mm_alignr_epi8::<ALIGN>(hi, lo);
    let mut out = avx2._mm256_inserti128_si256::<0>(d, lo2);
    out = avx2._mm256_inserti128_si256::<1>(out, hi2);
    out
}

/// Left-shift by 16 bytes: low <- 0, high <- old low.
#[inline]
fn shift_left_bytes_16(v3: V3, d: M256) -> M256 {
    let avx = v3.avx;
    let avx2 = v3.avx2;
    let lo = avx2._mm256_extracti128_si256::<0>(d);
    avx2._mm256_inserti128_si256::<1>(avx._mm256_setzero_si256(), lo)
}
