//! AVX2 (miraculix `Avx2`+`Sse2`+`Ssse3`) ZIP/RLE byte reconstruct -> 32-byte chunks.
//! Production: lane-prefix + SSE tail ([`differences_to_samples`]). A/B: scalar tail / full 32-wide tree.
//! `Avx2::slli_u8x32` is lane-local; full Hillis-Steele needs extract/alignr and loses to lane.

use miraculix::x86::ops::avx::avx2::Avx2;
use miraculix::x86::ops::sse::sse2::Sse2;
use miraculix::x86::ops::sse::ssse3::Ssse3;

use super::ssse3;
use super::ssse3::{to_i8x16, to_u8x16};

/// Production AVX2 entry: lane-prefix + hierarchical SSE rem.
#[inline]
pub fn differences_to_samples(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8]) {
    differences_to_samples_lane_sse_tail(avx2, sse2, ssse3, buffer);
}

/// Lane-prefix, 32B chunks, scalar rem. Per chunk: dual 16B OpenEXR trees,
/// hi half += lo last, all 32 += prev-chunk broadcast. A/B baseline;
/// production: [`differences_to_samples_lane_sse_tail`].
#[inline]
pub fn differences_to_samples_lane(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8]) {
    run_lane(avx2, sse2, ssse3, buffer, Tail::Scalar);
}

/// Same 32B lane body; rem >=16 via SSE-from-carry, then scalar &lt;16.
/// Production AVX2 path (also AVX-512 hierarchical rem).
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

miraculix::avx2_fn! {
    fn run_lane(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, buffer: &mut [u8], tail: Tail) {
        if buffer.is_empty() {
            return;
        }

        if buffer.len() < 32 {
            ssse3::differences_to_samples(sse2, ssse3, buffer);
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
            // Lane-local log-depth (two independent 16B scans).
            d = avx2.add_u8x32(d, avx2.slli_u8x32::<1>(d));
            d = avx2.add_u8x32(d, avx2.slli_u8x32::<2>(d));
            d = avx2.add_u8x32(d, avx2.slli_u8x32::<4>(d));
            d = avx2.add_u8x32(d, avx2.slli_u8x32::<8>(d));

            // Cross-lane: hi += broadcast(lo last).
            let lo128 = avx2.extract_u8x16_from_x32::<0>(d);
            let lo_last = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(lo128), shuffle15));
            let hi_carry = avx2.insert_u8x16_into_x32::<1>([0u8; 32], lo_last);
            d = avx2.add_u8x32(d, hi_carry);

            // Prev-chunk carry on all 32 lanes.
            d = avx2.add_u8x32(d, v_prev);

            buffer[offset..offset + 32].copy_from_slice(&d);

            // Next carry = broadcast d[31].
            let hi128 = avx2.extract_u8x16_from_x32::<1>(d);
            let last = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(hi128), shuffle15));
            v_prev = avx2.broadcast_u8x32(last[0]);
        }

        match tail {
            Tail::Scalar => finish_tail_scalar(sse2, ssse3, n_chunks * 32, v_prev[0], buffer),
            Tail::Sse => finish_tail_sse(sse2, ssse3, n_chunks * 32, v_prev[0], buffer),
        }
    }
}

miraculix::avx2_fn! {
    /// AVX2 lane chunks from `start` with known carry `prev` (AVX-512 rem).
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
        // Hierarchical rem: SSE then scalar.
        finish_tail_sse(sse2, ssse3, offset, v_prev[0], buffer);
    }
}

miraculix::avx2_fn! {
    /// Full 32-wide Hillis-Steele (byte left-shift 1/2/4/8/16 + prev-chunk).
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
}

/// Rem after full 32B chunks: full SSE if nothing ran, else scalar.
#[inline]
fn finish_tail_scalar(sse2: Sse2, ssse3: Ssse3, done: usize, carry: u8, buffer: &mut [u8]) {
    if done >= buffer.len() {
        return;
    }
    if done == 0 {
        // No 32B chunk: undo pre-bias, drop to SSE.
        buffer[0] = buffer[0].wrapping_sub(128);
        ssse3::differences_to_samples(sse2, ssse3, buffer);
        return;
    }
    residual_from_carry(carry, done, buffer);
}

/// Rem after full 32B: full SSE if nothing ran, else SSE-from-carry then scalar.
#[inline]
fn finish_tail_sse(sse2: Sse2, ssse3: Ssse3, done: usize, carry: u8, buffer: &mut [u8]) {
    if done >= buffer.len() {
        return;
    }
    if done == 0 {
        buffer[0] = buffer[0].wrapping_sub(128);
        ssse3::differences_to_samples(sse2, ssse3, buffer);
        return;
    }
    ssse3::differences_to_samples_from(sse2, ssse3, buffer, done, carry);
}

/// Scalar rem: `sample[i] = prev + diff[i] - 128` from `start`.
#[inline]
fn residual_from_carry(mut prev: u8, start: usize, buffer: &mut [u8]) {
    for byte in &mut buffer[start..] {
        let sample = prev.wrapping_add(*byte).wrapping_sub(128);
        *byte = sample;
        prev = sample;
    }
}

// Cross-lane byte left-shifts (zero-fill). `slli_u8x32` is lane-local only.
// Concrete fns (not shared const-generic): `avx2_fn!` captures `tt` per
// param and cannot match multi-token `const K: i32`.
miraculix::avx2_fn! {
    fn shift_left_bytes_1(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, d: [u8; 32]) -> [u8; 32] {
        let lo = avx2.extract_u8x16_from_x32::<0>(d);
        let hi = avx2.extract_u8x16_from_x32::<1>(d);
        let lo2 = sse2.slli_u8x16::<1>(lo);
        // hi' = [lo[15], hi[0..15]]: left-shift hi by 1 filled from lo.
        let hi2 = ssse3.alignr_u8x16::<15>(hi, lo);
        let out = avx2.insert_u8x16_into_x32::<0>(d, lo2);
        avx2.insert_u8x16_into_x32::<1>(out, hi2)
    }
}
miraculix::avx2_fn! {
    fn shift_left_bytes_2(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, d: [u8; 32]) -> [u8; 32] {
        let lo = avx2.extract_u8x16_from_x32::<0>(d);
        let hi = avx2.extract_u8x16_from_x32::<1>(d);
        let lo2 = sse2.slli_u8x16::<2>(lo);
        let hi2 = ssse3.alignr_u8x16::<14>(hi, lo);
        let out = avx2.insert_u8x16_into_x32::<0>(d, lo2);
        avx2.insert_u8x16_into_x32::<1>(out, hi2)
    }
}
miraculix::avx2_fn! {
    fn shift_left_bytes_4(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, d: [u8; 32]) -> [u8; 32] {
        let lo = avx2.extract_u8x16_from_x32::<0>(d);
        let hi = avx2.extract_u8x16_from_x32::<1>(d);
        let lo2 = sse2.slli_u8x16::<4>(lo);
        let hi2 = ssse3.alignr_u8x16::<12>(hi, lo);
        let out = avx2.insert_u8x16_into_x32::<0>(d, lo2);
        avx2.insert_u8x16_into_x32::<1>(out, hi2)
    }
}
miraculix::avx2_fn! {
    fn shift_left_bytes_8(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, d: [u8; 32]) -> [u8; 32] {
        let lo = avx2.extract_u8x16_from_x32::<0>(d);
        let hi = avx2.extract_u8x16_from_x32::<1>(d);
        let lo2 = sse2.slli_u8x16::<8>(lo);
        let hi2 = ssse3.alignr_u8x16::<8>(hi, lo);
        let out = avx2.insert_u8x16_into_x32::<0>(d, lo2);
        avx2.insert_u8x16_into_x32::<1>(out, hi2)
    }
}

miraculix::avx2_fn! {
    /// Left-shift 16B: lo <- 0, hi <- old lo.
    fn shift_left_bytes_16(avx2: Avx2, d: [u8; 32]) -> [u8; 32] {
        let lo = avx2.extract_u8x16_from_x32::<0>(d);
        avx2.insert_u8x16_into_x32::<1>([0u8; 32], lo)
    }
}
