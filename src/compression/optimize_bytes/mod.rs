//! Byte predictors and half-byte interleave shared by ZIP and RLE (and DWA
//! DC encode path).
//!
//! OpenEXR's C reference implements the decode-side reconstruct (un-diff) as a
//! log-depth SIMD prefix sum and interleave as SSE unpack. On x86-64 we ship
//! reconstruct via `x86/` tiers (`avx512` -> `avx2` -> `sse` -> scalar).
//! Non-x86 architectures fall back to [`portable_wide16`].

/// x86 reconstruct tiers (`sse`, `avx2`, `avx512`) + dispatch. `doc(hidden)`-public
/// so stage benches can call kernels directly, same pattern as DWA DCT.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[doc(hidden)]
pub mod x86;

/// Portable OpenEXR 16-byte log-depth reconstruct (pure Rust). Production
/// path on every non-x86 architecture; host-tested for bit-exactness.
#[doc(hidden)]
pub mod portable_wide16;

/// Integrate over all differences to the previous value in order to
/// reconstruct sample values (`sample[i] = sample[i-1] + diff[i] - 128`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn differences_to_samples(buffer: &mut [u8]) {
    if x86::try_differences_to_samples(buffer) {
        return;
    }
    differences_to_samples_scalar(buffer);
}

/// Non-x86: [`portable_wide16`]'s OpenEXR 16-byte log-depth reconstruct.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn differences_to_samples(buffer: &mut [u8]) {
    portable_wide16::differences_to_samples(buffer);
}

/// Derive differences to the previous value (`diff[i] = sample[i] - sample[i-1] + 128`).
///
/// Encode-side; already near memory bandwidth with the scalar 16-wide form, so
/// no SIMD dispatch (A/B was ~1.0×). x86 SIMD port is commented out in `x86/sse.rs`.
pub fn samples_to_differences(buffer: &mut [u8]) {
    samples_to_differences_scalar(buffer);
}

/// Interleave the two halves of the buffer so that even bytes come from the
/// first half and odd bytes from the second (OpenEXR ZIP/RLE layout).
///
/// Scalar only: on measured x86-64 the simple pair loop (already near memcpy
/// bandwidth once the final copy-back is included) matches or beats the SSE2
/// unpack port. SIMD kernels remain commented in `x86/sse.rs` with the A/B reason.
pub fn interleave_byte_blocks(separated: &mut [u8]) {
    with_reused_buffer(separated.len(), |interleaved| {
        interleave_byte_blocks_scalar(separated, interleaved);
        separated.copy_from_slice(interleaved);
    });
}

/// Separate interleaved bytes so the second half holds every other byte
/// (inverse of [`interleave_byte_blocks`]). Scalar for the same reason as
/// interleave — see that note.
pub fn separate_bytes_fragments(source: &mut [u8]) {
    with_reused_buffer(source.len(), |separated| {
        separate_bytes_fragments_scalar(source, separated);
        source.copy_from_slice(separated);
    });
}

// Scalar reference implementations (also the non-x86 path).

/// Scalar reference (also non-x86 path). Exposed for isolated stage benches.
#[doc(hidden)]
pub fn differences_to_samples_scalar(buffer: &mut [u8]) {
    // Pair-ILP form: two samples per iteration share the previous base so the
    // CPU can dual-issue the independent half of the chain. Still O(n) serial
    // depth overall — the SIMD path is the real fix.
    if let Some(first) = buffer.first() {
        let mut previous = i16::from(*first);
        for chunk in &mut buffer[1..].chunks_exact_mut(2) {
            let diff0 = i16::from(chunk[0]);
            let diff1 = i16::from(chunk[1]);
            let sample0 = (previous + diff0 - 128) as u8;
            let sample1 = (previous + diff0 + diff1 - 128 * 2) as u8;
            chunk[0] = sample0;
            chunk[1] = sample1;
            previous = i16::from(sample1);
        }
        for elem in &mut buffer[1..].chunks_exact_mut(2).into_remainder().iter_mut() {
            let sample = (previous + i16::from(*elem) - 128) as u8;
            *elem = sample;
            previous = i16::from(sample);
        }
    }
}

#[doc(hidden)]
pub fn samples_to_differences_scalar(buffer: &mut [u8]) {
    if let Some(first) = buffer.first() {
        let mut previous = i16::from(*first);
        for chunk in &mut buffer[1..].chunks_exact_mut(16) {
            let sample0 = i16::from(chunk[0]);
            let sample1 = i16::from(chunk[1]);
            let sample2 = i16::from(chunk[2]);
            let sample3 = i16::from(chunk[3]);
            let sample4 = i16::from(chunk[4]);
            let sample5 = i16::from(chunk[5]);
            let sample6 = i16::from(chunk[6]);
            let sample7 = i16::from(chunk[7]);
            let sample8 = i16::from(chunk[8]);
            let sample9 = i16::from(chunk[9]);
            let sample10 = i16::from(chunk[10]);
            let sample11 = i16::from(chunk[11]);
            let sample12 = i16::from(chunk[12]);
            let sample13 = i16::from(chunk[13]);
            let sample14 = i16::from(chunk[14]);
            let sample15 = i16::from(chunk[15]);
            chunk[0] = (sample0 - previous + 128) as u8;
            chunk[1] = (sample1 - sample0 + 128) as u8;
            chunk[2] = (sample2 - sample1 + 128) as u8;
            chunk[3] = (sample3 - sample2 + 128) as u8;
            chunk[4] = (sample4 - sample3 + 128) as u8;
            chunk[5] = (sample5 - sample4 + 128) as u8;
            chunk[6] = (sample6 - sample5 + 128) as u8;
            chunk[7] = (sample7 - sample6 + 128) as u8;
            chunk[8] = (sample8 - sample7 + 128) as u8;
            chunk[9] = (sample9 - sample8 + 128) as u8;
            chunk[10] = (sample10 - sample9 + 128) as u8;
            chunk[11] = (sample11 - sample10 + 128) as u8;
            chunk[12] = (sample12 - sample11 + 128) as u8;
            chunk[13] = (sample13 - sample12 + 128) as u8;
            chunk[14] = (sample14 - sample13 + 128) as u8;
            chunk[15] = (sample15 - sample14 + 128) as u8;
            previous = sample15;
        }
        for elem in &mut buffer[1..].chunks_exact_mut(16).into_remainder().iter_mut() {
            let diff = (i16::from(*elem) - previous + 128) as u8;
            previous = i16::from(*elem);
            *elem = diff;
        }
    }
}

#[doc(hidden)]
pub fn interleave_byte_blocks_scalar(separated: &[u8], interleaved: &mut [u8]) {
    debug_assert_eq!(separated.len(), interleaved.len());
    let (first_half, second_half) = separated.split_at((separated.len() + 1) / 2);
    let first_half_last = first_half.last().copied();
    let first_half_iter = &first_half[..second_half.len()];

    for ((first, second), out) in
        first_half_iter.iter().zip(second_half.iter()).zip(interleaved.chunks_exact_mut(2))
    {
        out[0] = *first;
        out[1] = *second;
    }

    if interleaved.len() % 2 == 1 {
        if let Some(value) = first_half_last {
            *interleaved.last_mut().unwrap() = value;
        }
    }
}

#[doc(hidden)]
pub fn separate_bytes_fragments_scalar(source: &[u8], separated: &mut [u8]) {
    debug_assert_eq!(source.len(), separated.len());
    let (first_half, second_half) = separated.split_at_mut((source.len() + 1) / 2);
    let last = source.last().copied();
    let first_half_iter = &mut first_half[..second_half.len()];

    for ((first, second), interleaved) in
        first_half_iter.iter_mut().zip(second_half.iter_mut()).zip(source.chunks_exact(2))
    {
        *first = interleaved[0];
        *second = interleaved[1];
    }

    if source.len() % 2 == 1 {
        if let Some(value) = last {
            *first_half.last_mut().unwrap() = value;
        }
    }
}

use std::cell::Cell;
thread_local! {
    // Reused between interleave/deinterleave calls. Zeroing a fresh Vec once
    // per block was historically ~10% of ZIP/RLE decode; grow-never-shrink.
    static SCRATCH_SPACE: Cell<Vec<u8>> = const { Cell::new(Vec::new()) };
}

fn with_reused_buffer<F>(length: usize, mut func: F)
where
    F: FnMut(&mut [u8]),
{
    SCRATCH_SPACE.with(|scratch_space| {
        let mut buffer = scratch_space.take();
        if buffer.len() < length {
            buffer = vec![0u8; length];
        }
        func(&mut buffer[..length]);
        scratch_space.set(buffer);
    });
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn roundtrip_interleave() {
        let source = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut modified = source.clone();
        separate_bytes_fragments(&mut modified);
        interleave_byte_blocks(&mut modified);
        assert_eq!(source, modified);
    }

    #[test]
    fn roundtrip_derive() {
        let source = vec![0, 1, 2, 7, 4, 5, 6, 7, 13, 9, 10];
        let mut modified = source.clone();
        samples_to_differences(&mut modified);
        differences_to_samples(&mut modified);
        assert_eq!(source, modified);
    }

    /// Reconstruct kernels must match the scalar reference on every length,
    /// including remainders around 16/32/64-byte chunk boundaries.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_reconstruct_kernels_match_scalar_all_lengths() {
        use crate::compression::simd_tier::x86::miraculix_x86;

        let sse_tokens = (miraculix_x86::sse2(), miraculix_x86::ssse3());
        let avx2_token = miraculix_x86::avx2();
        let avx512_tokens = (miraculix_x86::avx512f(), miraculix_x86::avx512bw());

        for len in 0..192 {
            let source: Vec<u8> =
                (0..len).map(|i| (i as u8).wrapping_mul(17).wrapping_add(3)).collect();

            let mut scalar = source.clone();
            differences_to_samples_scalar(&mut scalar);

            if let (Some(sse2), Some(ssse3)) = sse_tokens {
                let mut simd = source.clone();
                x86::sse::differences_to_samples(sse2, ssse3, &mut simd);
                assert_eq!(scalar, simd, "sse reconstruct len={len}");
            }
            if let (Some(avx2), Some(sse2), Some(ssse3)) = (avx2_token, sse_tokens.0, sse_tokens.1) {
                let mut lane = source.clone();
                x86::avx2::differences_to_samples_lane(avx2, sse2, ssse3, &mut lane);
                assert_eq!(scalar, lane, "avx2_lane reconstruct len={len}");

                let mut full = source.clone();
                x86::avx2::differences_to_samples_full(avx2, sse2, ssse3, &mut full);
                assert_eq!(scalar, full, "avx2_full reconstruct len={len}");

                let mut sse_tail = source.clone();
                x86::avx2::differences_to_samples_lane_sse_tail(avx2, sse2, ssse3, &mut sse_tail);
                assert_eq!(scalar, sse_tail, "avx2_lane_sse_tail reconstruct len={len}");
            }
            if let (Some(f), Some(bw), Some(avx2), Some(sse2), Some(ssse3)) =
                (avx512_tokens.0, avx512_tokens.1, avx2_token, sse_tokens.0, sse_tokens.1)
            {
                let mut lane = source.clone();
                x86::avx512::differences_to_samples_lane(f, bw, avx2, sse2, ssse3, &mut lane);
                assert_eq!(scalar, lane, "avx512_lane reconstruct len={len}");

                let mut masked = source.clone();
                x86::avx512::differences_to_samples_lane_masked(f, bw, avx2, sse2, ssse3, &mut masked);
                assert_eq!(scalar, masked, "avx512_lane_masked reconstruct len={len}");
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_reconstruct_kernels_match_scalar_large() {
        use crate::compression::simd_tier::x86::miraculix_x86;

        let len = 64 * 1024 + 17;
        let source: Vec<u8> =
            (0..len).map(|i| ((i * 131) as u8).wrapping_add((i >> 8) as u8)).collect();

        let mut scalar = source.clone();
        differences_to_samples_scalar(&mut scalar);

        if let (Some(sse2), Some(ssse3)) = (miraculix_x86::sse2(), miraculix_x86::ssse3()) {
            let mut a = source.clone();
            x86::sse::differences_to_samples(sse2, ssse3, &mut a);
            assert_eq!(scalar, a, "sse large");
        }
        if let (Some(avx2), Some(sse2), Some(ssse3)) =
            (miraculix_x86::avx2(), miraculix_x86::sse2(), miraculix_x86::ssse3())
        {
            let mut a = source.clone();
            x86::avx2::differences_to_samples_lane(avx2, sse2, ssse3, &mut a);
            assert_eq!(scalar, a, "avx2_lane large");
            let mut a = source.clone();
            x86::avx2::differences_to_samples_full(avx2, sse2, ssse3, &mut a);
            assert_eq!(scalar, a, "avx2_full large");
            let mut a = source.clone();
            x86::avx2::differences_to_samples_lane_sse_tail(avx2, sse2, ssse3, &mut a);
            assert_eq!(scalar, a, "avx2_lane_sse_tail large");
        }
        if let (Some(f), Some(bw), Some(avx2), Some(sse2), Some(ssse3)) = (
            miraculix_x86::avx512f(),
            miraculix_x86::avx512bw(),
            miraculix_x86::avx2(),
            miraculix_x86::sse2(),
            miraculix_x86::ssse3(),
        ) {
            let mut a = source.clone();
            x86::avx512::differences_to_samples_lane(f, bw, avx2, sse2, ssse3, &mut a);
            assert_eq!(scalar, a, "avx512_lane large");
            let mut a = source.clone();
            x86::avx512::differences_to_samples_lane_masked(f, bw, avx2, sse2, ssse3, &mut a);
            assert_eq!(scalar, a, "avx512_lane_masked large");
        }
    }

    #[test]
    fn production_dispatch_matches_scalar() {
        for &len in &[0usize, 1, 15, 16, 17, 31, 32, 33, 64, 1000, 4096 + 3] {
            let source: Vec<u8> =
                (0..len).map(|i| (i as u8).wrapping_mul(13).wrapping_add(7)).collect();
            let mut scalar = source.clone();
            let mut prod = source;
            differences_to_samples_scalar(&mut scalar);
            differences_to_samples(&mut prod);
            assert_eq!(scalar, prod, "dispatch len={len}");
        }
    }

    /// Portable 16-byte OpenEXR tree (non-x86 production kernel) — host-tested.
    #[test]
    fn portable_wide16_matches_scalar_all_lengths() {
        for len in 0..192 {
            let source: Vec<u8> =
                (0..len).map(|i| (i as u8).wrapping_mul(17).wrapping_add(3)).collect();
            let mut scalar = source.clone();
            let mut wide = source;
            differences_to_samples_scalar(&mut scalar);
            portable_wide16::differences_to_samples(&mut wide);
            assert_eq!(scalar, wide, "portable_wide16 len={len}");
        }
    }

    #[test]
    fn portable_wide16_matches_scalar_large() {
        let len = 64 * 1024 + 17;
        let source: Vec<u8> =
            (0..len).map(|i| ((i * 131) as u8).wrapping_add((i >> 8) as u8)).collect();
        let mut scalar = source.clone();
        let mut wide = source;
        differences_to_samples_scalar(&mut scalar);
        portable_wide16::differences_to_samples(&mut wide);
        assert_eq!(scalar, wide, "portable_wide16 large");
    }
}
