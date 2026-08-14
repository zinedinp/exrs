// Runtime x86 SIMD dispatch for DWA fused lossy-DCT decode.
// Order: AVX-512 (2 blocks/reg, 16-lane pair write) -> AVX2+F16C fused unRLE->write -> strip-tile scalar/SSE2.
// Kernels in `avx2`/`avx512`/`sse2`; this file picks the tier and owns shared write-row convert constants.

mod avx2;
mod avx512;
mod sse2;

use crate::{compression::simd_tier::x86::miraculix_x86, error::Result as ExrResult};

use super::{PackedStream, ScanlineTarget};

/// Rounding-mode immediate for `vcvtps2ph`: `_MM_FROUND_TO_NEAREST_INT` (0).
/// Must match the immediate `half::f16::from_f32`'s own F16C fast path uses
/// (half's `arch/x86.rs`), so the two conversions are bit-identical. Shared
/// by the AVX2 8-lane and AVX-512 16-lane write-row conversions.
pub(super) const ROUND_TO_NEAREST: i32 = 0;

pub(super) fn try_from_half_zigzag(zig_zag: &[u16; 64], dst: &mut [f32; 64]) -> bool {
    let (Some(sse2), Some(ssse3), Some(sse41), Some(f16c)) =
        (miraculix_x86::sse2(), miraculix_x86::ssse3(), miraculix_x86::sse41(), miraculix_x86::f16c())
    else {
        return false;
    };
    avx2::from_half_zigzag(sse2, ssse3, sse41, f16c, zig_zag, dst);
    true
}

/// Vectorized version of `decode_lossy_dct_group`'s F16-output write-row
/// loop for a full 8-wide row.
pub(super) fn try_write_row_f16(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let Some(f16c) = miraculix_x86::f16c() else {
        return false;
    };
    avx2::write_row_f16(f16c, row, to_linear, out_row)
}

/// Same as `try_write_row_f16`, but widens the linearized halves back to f32
/// (via a second `vcvtph2ps`) for F32-sample-type channels, matching the
/// scalar path's `linear.to_f32()`.
pub(super) fn try_write_row_f32(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let Some(f16c) = miraculix_x86::f16c() else {
        return false;
    };
    avx2::write_row_f32(f16c, row, to_linear, out_row)
}

/// SSE2-only fallback for `try_write_row_f16`/`try_write_row_f32`
pub(super) fn try_write_row_f16_sse2(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let (Some(sse), Some(sse2)) = (miraculix_x86::sse(), miraculix_x86::sse2()) else {
        return false;
    };
    sse2::write_row_f16(sse, sse2, row, to_linear, out_row)
}

/// SSE2-only fallback for `try_write_row_f32`, see `try_write_row_f16_sse2`.
pub(super) fn try_write_row_f32_sse2(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let (Some(sse), Some(sse2)) = (miraculix_x86::sse(), miraculix_x86::sse2()) else {
        return false;
    };
    sse2::write_row_f32(sse, sse2, row, to_linear, out_row)
}

/// For each spatial 8x8, finish unRLE -> zigzag -> iDCT -> CSC -> scanline
/// write before touching the next block (AVX2+F16C tier). `None` when the
/// host lacks AVX2+F16C (caller falls back to the strip-tiled path).
pub(super) fn try_decode_group_fused(
    ac: &mut PackedStream<'_>,
    dc: &mut PackedStream<'_>,
    width: usize,
    height: usize,
    to_linear: Option<&[u16; 65536]>,
    targets: &mut [ScanlineTarget<'_>],
    out: &mut [u8],
) -> Option<ExrResult<()>> {
    let (Some(sse2), Some(ssse3), Some(sse41), Some(f16c)) =
        (miraculix_x86::sse2(), miraculix_x86::ssse3(), miraculix_x86::sse41(), miraculix_x86::f16c())
    else {
        return None;
    };
    // F16C gates this whole path, so AVX2 (and hence base AVX) must already
    // be present; the CSC/DCT step needs its own token since it was ported
    // to miraculix.
    let avx = miraculix_x86::avx().expect("AVX confirmed available by the AVX2 tier gate above");
    Some(avx2::decode_group_fused(sse2, ssse3, sse41, f16c, avx, ac, dc, width, height, to_linear, targets, out))
}

/// AVX-512 analog of `try_decode_group_fused`: same fused shape, but the
/// DCT/CSC step processes 2 spatial blocks at once (one 512-bit register per
/// step), and the write step widens to 16 lanes for horizontally-adjacent
/// pairs (see `avx512::write_pair_block`). Zigzag stays on the AVX2 tier per
/// block either way. `None` without AVX-512+F16C; caller then tries
/// `try_decode_group_fused`, then the strip-tiled fallback.
pub(super) fn try_decode_group_fused_avx512(
    ac: &mut PackedStream<'_>,
    dc: &mut PackedStream<'_>,
    width: usize,
    height: usize,
    to_linear: Option<&[u16; 65536]>,
    targets: &mut [ScanlineTarget<'_>],
    out: &mut [u8],
) -> Option<ExrResult<()>> {
    let Some(avx512f) = miraculix_x86::avx512f() else {
        return None;
    };
    let (Some(sse2), Some(ssse3), Some(sse41), Some(f16c)) =
        (miraculix_x86::sse2(), miraculix_x86::ssse3(), miraculix_x86::sse41(), miraculix_x86::f16c())
    else {
        return None;
    };
    // AVX-512 gates this path, so base AVX and AVX-512DQ are all already
    // present; the DCT/CSC step needs all three tokens.
    let avx = miraculix_x86::avx().expect("AVX confirmed available by the AVX-512 tier gate above");
    let avx512dq = miraculix_x86::avx512dq()
        .expect("AVX-512DQ confirmed available by the AVX-512 tier gate above");
    Some(avx512::decode_group_fused(
        sse2, ssse3, sse41, f16c, avx, avx512f, avx512dq, ac, dc, width, height, to_linear, targets, out,
    ))
}
