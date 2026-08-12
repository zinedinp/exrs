// Runtime x86 SIMD dispatch for the DWA fused lossy-DCT decode path: picks
// the AVX-512 tier (2 spatial blocks/register for DCT/CSC, 16-lane write for
// adjacent pairs) when available, else the AVX2+F16C tier (1 block per step,
// fused unRLE -> zigzag -> iDCT -> CSC -> write so the ~1 KiB working set
// stays L1-hot instead of four passes over a wide strip tile), else lets the
// caller fall back to the strip-tiled scalar/SSE2 path.
//
// Kernels live in `avx2`/`avx512` (one file per tier, mirroring how
// `discrete_cosine_transform::x86` and `color_space_conversion::x86` are
// organized); this file only decides which tier applies and owns the one
// piece genuinely shared by both tiers' write-row conversion.

mod avx2;
mod avx512;
mod sse2;

use crate::{
    compression::simd_tier::x86::{
        f16c as tier_f16c, miraculix_x86, v1 as tier_v1, v3 as tier_v3, v4 as tier_v4,
    },
    error::Result as ExrResult,
};

use super::{PackedStream, ScanlineTarget};

/// Rounding-mode immediate for `vcvtps2ph`: `_MM_FROUND_TO_NEAREST_INT` (0).
/// Must match the immediate `half::f16::from_f32`'s own F16C fast path uses
/// (half's `arch/x86.rs`), so the two conversions are bit-identical. Shared
/// by the AVX2 8-lane and AVX-512 16-lane write-row conversions.
pub(super) const ROUND_TO_NEAREST: i32 = 0;

pub(super) fn try_from_half_zigzag(zig_zag: &[u16; 64], dst: &mut [f32; 64]) -> bool {
    let (Some(v3), Some(f16c)) = (tier_v3(), tier_f16c()) else {
        return false;
    };
    avx2::from_half_zigzag(v3, f16c, zig_zag, dst);
    true
}

/// Vectorized version of `decode_lossy_dct_group`'s F16-output write-row
/// loop for a full 8-wide row.
pub(super) fn try_write_row_f16(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let (Some(v3), Some(f16c)) = (tier_v3(), tier_f16c()) else {
        return false;
    };
    avx2::write_row_f16(v3, f16c, row, to_linear, out_row)
}

/// Same as `try_write_row_f16`, but widens the linearized halves back to f32
/// (via a second `vcvtph2ps`) for F32-sample-type channels, matching the
/// scalar path's `linear.to_f32()`.
pub(super) fn try_write_row_f32(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let (Some(v3), Some(f16c)) = (tier_v3(), tier_f16c()) else {
        return false;
    };
    avx2::write_row_f32(v3, f16c, row, to_linear, out_row)
}

/// SSE2-only fallback for `try_write_row_f16`/`try_write_row_f32`
pub(super) fn try_write_row_f16_sse2(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let Some(v1) = tier_v1() else {
        return false;
    };
    sse2::write_row_f16(v1, row, to_linear, out_row)
}

/// SSE2-only fallback for `try_write_row_f32`, see `try_write_row_f16_sse2`.
pub(super) fn try_write_row_f32_sse2(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let Some(v1) = tier_v1() else {
        return false;
    };
    sse2::write_row_f32(v1, row, to_linear, out_row)
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
    let (Some(v3), Some(f16c)) = (tier_v3(), tier_f16c()) else {
        return None;
    };
    // `v3` (AVX2) gates this whole path, so base AVX must
    // already be present; the CSC step needs its own token since it was
    // ported to miraculix.
    let avx = miraculix_x86::avx().expect("AVX confirmed available by the AVX2 tier gate above");
    Some(avx2::decode_group_fused(v3, f16c, avx, ac, dc, width, height, to_linear, targets, out))
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
    let Some(v4) = tier_v4() else {
        return None;
    };
    let Some(f16c) = tier_f16c() else {
        return None;
    };
    // AVX-512 gates this path, so base AVX and AVX-512F are both already
    // present; both CSC tokens are needed.
    let avx = miraculix_x86::avx().expect("AVX confirmed available by the AVX-512 tier gate above");
    let avx512f =
        miraculix_x86::avx512f().expect("AVX-512F confirmed available by the AVX-512 tier gate above");
    Some(avx512::decode_group_fused(
        v4, f16c, avx, avx512f, ac, dc, width, height, to_linear, targets, out,
    ))
}
