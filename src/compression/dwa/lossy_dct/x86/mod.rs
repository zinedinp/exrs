// SIMD version of the DWA zig-zag undo, ported from OpenEXR's
// `fromHalfZigZag_f16c` (internal_dwa_simd.h): a fixed shuffle network instead
// of a scalar gather, then F16C to widen the halves to f32.
//
// Needs the `V3` tier (for its SSE2/SSSE3/SSE4.1 tokens) and `F16c`; if either
// is missing, the caller falls back to the scalar path.

use std::convert::TryInto;

use pulp::core_arch::x86::F16c;
use pulp::x86::V3;

pub(super) fn try_from_half_zigzag(zig_zag: &[u16; 64], dst: &mut [f32; 64]) -> bool {
    if let (Some(v3), Some(f16c)) = (V3::try_new(), F16c::try_new()) {
        from_half_zigzag(v3, f16c, zig_zag, dst);
        true
    } else {
        false
    }
}

// Rounding-mode immediate for `vcvtps2ph`: `_MM_FROUND_TO_NEAREST_INT` (0).
// Must match the immediate `half::f16::from_f32`'s own F16C fast path uses
// (half's `arch/x86.rs`), so the two conversions are bit-identical.
const ROUND_TO_NEAREST: i32 = 0;

// Mirrors OpenEXR's `LossyDctDecoder_execute` SSE2 fast path
// (internal_dwa_decoder.h): one `vcvtps2ph` converts a full 8-wide row of DCT
// output to nonlinear half bits (matching `half::f16::from_f32`'s own F16C
// path bit-for-bit), each lane is extracted to a GPR to index `to_linear`
fn linearize_lanes(
    v3: V3,
    bits: std::arch::x86_64::__m128i,
    to_linear: &[u16; 65536],
) -> std::arch::x86_64::__m128i {
    let sse2 = v3.sse2;

    let i0 = sse2._mm_extract_epi16::<0>(bits);
    let i1 = sse2._mm_extract_epi16::<1>(bits);
    let i2 = sse2._mm_extract_epi16::<2>(bits);
    let i3 = sse2._mm_extract_epi16::<3>(bits);
    let i4 = sse2._mm_extract_epi16::<4>(bits);
    let i5 = sse2._mm_extract_epi16::<5>(bits);
    let i6 = sse2._mm_extract_epi16::<6>(bits);
    let i7 = sse2._mm_extract_epi16::<7>(bits);

    // `_mm_extract_epi16` zero-extends, so each `iN` is already a valid
    // 0..=65535 table index.
    let r0 = to_linear[i0 as usize] as i32;
    let r1 = to_linear[i1 as usize] as i32;
    let r2 = to_linear[i2 as usize] as i32;
    let r3 = to_linear[i3 as usize] as i32;
    let r4 = to_linear[i4 as usize] as i32;
    let r5 = to_linear[i5 as usize] as i32;
    let r6 = to_linear[i6 as usize] as i32;
    let r7 = to_linear[i7 as usize] as i32;

    let v = sse2._mm_insert_epi16::<0>(sse2._mm_setzero_si128(), r0);
    let v = sse2._mm_insert_epi16::<1>(v, r1);
    let v = sse2._mm_insert_epi16::<2>(v, r2);
    let v = sse2._mm_insert_epi16::<3>(v, r3);
    let v = sse2._mm_insert_epi16::<4>(v, r4);
    let v = sse2._mm_insert_epi16::<5>(v, r5);
    let v = sse2._mm_insert_epi16::<6>(v, r6);
    let v = sse2._mm_insert_epi16::<7>(v, r7);
    v
}

/// Vectorized version of `decode_lossy_dct_group`'s F16-output write-row
/// loop for a full 8-wide row.
pub(super) fn try_write_row_f16(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let (Some(v3), Some(f16c)) = (V3::try_new(), F16c::try_new()) else {
        return false;
    };
    let Ok(&row): Result<&[f32; 8], _> = row.try_into() else {
        return false;
    };
    if out_row.len() != 16 {
        return false;
    }

    let vec: std::arch::x86_64::__m256 = pulp::cast!(row);
    let nonlinear = f16c._mm256_cvtps_ph::<ROUND_TO_NEAREST>(vec);
    let linear = match to_linear {
        Some(table) => linearize_lanes(v3, nonlinear, table),
        None => nonlinear,
    };
    let bytes: [u8; 16] = pulp::cast!(linear);
    out_row.copy_from_slice(&bytes);
    true
}

/// Same as `try_write_row_f16`, but widens the linearized halves back to f32
/// (via a second `vcvtph2ps`) for F32-sample-type channels, matching the
/// scalar path's `linear.to_f32()`.
pub(super) fn try_write_row_f32(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let (Some(v3), Some(f16c)) = (V3::try_new(), F16c::try_new()) else {
        return false;
    };
    let Ok(&row): Result<&[f32; 8], _> = row.try_into() else {
        return false;
    };
    if out_row.len() != 32 {
        return false;
    }

    let vec: std::arch::x86_64::__m256 = pulp::cast!(row);
    let nonlinear = f16c._mm256_cvtps_ph::<ROUND_TO_NEAREST>(vec);
    let linear = match to_linear {
        Some(table) => linearize_lanes(v3, nonlinear, table),
        None => nonlinear,
    };
    let widened = f16c._mm256_cvtph_ps(linear);
    let bytes: [u8; 32] = pulp::cast!(widened);
    out_row.copy_from_slice(&bytes);
    true
}

fn from_half_zigzag(v3: V3, f16c: F16c, src: &[u16; 64], dst: &mut [f32; 64]) {
    let sse2 = v3.sse2;
    let ssse3 = v3.ssse3;
    let sse4_1 = v3.sse4_1;

    let e = |i: usize| src[i] as i16;
    let setr = |a: usize, b: usize, c: usize, d: usize, f: usize, g: usize, h: usize, i: usize| {
        sse2._mm_setr_epi16(e(a), e(b), e(c), e(d), e(f), e(g), e(h), e(i))
    };

    // x8 <- [0-7]; x6 <- [56-63]; x9 <- [21-28]; x7 <- [28-35]; x3 <- [6-9,54-57]
    let xmm8 = setr(0, 1, 2, 3, 4, 5, 6, 7);
    let xmm6_init = setr(56, 57, 58, 59, 60, 61, 62, 63);
    let xmm9 = setr(21, 22, 23, 24, 25, 26, 27, 28);
    let xmm7 = setr(28, 29, 30, 31, 32, 33, 34, 35);
    let xmm3 = setr(6, 7, 8, 9, 54, 55, 56, 57);

    let mem70 = setr(35, 36, 37, 38, 39, 40, 41, 42);
    let mem82 = setr(41, 42, 43, 44, 45, 46, 47, 48);
    let mem98 = setr(49, 50, 51, 52, 53, 54, 55, 56);
    let mem14 = setr(7, 8, 9, 10, 11, 12, 13, 14);
    let mem30 = setr(15, 16, 17, 18, 19, 20, 21, 22);

    // Setup rows 0-2 of A in xmm0-xmm2
    let xmm1 = sse2._mm_srli_si128::<2>(xmm8);
    let xmm2 = sse2._mm_slli_si128::<4>(xmm8);
    let xmm0 = ssse3._mm_alignr_epi8::<2>(xmm8, mem70);
    let xmm1 = sse4_1._mm_blend_epi16::<0xfc>(xmm1, mem82);
    let xmm2 = sse4_1._mm_blend_epi16::<0x1f>(xmm2, mem98);

    // Setup rows 4-6 of A in xmm4-xmm6
    let xmm4 = sse2._mm_srli_si128::<4>(xmm6_init);
    let xmm5 = sse2._mm_slli_si128::<2>(xmm6_init);
    let xmm6 = ssse3._mm_alignr_epi8::<14>(xmm9, xmm6_init);
    let xmm4 = sse4_1._mm_blend_epi16::<0xf8>(xmm4, mem14);
    let xmm5 = sse4_1._mm_blend_epi16::<0x3f>(xmm5, mem30);

    // Reverse the even rows (pshuflw+pshufhw+pshufd with 0x1b/0x1b/0x4e is a
    // full 8-lane reversal, confirmed by hand-tracing the immediate fields).
    let reverse = |v: std::arch::x86_64::__m128i| {
        let v = sse2._mm_shufflelo_epi16::<0x1b>(v);
        let v = sse2._mm_shufflehi_epi16::<0x1b>(v);
        sse2._mm_shuffle_epi32::<0x4e>(v)
    };
    let xmm0 = reverse(xmm0);
    let xmm2 = reverse(xmm2);
    let xmm4 = reverse(xmm4);
    let xmm6 = reverse(xmm6);

    // Transpose xmm0-xmm7 into xmm8-xmm15 (word stage)
    let t8 = sse2._mm_unpacklo_epi16(xmm0, xmm1);
    let t9 = sse2._mm_unpacklo_epi16(xmm2, xmm3);
    let t10 = sse2._mm_unpacklo_epi16(xmm4, xmm5);
    let t11 = sse2._mm_unpacklo_epi16(xmm6, xmm7);
    let t12 = sse2._mm_unpackhi_epi16(xmm0, xmm1);
    let t13 = sse2._mm_unpackhi_epi16(xmm2, xmm3);
    let t14 = sse2._mm_unpackhi_epi16(xmm4, xmm5);
    let t15 = sse2._mm_unpackhi_epi16(xmm6, xmm7);

    // dword stage
    let u0 = sse2._mm_unpacklo_epi32(t8, t9);
    let u1 = sse2._mm_unpacklo_epi32(t10, t11);
    let u2 = sse2._mm_unpackhi_epi32(t8, t9);
    let u3 = sse2._mm_unpackhi_epi32(t10, t11);
    let u4 = sse2._mm_unpacklo_epi32(t12, t13);
    let u5 = sse2._mm_unpacklo_epi32(t14, t15);
    let u6 = sse2._mm_unpackhi_epi32(t12, t13);
    let u7 = sse2._mm_unpackhi_epi32(t14, t15);

    // qword stage
    let v8 = sse2._mm_unpacklo_epi64(u0, u1);
    let v9 = sse2._mm_unpackhi_epi64(u0, u1);
    let v10 = sse2._mm_unpacklo_epi64(u2, u3);
    let v11 = sse2._mm_unpackhi_epi64(u2, u3);
    let v12 = sse2._mm_unpacklo_epi64(u5, u4);
    let v13 = sse2._mm_unpackhi_epi64(u4, u5);
    let v14 = sse2._mm_unpacklo_epi64(u6, u7);
    let v15 = sse2._mm_unpackhi_epi64(u6, u7);

    // Rotate the rows to get the correct final order (v8, v12 need no rotation).
    let v9 = ssse3._mm_alignr_epi8::<2>(v9, v9);
    let v10 = ssse3._mm_alignr_epi8::<4>(v10, v10);
    let v11 = ssse3._mm_alignr_epi8::<6>(v11, v11);
    let v13 = ssse3._mm_alignr_epi8::<10>(v13, v13);
    let v14 = ssse3._mm_alignr_epi8::<12>(v14, v14);
    let v15 = ssse3._mm_alignr_epi8::<14>(v15, v15);

    // Widen each permuted row of 8 halves to 8 f32 with a single `vcvtph2ps`,
    // exactly as the OpenEXR original does.
    let store8 = |reg: std::arch::x86_64::__m128i, out: &mut [f32]| {
        let wide: [f32; 8] = pulp::cast!(f16c._mm256_cvtph_ps(reg));
        out.copy_from_slice(&wide);
    };
    store8(v8, &mut dst[0..8]);
    store8(v9, &mut dst[8..16]);
    store8(v10, &mut dst[16..24]);
    store8(v11, &mut dst[24..32]);
    store8(v12, &mut dst[32..40]);
    store8(v13, &mut dst[40..48]);
    store8(v14, &mut dst[48..56]);
    store8(v15, &mut dst[56..64]);
}

// Requires a host with AVX2 + F16C, hence gated behind the same opt-in feature
// as the DCT tier tests: `cargo test --lib --features avx2-tests -- zigzag`
#[cfg(all(test, feature = "avx2-tests"))]
mod test {
    use super::super::quantization::ZIGZAG_ORDER;
    use super::super::transfer_curve::to_linear_table;
    use half::f16;

    /// The permuted+widened output must be bit-identical to the scalar gather,
    /// including NaN payloads `vcvtph2ps` and `half`'s conversion must not
    /// be allowed to disagree on any input. Every one of the 65536 possible
    /// half bit patterns is covered, in every one of the 64 lanes.
    #[test]
    fn zigzag_simd_matches_scalar() {
        let mut simd = [0.0f32; 64];
        let mut scalar = [0.0f32; 64];

        for base in 0..=u16::MAX {
            let mut zig_zag = [0u16; 64];
            for (lane, slot) in zig_zag.iter_mut().enumerate() {
                // A stride coprime with 65536 walks every lane through every
                // bit pattern as `base` advances.
                *slot = base.wrapping_add(lane as u16 * 1013);
            }

            assert!(super::try_from_half_zigzag(&zig_zag, &mut simd));
            for (slot, &src_index) in scalar.iter_mut().zip(ZIGZAG_ORDER.iter()) {
                *slot = f16::from_bits(zig_zag[src_index]).to_f32();
            }

            for (index, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "mismatch at index {index} for base {base}"
                );
            }
        }
    }

    /// Scalar reference for the write-row SIMD path: exactly what
    /// `decode_lossy_dct_group`'s `write_row!` macro computes per pixel.
    fn scalar_linear_bits(value: f32, to_linear: Option<&[u16; 65536]>) -> u16 {
        let nonlinear = f16::from_f32(value);
        match to_linear {
            Some(table) => table[nonlinear.to_bits() as usize],
            None => nonlinear.to_bits(),
        }
    }

    /// A wide, pseudo-random sweep across the full f32 bit space (every
    /// exponent/mantissa/sign region gets hit, not just small values near
    /// zero), plus explicit special values DCT output could plausibly
    /// produce or a table lookup could return.
    fn sweep_rows() -> impl Iterator<Item = [f32; 8]> {
        let special = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            65504.0,  // half::MAX
            65520.0,  // rounds to infinity in half
            -65504.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc00001), // NaN, alternate payload
            f32::from_bits(0x00000001), // smallest positive subnormal
            f32::from_bits(0x0000ffff), // subnormal, near boundary
            6.104_f32.exp2() * f32::EPSILON, // arbitrary small-normal-ish value
        ];
        let special_rows: Vec<[f32; 8]> = special
            .chunks(8)
            .map(|chunk| {
                let mut row = [0.0f32; 8];
                row[..chunk.len()].copy_from_slice(chunk);
                row
            })
            .collect();

        let swept_rows = (0u32..=0xFFFF).map(|base| {
            std::array::from_fn(|lane| {
                // A large odd stride spreads `lane` across every exponent
                // range as `base` walks the low bits, unlike a plain +lane
                // which would only ever perturb the mantissa.
                f32::from_bits(base.wrapping_add(lane as u32 * 0x1000_0001))
            })
        });

        special_rows.into_iter().chain(swept_rows)
    }

    fn assert_bits_match(a: u16, b: u16, context: &str) {
        // NaN half bit patterns aren't unique (many payloads map to "NaN"),
        // so only require both sides agree on NaN-ness, matching `half`'s
        // own equality semantics; everything else must be bit-exact.
        let a_nan = f16::from_bits(a).is_nan();
        let b_nan = f16::from_bits(b).is_nan();
        if a_nan || b_nan {
            assert_eq!(
                a_nan, b_nan,
                "{}: one side is NaN, the other isn't (a=0x{:04x}, b=0x{:04x})",
                context, a, b
            );
        } else {
            assert_eq!(a, b, "{}: bit mismatch (a=0x{:04x}, b=0x{:04x})", context, a, b);
        }
    }

    #[test]
    fn write_row_f16_matches_scalar() {
        for to_linear in [None, Some(to_linear_table())] {
            for row in sweep_rows() {
                let mut simd = [0u8; 16];
                assert!(super::try_write_row_f16(&row, to_linear, &mut simd));

                for (lane, &value) in row.iter().enumerate() {
                    let expected = scalar_linear_bits(value, to_linear);
                    let actual = u16::from_le_bytes([simd[lane * 2], simd[lane * 2 + 1]]);
                    assert_bits_match(
                        actual,
                        expected,
                        &format!("f16 lane {lane}, value {value:e}, to_linear={}", to_linear.is_some()),
                    );
                }
            }
        }
    }

    #[test]
    fn write_row_f32_matches_scalar() {
        for to_linear in [None, Some(to_linear_table())] {
            for row in sweep_rows() {
                let mut simd = [0u8; 32];
                assert!(super::try_write_row_f32(&row, to_linear, &mut simd));

                for (lane, &value) in row.iter().enumerate() {
                    let expected = f16::from_bits(scalar_linear_bits(value, to_linear)).to_f32();
                    let actual = f32::from_le_bytes([
                        simd[lane * 4],
                        simd[lane * 4 + 1],
                        simd[lane * 4 + 2],
                        simd[lane * 4 + 3],
                    ]);
                    if expected.is_nan() {
                        assert!(
                            actual.is_nan(),
                            "f32 lane {}, value {:e}: expected NaN, got {:e}",
                            lane, value, actual,
                        );
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "f32 lane {}, value {:e}, to_linear={}",
                            lane, value, to_linear.is_some()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn write_row_falls_back_for_short_rows() {
        let row = [0.0f32; 7];
        let mut out16 = [0u8; 14];
        let mut out32 = [0u8; 28];
        assert!(!super::try_write_row_f16(&row, None, &mut out16));
        assert!(!super::try_write_row_f32(&row, None, &mut out32));
    }
}
