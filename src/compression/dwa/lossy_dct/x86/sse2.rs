// SSE2-only write-row: covers hosts with no AVX2/F16C (the strip-tiled
// fallback's write-row step currently has no vectorization at all below the
// AVX2 tier -- everything falls to scalar per-pixel `linearize_scalar`).
//
// SSE2 has no F16C, so unlike `avx2::write_row_f16`/`write_row_f32` there is
// no hardware `vcvtps2ph`/`vcvtph2ps`.
// The `to_linear` lookup reuses `avx2::linearize_lanes` unchanged

use std::convert::TryInto;

use pulp::core_arch::x86::{Sse, Sse2};
use pulp::x86::V1;

use super::avx2::linearize_lanes;

/// `sign | mantissa_result`, where `mantissa_result` is either the
/// finite-path result or the inf/NaN fixup
#[inline(always)]
fn blend_epi32(
    sse2: Sse2,
    mask: std::arch::x86_64::__m128i,
    if_true: std::arch::x86_64::__m128i,
    if_false: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m128i {
    sse2._mm_or_si128(
        sse2._mm_and_si128(mask, if_true),
        sse2._mm_andnot_si128(mask, if_false),
    )
}

/// Narrows 4 `u32` lanes (each holding a value in `0..=0xffff`) to 4 `u16`
/// lanes, packed into the low 64 bits of the result (high 64 bits are a
/// duplicate, from `_mm_packs_epi32`'s 2-input shape). SSE2 has no unsigned
/// 32->16 saturating pack (that's SSE4.1's `packus_epi32`); this uses the
/// standard bias-then-xor trick: subtracting 0x8000 brings `0..=0xffff` into
/// signed 16-bit range so `_mm_packs_epi32`,
/// doesn't saturate, and adding/subtracting 0x8000 mod 65536 is exactly an
/// XOR of bit 15
#[inline(always)]
fn narrow_u32x4_to_u16x4_biased(
    sse2: Sse2,
    v: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m128i {
    let biased = sse2._mm_sub_epi32(v, sse2._mm_set1_epi32(0x8000));
    let packed = sse2._mm_packs_epi32(biased, biased);
    sse2._mm_xor_si128(packed, sse2._mm_set1_epi16(-0x8000i16))
}

/// Lane-wise, branchless, round-to-nearest-even f32->f16 bit conversion for
/// 4 lanes (an 8-wide row needs two calls, one per half, SSE2's `__m128`
/// is 4-wide, unlike AVX2's 8-wide `__m256`).
#[inline(always)]
fn f32_bits_to_f16_bits_x4(
    sse: Sse,
    sse2: Sse2,
    x_in: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m128i {
    const F16_OVERFLOW_EXP_THRESHOLD: i32 = (127 + 16) << 23;
    const F32_INFINITY_BITS: i32 = 0xFFi32 << 23;
    const F16_SUBNORMAL_THRESHOLD: i32 = 113i32 << 23;
    const DENORM_MAGIC_BITS: i32 = ((127 - 15) + (23 - 10) + 1) << 23;
    const BIAS_ADJUST: i32 = ((15i32 - 127) << 23).wrapping_add(0xfff);

    let sign_mask = sse2._mm_set1_epi32(i32::MIN);
    let x_sgn = sse2._mm_and_si128(x_in, sign_mask);
    let x = sse2._mm_xor_si128(x_in, x_sgn); // abs-value bits; top bit now 0

    let is_overflow = sse2._mm_cmpgt_epi32(x, sse2._mm_set1_epi32(F16_OVERFLOW_EXP_THRESHOLD - 1));
    let is_nan = sse2._mm_cmpgt_epi32(x, sse2._mm_set1_epi32(F32_INFINITY_BITS));
    let inf_or_nan_result = blend_epi32(
        sse2,
        is_nan,
        sse2._mm_set1_epi32(0x7e00),
        sse2._mm_set1_epi32(0x7c00),
    );

    // Denormal/zero path: add a magic float so IEEE round-to-nearest-even
    // addition does the mantissa rounding for us, then subtract the magic
    // bits back off.
    let denorm_magic_bits = sse2._mm_set1_epi32(DENORM_MAGIC_BITS);
    let x_f = sse2._mm_castsi128_ps(x);
    let magic_f = sse2._mm_castsi128_ps(denorm_magic_bits);
    let denorm_sum_bits = sse2._mm_castps_si128(sse._mm_add_ps(x_f, magic_f));
    let denorm_result = sse2._mm_sub_epi32(denorm_sum_bits, denorm_magic_bits);

    // Normal path: rebias the exponent and round-to-nearest-even via the
    // "add 0xfff plus the truncated bit" trick.
    let mant_odd = sse2._mm_and_si128(sse2._mm_srli_epi32::<13>(x), sse2._mm_set1_epi32(1));
    let x_adjusted = sse2._mm_add_epi32(
        sse2._mm_add_epi32(x, sse2._mm_set1_epi32(BIAS_ADJUST)),
        mant_odd,
    );
    let normal_result = sse2._mm_srli_epi32::<13>(x_adjusted);

    let is_denorm = sse2._mm_cmplt_epi32(x, sse2._mm_set1_epi32(F16_SUBNORMAL_THRESHOLD));
    let finite_result = blend_epi32(sse2, is_denorm, denorm_result, normal_result);
    let mantissa_result = blend_epi32(sse2, is_overflow, inf_or_nan_result, finite_result);

    let signed = sse2._mm_or_si128(mantissa_result, sse2._mm_srli_epi32::<16>(x_sgn));
    narrow_u32x4_to_u16x4_biased(sse2, signed)
}

/// Lane-wise f16->f32 bit widening for 4 lanes, each already zero-extended
/// into the low 16 bits of a 32-bit lane (see call sites: `_mm_unpacklo_epi16`
/// / `_mm_unpackhi_epi16` against a zero register).
#[inline(always)]
fn f16_bits_to_f32_bits_x4(
    sse: Sse,
    sse2: Sse2,
    h_in: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m128i {
    const MAGIC_EXP_BITS: i32 = (254 - 15) << 23;
    const INFNAN_THRESHOLD_BITS: i32 = (127 + 16) << 23;

    let sign = sse2._mm_slli_epi32::<16>(sse2._mm_and_si128(h_in, sse2._mm_set1_epi32(0x8000)));
    let value_bits = sse2._mm_slli_epi32::<13>(sse2._mm_and_si128(h_in, sse2._mm_set1_epi32(0x7fff)));

    let value_f = sse2._mm_castsi128_ps(value_bits);
    let magic_f = sse2._mm_castsi128_ps(sse2._mm_set1_epi32(MAGIC_EXP_BITS));
    let scaled_bits = sse2._mm_castps_si128(sse._mm_mul_ps(value_f, magic_f));

    let is_infnan = sse2._mm_cmpgt_epi32(scaled_bits, sse2._mm_set1_epi32(INFNAN_THRESHOLD_BITS - 1));
    let infnan_exp_fixup = sse2._mm_and_si128(is_infnan, sse2._mm_set1_epi32(0xFFi32 << 23));

    sse2._mm_or_si128(sse2._mm_or_si128(scaled_bits, infnan_exp_fixup), sign)
}

/// SSE2 analog of `x86::avx2::write_row_f16`. `row` must be exactly 8 `f32`
/// DCT-output samples; `out_row` exactly 16 bytes (8 half-float samples).
#[inline(always)]
pub(super) fn write_row_f16(
    v1: V1,
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let Ok(&row): Result<&[f32; 8], _> = row.try_into() else {
        return false;
    };
    if out_row.len() != 16 {
        return false;
    }

    let sse = v1.sse;
    let sse2 = v1.sse2;

    let lo_bits: [i32; 4] = std::array::from_fn(|i| row[i].to_bits() as i32);
    let hi_bits: [i32; 4] = std::array::from_fn(|i| row[4 + i].to_bits() as i32);
    let lo: std::arch::x86_64::__m128i = pulp::cast!(lo_bits);
    let hi: std::arch::x86_64::__m128i = pulp::cast!(hi_bits);

    let lo_half = f32_bits_to_f16_bits_x4(sse, sse2, lo);
    let hi_half = f32_bits_to_f16_bits_x4(sse, sse2, hi);
    let nonlinear = sse2._mm_unpacklo_epi64(lo_half, hi_half);

    let linear = match to_linear {
        Some(table) => linearize_lanes(sse2, nonlinear, table),
        None => nonlinear,
    };

    let bytes: [u8; 16] = pulp::cast!(linear);
    out_row.copy_from_slice(&bytes);
    true
}

/// Same as `write_row_f16`, but widens the linearized halves back to `f32`
/// for F32-sample-type channels, matching `write_row_f16`'s AVX2.
#[inline(always)]
pub(super) fn write_row_f32(
    v1: V1,
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    let Ok(&row): Result<&[f32; 8], _> = row.try_into() else {
        return false;
    };
    if out_row.len() != 32 {
        return false;
    }

    let sse = v1.sse;
    let sse2 = v1.sse2;

    let lo_bits: [i32; 4] = std::array::from_fn(|i| row[i].to_bits() as i32);
    let hi_bits: [i32; 4] = std::array::from_fn(|i| row[4 + i].to_bits() as i32);
    let lo: std::arch::x86_64::__m128i = pulp::cast!(lo_bits);
    let hi: std::arch::x86_64::__m128i = pulp::cast!(hi_bits);

    let lo_half = f32_bits_to_f16_bits_x4(sse, sse2, lo);
    let hi_half = f32_bits_to_f16_bits_x4(sse, sse2, hi);
    let nonlinear = sse2._mm_unpacklo_epi64(lo_half, hi_half);

    let linear = match to_linear {
        Some(table) => linearize_lanes(sse2, nonlinear, table),
        None => nonlinear,
    };

    let zero = sse2._mm_setzero_si128();
    let linear_lo32 = sse2._mm_unpacklo_epi16(linear, zero);
    let linear_hi32 = sse2._mm_unpackhi_epi16(linear, zero);

    let widened_lo = f16_bits_to_f32_bits_x4(sse, sse2, linear_lo32);
    let widened_hi = f16_bits_to_f32_bits_x4(sse, sse2, linear_hi32);

    let bytes_lo: [u8; 16] = pulp::cast!(widened_lo);
    let bytes_hi: [u8; 16] = pulp::cast!(widened_hi);
    out_row[..16].copy_from_slice(&bytes_lo);
    out_row[16..].copy_from_slice(&bytes_hi);
    true
}

#[cfg(test)]
mod test {
    use half::f16;

    use super::super::super::transfer_curve::to_linear_table;
    use super::{write_row_f16, write_row_f32};
    use pulp::x86::V1;

    fn expect_sse2() -> V1 {
        V1::try_new().expect("SSE2 SIMD mode requested, but the SSE2 tier is unavailable")
    }

    fn scalar_linear_bits(value: f32, to_linear: Option<&[u16; 65536]>) -> u16 {
        let nonlinear = f16::from_f32(value);
        match to_linear {
            Some(table) => table[nonlinear.to_bits() as usize],
            None => nonlinear.to_bits(),
        }
    }

    /// Same sweep shape as `avx2::test::sweep_rows`, full f32 bit space plus
    /// a handful of named specials.
    fn sweep_rows() -> impl Iterator<Item = [f32; 8]> {
        let special = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            65504.0,
            65520.0,
            -65504.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc00001),
            f32::from_bits(0x00000001),
            f32::from_bits(0x0000ffff),
            6.104_f32.exp2() * f32::EPSILON,
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
                f32::from_bits(base.wrapping_add(lane as u32 * 0x1000_0001))
            })
        });

        special_rows.into_iter().chain(swept_rows)
    }

    fn assert_bits_match(a: u16, b: u16, context: &str) {
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
        let v1 = expect_sse2();
        for to_linear in [None, Some(to_linear_table())] {
            for row in sweep_rows() {
                let mut simd = [0u8; 16];
                assert!(write_row_f16(v1, &row, to_linear, &mut simd));

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
        let v1 = expect_sse2();
        for to_linear in [None, Some(to_linear_table())] {
            for row in sweep_rows() {
                let mut simd = [0u8; 32];
                assert!(write_row_f32(v1, &row, to_linear, &mut simd));

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
        let v1 = expect_sse2();
        let row = [0.0f32; 7];
        let mut out16 = [0u8; 14];
        let mut out32 = [0u8; 28];
        assert!(!write_row_f16(v1, &row, None, &mut out16));
        assert!(!write_row_f32(v1, &row, None, &mut out32));
    }
}
