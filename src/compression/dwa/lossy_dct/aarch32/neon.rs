//! 32-bit ARM NEON analog of `x86::sse2::write_row_f16`/`write_row_f32` (and
//! 1:1 port of `aarch64::neon`'s version of the same): convert an 8-wide row
//! of DCT output (`f32`) to nonlinear half bits, run it through the optional
//! `to_linear` table, and pack the result into an output row of bytes
//! (`write_row_f16`) or widen it back to `f32` first (`write_row_f32`).

use std::convert::TryInto;

use core::arch::arm::{float32x4_t, uint16x4_t, uint16x8_t, uint32x4_t};
use pulp::aarch32::Neon;

// `core::arch::arm`'s NEON vector types don't implement bytemuck `Pod`
// (unlike their aarch64 equivalents, which `pulp::cast!` handles directly),
// so the plain-array boundary conversions need `transmute` -> `#![forbid(unsafe_code)]`.
// `Neon::u32x4_from_bits` in pulp

/// Lane-wise, branchless, round-to-nearest-even f32->f16 bit conversion.
/// Ported onto 4 NEON lanes at once.
#[inline]
fn f32_bits_to_f16_bits_x4(simd: Neon, x_in: uint32x4_t) -> uint16x4_t {
    let neon = simd.neon;

    // Exponent-bias constants, derived symbolically (not copied as hex) so a
    // slipped digit can't hide silently in an unrunnable file.
    const F16_OVERFLOW_EXP_THRESHOLD: u32 = (127 + 16) << 23;
    const F32_INFINITY_BITS: u32 = 0xFFu32 << 23;
    const F16_SUBNORMAL_THRESHOLD: u32 = 113u32 << 23;
    const DENORM_MAGIC_BITS: u32 = ((127 - 15) + (23 - 10) + 1) << 23;
    const BIAS_ADJUST: u32 = (((15i32 - 127) << 23) as u32).wrapping_add(0xfff);

    let sign_mask = neon.vdupq_n_u32(0x8000_0000);
    let x_sgn = neon.vandq_u32(x_in, sign_mask);
    let x = neon.veorq_u32(x_in, x_sgn); // == abs-value bits

    let is_overflow = neon.vcgeq_u32(x, neon.vdupq_n_u32(F16_OVERFLOW_EXP_THRESHOLD));
    let is_nan = neon.vcgtq_u32(x, neon.vdupq_n_u32(F32_INFINITY_BITS));
    let inf_or_nan_result =
        neon.vbslq_u32(is_nan, neon.vdupq_n_u32(0x7e00), neon.vdupq_n_u32(0x7c00));

    // Denormal/zero path: add a magic float so IEEE round-to-nearest-even
    // addition does the mantissa rounding for us.
    let denorm_magic_bits = neon.vdupq_n_u32(DENORM_MAGIC_BITS);
    let x_f: float32x4_t = neon.vreinterpretq_f32_u32(x);
    let magic_f: float32x4_t = neon.vreinterpretq_f32_u32(denorm_magic_bits);
    let denorm_sum_bits = neon.vreinterpretq_u32_f32(neon.vaddq_f32(x_f, magic_f));
    let denorm_result = neon.vsubq_u32(denorm_sum_bits, denorm_magic_bits);

    // Normal path: rebias the exponent and round-to-nearest-even via the
    // "add 0xfff plus the truncated bit" trick.
    let mant_odd = neon.vandq_u32(neon.vshrq_n_u32::<13>(x), neon.vdupq_n_u32(1));
    let x_adjusted = neon.vaddq_u32(
        neon.vaddq_u32(x, neon.vdupq_n_u32(BIAS_ADJUST)),
        mant_odd,
    );
    let normal_result = neon.vshrq_n_u32::<13>(x_adjusted);

    let is_denorm = neon.vcltq_u32(x, neon.vdupq_n_u32(F16_SUBNORMAL_THRESHOLD));
    let finite_result = neon.vbslq_u32(is_denorm, denorm_result, normal_result);
    let mantissa_result = neon.vbslq_u32(is_overflow, inf_or_nan_result, finite_result);

    let signed = neon.vorrq_u32(mantissa_result, neon.vshrq_n_u32::<16>(x_sgn));
    neon.vmovn_u32(signed)
}

/// Lane-wise f16->f32 bit widening for 4 lanes, each already zero-extended
/// into a `uint32x4_t` lane
#[inline]
fn f16_bits_to_f32_bits_x4(simd: Neon, h_in: uint32x4_t) -> uint32x4_t {
    let neon = simd.neon;

    const MAGIC_EXP_BITS: u32 = (254 - 15) << 23;
    const INFNAN_THRESHOLD_BITS: u32 = (127 + 16) << 23;

    let sign = neon.vshlq_n_u32::<16>(neon.vandq_u32(h_in, neon.vdupq_n_u32(0x8000)));
    let value_bits = neon.vshlq_n_u32::<13>(neon.vandq_u32(h_in, neon.vdupq_n_u32(0x7fff)));

    let value_f: float32x4_t = neon.vreinterpretq_f32_u32(value_bits);
    let magic_f: float32x4_t = neon.vreinterpretq_f32_u32(neon.vdupq_n_u32(MAGIC_EXP_BITS));
    let scaled_bits = neon.vreinterpretq_u32_f32(neon.vmulq_f32(value_f, magic_f));

    let is_infnan = neon.vcgtq_u32(scaled_bits, neon.vdupq_n_u32(INFNAN_THRESHOLD_BITS - 1));
    let infnan_exp_fixup = neon.vandq_u32(is_infnan, neon.vdupq_n_u32(0xFFu32 << 23));

    neon.vorrq_u32(neon.vorrq_u32(scaled_bits, infnan_exp_fixup), sign)
}

/// NEON analog of `x86::avx2::write_row_f16`. `row` must be exactly 8 `f32`
/// DCT-output samples; `out_row` exactly 16 bytes (8 half-float samples).
#[inline]
pub fn write_row_f16(
    simd: Neon,
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

    let neon = simd.neon;

    let lo_bits: [u32; 4] = std::array::from_fn(|i| row[i].to_bits());
    let hi_bits: [u32; 4] = std::array::from_fn(|i| row[4 + i].to_bits());
    let lo: uint32x4_t = simd.u32x4_from_bits(lo_bits);
    let hi: uint32x4_t = simd.u32x4_from_bits(hi_bits);

    let lo_half = f32_bits_to_f16_bits_x4(simd, lo);
    let hi_half = f32_bits_to_f16_bits_x4(simd, hi);
    let nonlinear: uint16x8_t = neon.vcombine_u16(lo_half, hi_half);

    let linear = match to_linear {
        Some(table) => {
            // Same scalar extract/load/insert shape as x86's
            // `linearize_lanes`
            let mut v = nonlinear;
            v = neon.vsetq_lane_u16::<0>(table[neon.vgetq_lane_u16::<0>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<1>(table[neon.vgetq_lane_u16::<1>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<2>(table[neon.vgetq_lane_u16::<2>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<3>(table[neon.vgetq_lane_u16::<3>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<4>(table[neon.vgetq_lane_u16::<4>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<5>(table[neon.vgetq_lane_u16::<5>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<6>(table[neon.vgetq_lane_u16::<6>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<7>(table[neon.vgetq_lane_u16::<7>(nonlinear) as usize], v);
            v
        }
        None => nonlinear,
    };

    let bytes: [u8; 16] = simd.u16x8_to_bytes(linear);
    out_row.copy_from_slice(&bytes);
    true
}

/// Same as `write_row_f16`, but widens the linearized halves back to `f32`
/// for F32-sample-type channels, matching `x86::sse2::write_row_f32`.
#[inline]
pub fn write_row_f32(
    simd: Neon,
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

    let neon = simd.neon;

    let lo_bits: [u32; 4] = std::array::from_fn(|i| row[i].to_bits());
    let hi_bits: [u32; 4] = std::array::from_fn(|i| row[4 + i].to_bits());
    let lo: uint32x4_t = simd.u32x4_from_bits(lo_bits);
    let hi: uint32x4_t = simd.u32x4_from_bits(hi_bits);

    let lo_half = f32_bits_to_f16_bits_x4(simd, lo);
    let hi_half = f32_bits_to_f16_bits_x4(simd, hi);
    let nonlinear: uint16x8_t = neon.vcombine_u16(lo_half, hi_half);

    let linear = match to_linear {
        Some(table) => {
            let mut v = nonlinear;
            v = neon.vsetq_lane_u16::<0>(table[neon.vgetq_lane_u16::<0>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<1>(table[neon.vgetq_lane_u16::<1>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<2>(table[neon.vgetq_lane_u16::<2>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<3>(table[neon.vgetq_lane_u16::<3>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<4>(table[neon.vgetq_lane_u16::<4>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<5>(table[neon.vgetq_lane_u16::<5>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<6>(table[neon.vgetq_lane_u16::<6>(nonlinear) as usize], v);
            v = neon.vsetq_lane_u16::<7>(table[neon.vgetq_lane_u16::<7>(nonlinear) as usize], v);
            v
        }
        None => nonlinear,
    };

    // Zero-extend each half of the u16x8 into its own u32x4 lane (`vmovl_u16`
    // is NEON's direct widen; x86 SSE2 emulates the same thing with
    // `_mm_unpacklo/hi_epi16` against a zero register).
    let linear_lo32 = neon.vmovl_u16(neon.vget_low_u16(linear));
    let linear_hi32 = neon.vmovl_u16(neon.vget_high_u16(linear));

    let widened_lo = f16_bits_to_f32_bits_x4(simd, linear_lo32);
    let widened_hi = f16_bits_to_f32_bits_x4(simd, linear_hi32);

    let bytes_lo: [u8; 16] = simd.u32x4_to_bytes(widened_lo);
    let bytes_hi: [u8; 16] = simd.u32x4_to_bytes(widened_hi);
    out_row[..16].copy_from_slice(&bytes_lo);
    out_row[16..].copy_from_slice(&bytes_hi);
    true
}

/// Bit-exactness check against `half::f16::from_f32`.
#[cfg(all(test, target_arch = "arm", feature = "arm-neon"))]
mod test {
    use super::*;
    use half::f16;

    #[test]
    fn write_row_f16_matches_scalar_no_table() {
        let simd = Neon::try_new().expect("NEON requested but unavailable");

        let mut rng_state: u32 = 0x2545F491;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state
        };

        // Special values first (signs, zero, inf, subnormal boundary, values
        // that exercise the mant_odd rounding tie-break), then random sweep.
        let specials: [f32; 8] = [0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 65504.0];
        let mut cases: Vec<[f32; 8]> = vec![specials];
        for _ in 0..10_000 {
            cases.push(std::array::from_fn(|_| {
                let bits = next();
                f32::from_bits(bits)
            }));
        }

        for row in cases {
            let mut simd_out = [0u8; 16];
            assert!(write_row_f16(simd, &row, None, &mut simd_out));

            for (lane, &value) in row.iter().enumerate() {
                let expected = f16::from_f32(value).to_bits();
                let actual_bytes = [simd_out[lane * 2], simd_out[lane * 2 + 1]];
                let actual = u16::from_le_bytes(actual_bytes);
                // NaN payloads aren't required to match bit-for-bit, only
                // "is NaN" -> matches this project's convention elsewhere.
                if expected & 0x7c00 == 0x7c00 && expected & 0x03ff != 0 {
                    assert!(
                        actual & 0x7c00 == 0x7c00 && actual & 0x03ff != 0,
                        "NaN lane {lane}: got {actual:#06x}"
                    );
                } else {
                    assert_eq!(actual, expected, "lane {lane}, value {value}: got {actual:#06x}, want {expected:#06x}");
                }
            }
        }
    }

    /// `write_row_f32` must round-trip through half precision the same way
    /// `write_row_f16` does, then widen back to `f32`. i.e. match
    /// `f16::from_f32(value).to_f32()` bit-for-bit.
    #[test]
    fn write_row_f32_matches_scalar_no_table() {
        let simd = Neon::try_new().expect("NEON requested but unavailable");

        let mut rng_state: u32 = 0x2545F491;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state
        };

        let specials: [f32; 8] = [0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 65504.0];
        let mut cases: Vec<[f32; 8]> = vec![specials];
        for _ in 0..10_000 {
            cases.push(std::array::from_fn(|_| f32::from_bits(next())));
        }

        for row in cases {
            let mut simd_out = [0u8; 32];
            assert!(write_row_f32(simd, &row, None, &mut simd_out));

            for (lane, &value) in row.iter().enumerate() {
                let expected = f16::from_f32(value).to_f32().to_bits();
                let actual_bytes: [u8; 4] = simd_out[lane * 4..lane * 4 + 4].try_into().unwrap();
                let actual = u32::from_le_bytes(actual_bytes);
                if expected & 0x7f80_0000 == 0x7f80_0000 && expected & 0x007f_ffff != 0 {
                    assert!(
                        actual & 0x7f80_0000 == 0x7f80_0000 && actual & 0x007f_ffff != 0,
                        "NaN lane {lane}: got {actual:#010x}"
                    );
                } else {
                    assert_eq!(actual, expected, "lane {lane}, value {value}: got {actual:#010x}, want {expected:#010x}");
                }
            }
        }
    }
}
