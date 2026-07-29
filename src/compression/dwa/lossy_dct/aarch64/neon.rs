//! NEON sketch of the write-row step (`x86::avx2::write_row_f16`'s analog):
//! convert an 8-wide row of DCT output (`f32`) to nonlinear half bits, run it
//! through the optional `to_linear` table, and pack the result into an
//! output row of bytes.
//!
//! **UNTESTED. NOT WIRED INTO ANY DISPATCH PATH. NOT READY TO SHIP.**
//!
//! ## Why NEON needs more work than SSE2/AVX2 here
//!
//! x86's F16C extension gives `write_row_f16` a single hardware instruction
//! (`vcvtps2ph`) for the f32->f16 conversion. AArch64 NEON's hardware FP16
//! conversion (`FCVTN`/`vcvt_f16_f32`) is not currently wrapped by the local
//! pulp fork (`pulp::aarch64::Neon` has no `f16` conversion method
//! Rather than add an unverified wrapper for an
//! intrinsic that can't be exercised on this machine either, this sketch uses
//! a **software** round-to-nearest-even f32->f16 bit-trick instead, ported
//! lane-wise onto NEON integer ops that pulp already wraps (`vandq_u32`,
//! `vorrq_u32`, `vshlq_n_u32`/`vshrq_n_u32`, `vbslq_u32`, compares, etc.) —
//! the same category of technique this project already uses for scalar
//! fallbacks. Algorithm: the classic branchless "float_to_half" bit-trick
//! (Fabian Giesen / Bit Twiddling Hacks.
//!
//! Out of scope for this sketch: the `f16`-output-only path (`write_row_f16`
//! analog). An `F32`-output analog (`write_row_f32`) would additionally need
//! the *inverse* conversion (f16->f32 widening), not written here. The
//! DCT/CSC/zigzag kernels also have no NEON port in this tree at all

use std::convert::TryInto;

use core::arch::aarch64::{float32x4_t, uint16x4_t, uint16x8_t, uint32x4_t};
use pulp::aarch64::Neon;

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
    // addition does the mantissa rounding for us, then subtract the magic
    // bits back off.
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

/// NEON analog of `x86::avx2::write_row_f16`. `row` must be exactly 8 `f32`
/// DCT-output samples; `out_row` exactly 16 bytes (8 half-float samples).
/// Returns `false` (caller falls back to scalar) on any shape mismatch.
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
    let lo: uint32x4_t = pulp::cast!(lo_bits);
    let hi: uint32x4_t = pulp::cast!(hi_bits);

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

    let bytes: [u8; 16] = pulp::cast!(linear);
    out_row.copy_from_slice(&bytes);
    true
}

/// Bit-exactness check against `half::f16::from_f32`. Only compiles for
/// `target_arch = "aarch64"`
#[cfg(all(test, target_arch = "aarch64"))]
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
}
