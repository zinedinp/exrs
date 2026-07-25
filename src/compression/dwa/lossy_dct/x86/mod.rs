// SIMD version of the DWA zig-zag undo, ported from OpenEXR's
// `fromHalfZigZag_f16c` (internal_dwa_simd.h): a fixed shuffle network instead
// of a scalar gather, then F16C to widen the halves to f32.
//
// Needs the `V3` tier (for its SSE2/SSSE3/SSE4.1 tokens) and `F16c`; if either
// is missing, the caller falls back to the scalar path.

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
}
