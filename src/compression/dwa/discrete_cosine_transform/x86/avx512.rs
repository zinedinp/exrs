// AVX-512 (V4) tier: processes 2 blocks per 512-bit register instead of
// inventing a new 16-lane-interleaved algorithm. Block A lives in float lanes
// 0-7 (the low 256 bits, i.e. sub-lanes 0-1 of the four 128-bit sub-lanes a
// zmm register has), block B in lanes 8-15 (sub-lanes 2-3). Every op below
// except the final transpose recombination is lane-preserving (never crosses
// a 128-bit sub-lane boundary), so `row_pass`/`column_pass`'s butterfly math
// is a mechanical width-doubling of `avx2.rs` — same coefficients, same
// shape, just running block A's and block B's arithmetic side by side in one
// instruction instead of two.
//
// Transpose recombine (AVX2's `_mm256_permute2f128_ps`):
// - `_mm512_shuffle_f32x4` alone cannot express a-b-a-b (dest 128-bit lanes
//   0-1 always from a, 2-3 from b).
// - A pure zmm shuffle chain (or `_mm512_permutex2var_ps`) is bit-exact, but
//   LLVM rewrites it into `vpermt2pd`, which is ~1.8× slower than the AVX2
//   batch on Zen5.
// - Instead: split each zmm into its two ymm halves, run the same
//   `permute2f128` the AVX2 path already uses, and re-pack. That keeps the
//   cheap 256-bit lane moves and stays inside pulp's safe wrappers
//   (`#![forbid(unsafe_code)]`).

use std::arch::x86_64::__m512;

use pulp::{cast, f32x8, f32x16, x86::V4};

/// Rebuild one half of the 8×8×2 transpose from a (tt_x, tt_{x+4}) pair by
/// applying AVX2's `permute2f128` independently to each block's 256-bit half.
///
/// `IMM` is 0x20 (lo: low 128 of a, low 128 of b) or 0x31 (hi: high 128 of a,
/// high 128 of b). Result layout: for 0x20 → a0,b0,a2,b2; for 0x31 → a1,b1,a3,b3.
#[inline(always)]
fn recombine<const IMM: i32>(v4: V4, a: __m512, b: __m512) -> __m512 {
    let a_lo = v4.avx512f._mm512_castps512_ps256(a);
    let b_lo = v4.avx512f._mm512_castps512_ps256(b);
    let a_hi = v4.avx512dq._mm512_extractf32x8_ps::<1>(a);
    let b_hi = v4.avx512dq._mm512_extractf32x8_ps::<1>(b);
    let out_lo = v4.avx._mm256_permute2f128_ps::<IMM>(a_lo, b_lo);
    let out_hi = v4.avx._mm256_permute2f128_ps::<IMM>(a_hi, b_hi);
    v4.avx512dq
        ._mm512_insertf32x8::<1>(v4.avx512f._mm512_castps256_ps512(out_lo), out_hi)
}

#[inline(always)] // must fuse into the `vectorize` closure -> see `Coefficients::new`
fn transpose8x8x2(v4: V4, rows: [f32x16; 8]) -> [f32x16; 8] {
    let avx512f = v4.avx512f;
    let r: [__m512; 8] = rows.map(|row| cast!(row));

    let t0 = avx512f._mm512_unpacklo_ps(r[0], r[1]);
    let t1 = avx512f._mm512_unpackhi_ps(r[0], r[1]);
    let t2 = avx512f._mm512_unpacklo_ps(r[2], r[3]);
    let t3 = avx512f._mm512_unpackhi_ps(r[2], r[3]);
    let t4 = avx512f._mm512_unpacklo_ps(r[4], r[5]);
    let t5 = avx512f._mm512_unpackhi_ps(r[4], r[5]);
    let t6 = avx512f._mm512_unpacklo_ps(r[6], r[7]);
    let t7 = avx512f._mm512_unpackhi_ps(r[6], r[7]);

    let tt0 = avx512f._mm512_shuffle_ps::<0x44>(t0, t2);
    let tt1 = avx512f._mm512_shuffle_ps::<0xEE>(t0, t2);
    let tt2 = avx512f._mm512_shuffle_ps::<0x44>(t1, t3);
    let tt3 = avx512f._mm512_shuffle_ps::<0xEE>(t1, t3);
    let tt4 = avx512f._mm512_shuffle_ps::<0x44>(t4, t6);
    let tt5 = avx512f._mm512_shuffle_ps::<0xEE>(t4, t6);
    let tt6 = avx512f._mm512_shuffle_ps::<0x44>(t5, t7);
    let tt7 = avx512f._mm512_shuffle_ps::<0xEE>(t5, t7);

    [
        cast!(recombine::<0x20>(v4, tt0, tt4)),
        cast!(recombine::<0x20>(v4, tt1, tt5)),
        cast!(recombine::<0x20>(v4, tt2, tt6)),
        cast!(recombine::<0x20>(v4, tt3, tt7)),
        cast!(recombine::<0x31>(v4, tt0, tt4)),
        cast!(recombine::<0x31>(v4, tt1, tt5)),
        cast!(recombine::<0x31>(v4, tt2, tt6)),
        cast!(recombine::<0x31>(v4, tt3, tt7)),
    ]
}

// Mirrors `avx2::Coefficients`, just splat to 16 lanes so the same constant
// feeds both blocks at once.
const A: f32 = 3.535536e-1;
const B: f32 = 4.903927e-1;
const C: f32 = 4.619398e-1;
const D: f32 = 4.157349e-1;
const E: f32 = 2.777855e-1;
const F: f32 = 1.913422e-1;
const G: f32 = 9.754573e-2;

pub(crate) struct Coefficients {
    a: f32x16,
    na: f32x16,
    b: f32x16,
    nb: f32x16,
    c: f32x16,
    nc: f32x16,
    d: f32x16,
    e: f32x16,
    ne: f32x16,
    f: f32x16,
    nf: f32x16,
    g: f32x16,
    ng: f32x16,
}

impl Coefficients {
    #[inline(always)]
    pub(crate) fn new(v4: V4) -> Self {
        Self {
            a: v4.splat_f32x16(A),
            na: v4.splat_f32x16(-A),
            b: v4.splat_f32x16(B),
            nb: v4.splat_f32x16(-B),
            c: v4.splat_f32x16(C),
            nc: v4.splat_f32x16(-C),
            d: v4.splat_f32x16(D),
            e: v4.splat_f32x16(E),
            ne: v4.splat_f32x16(-E),
            f: v4.splat_f32x16(F),
            nf: v4.splat_f32x16(-F),
            g: v4.splat_f32x16(G),
            ng: v4.splat_f32x16(-G),
        }
    }
}

// Mechanical width-doubling of `avx2::row_pass` (same butterfly, f32x16
// instead of f32x8) -- this step is purely elementwise, so it never needs to
// know about the block-A/block-B split at all.
//
// Fusing via `mul_add_f32x16` was tried and reverted:
// bit-exactness broke (4096/4096 test blocks produced at least one differing
// f32 vs the plain mul/add version, single rounding vs double), and it was
// even ~2-4% *slower* in isolation despite ~9% fewer vector instructions.
// Root cause confirmed via `llvm-mca` (znver4/znver5 sched models): the
// fused version isn't an LLVM codegen defect -> AMD's own port-mapping data
// shows plain mul/add spreads across all 4 FP pipes (FP0-FP3, resource
// pressure 65-96% each) while `vfmadd231ps` collapses almost entirely onto
// one pipe (97% on FP1 alone), and register-dependency pressure rises
// 17%->60%. This kernel already had enough independent mul/add work to
// saturate 4 ports; fusing pairs into one op *removes* that port-spreading
// opportunity.
fn row_pass(v4: V4, coef: &Coefficients, input: [f32x16; 8]) -> [f32x16; 8] {
    let mul = |a, b| v4.mul_f32x16(a, b);
    let add = |a, b| v4.add_f32x16(a, b);
    let sub = |a, b| v4.sub_f32x16(a, b);

    let (in0, in2, in4, in6) = (input[0], input[2], input[4], input[6]);
    let (in1, in3, in5, in7) = (input[1], input[3], input[5], input[7]);

    let even0 =
        add(add(mul(in4, coef.a), mul(in6, coef.f)), add(mul(in0, coef.a), mul(in2, coef.c)));
    let even1 =
        add(add(mul(in4, coef.na), mul(in6, coef.nc)), add(mul(in0, coef.a), mul(in2, coef.f)));
    let even2 =
        add(add(mul(in4, coef.na), mul(in6, coef.c)), add(mul(in0, coef.a), mul(in2, coef.nf)));
    let even3 =
        add(add(mul(in4, coef.a), mul(in6, coef.nf)), add(mul(in0, coef.a), mul(in2, coef.nc)));

    let odd0 =
        add(add(mul(in5, coef.e), mul(in7, coef.g)), add(mul(in1, coef.b), mul(in3, coef.d)));
    let odd1 =
        add(add(mul(in5, coef.nb), mul(in7, coef.ne)), add(mul(in1, coef.d), mul(in3, coef.ng)));
    let odd2 =
        add(add(mul(in5, coef.g), mul(in7, coef.d)), add(mul(in1, coef.e), mul(in3, coef.nb)));
    let odd3 =
        add(add(mul(in5, coef.d), mul(in7, coef.nb)), add(mul(in1, coef.g), mul(in3, coef.ne)));

    [
        add(even0, odd0),
        add(even1, odd1),
        add(even2, odd2),
        add(even3, odd3),
        sub(even3, odd3),
        sub(even2, odd2),
        sub(even1, odd1),
        sub(even0, odd0),
    ]
}

// Mechanical width-doubling of `avx2::column_pass`.
#[inline(always)]
fn column_pass(v4: V4, coef: &Coefficients, input: [f32x16; 8]) -> [f32x16; 8] {
    let mul = |a, b| v4.mul_f32x16(a, b);
    let add = |a, b| v4.add_f32x16(a, b);
    let sub = |a, b| v4.sub_f32x16(a, b);

    let (in0, in1, in2, in3, in4, in5, in6, in7) =
        (input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);

    let beta0 =
        add(add(mul(coef.g, in7), mul(coef.e, in5)), add(mul(coef.d, in3), mul(coef.b, in1)));
    let beta1 =
        sub(sub(mul(coef.d, in1), add(mul(coef.b, in5), mul(coef.g, in3))), mul(coef.e, in7));
    let beta2 =
        add(mul(coef.d, in7), add(mul(coef.g, in5), sub(mul(coef.e, in1), mul(coef.b, in3))));
    let beta3 =
        sub(add(mul(coef.d, in5), mul(coef.g, in1)), add(mul(coef.b, in7), mul(coef.e, in3)));

    let theta0 = add(mul(coef.a, in4), mul(coef.a, in0));
    let theta3 = sub(mul(coef.a, in0), mul(coef.a, in4));

    let theta1 = add(mul(coef.f, in6), mul(coef.c, in2));
    let gamma0 = add(theta1, theta0);
    let gamma3 = sub(theta0, theta1);

    let theta2 = sub(mul(coef.f, in2), mul(coef.c, in6));
    let gamma1 = add(theta3, theta2);
    let gamma2 = sub(theta3, theta2);

    [
        add(gamma0, beta0),
        add(gamma1, beta1),
        add(gamma2, beta2),
        add(gamma3, beta3),
        sub(gamma3, beta3),
        sub(gamma2, beta2),
        sub(gamma1, beta1),
        sub(gamma0, beta0),
    ]
}

/// Pack one row of block A (lanes 0-7) and block B (lanes 8-15).
/// Builds each half as contiguous `f32x8` (LLVM turns that into `vmovups`)
/// then stitches with `insertf32x8` — avoids the scalar-16-lane `f32x16(...)`
/// form that spilled the whole kernel through the stack.
#[inline(always)]
fn load_pair(v4: V4, a: &[f32; 64], b: &[f32; 64], row: usize) -> f32x16 {
    let ba = row * 8;
    let va = f32x8(
        a[ba],
        a[ba + 1],
        a[ba + 2],
        a[ba + 3],
        a[ba + 4],
        a[ba + 5],
        a[ba + 6],
        a[ba + 7],
    );
    let vb = f32x8(
        b[ba],
        b[ba + 1],
        b[ba + 2],
        b[ba + 3],
        b[ba + 4],
        b[ba + 5],
        b[ba + 6],
        b[ba + 7],
    );
    let lo = v4.avx512f._mm512_castps256_ps512(cast!(va));
    cast!(v4.avx512dq._mm512_insertf32x8::<1>(lo, cast!(vb)))
}

#[inline(always)]
fn store_pair(v4: V4, a: &mut [f32; 64], b: &mut [f32; 64], row: usize, value: f32x16) {
    let ba = row * 8;
    let z: __m512 = cast!(value);
    let fa: f32x8 = cast!(v4.avx512f._mm512_castps512_ps256(z));
    let fb: f32x8 = cast!(v4.avx512dq._mm512_extractf32x8_ps::<1>(z));
    a[ba] = fa.0;
    a[ba + 1] = fa.1;
    a[ba + 2] = fa.2;
    a[ba + 3] = fa.3;
    a[ba + 4] = fa.4;
    a[ba + 5] = fa.5;
    a[ba + 6] = fa.6;
    a[ba + 7] = fa.7;
    b[ba] = fb.0;
    b[ba + 1] = fb.1;
    b[ba + 2] = fb.2;
    b[ba + 3] = fb.3;
    b[ba + 4] = fb.4;
    b[ba + 5] = fb.5;
    b[ba + 6] = fb.6;
    b[ba + 7] = fb.7;
}

/// One 8x8 inverse DCT for each of two blocks at once. Must be called from
/// inside a `V4::vectorize` trampoline. Same in-register shape as
/// `avx2::inverse_one`: load 8 rows -> transpose -> row_pass -> transpose
/// back -> column_pass -> store, just processing block A and block B side by
/// side in every step.
#[inline(always)]
pub(crate) fn inverse_pair(
    v4: V4,
    coef: &Coefficients,
    a: &mut [f32; 64],
    b: &mut [f32; 64],
) {
    let rows: [f32x16; 8] = std::array::from_fn(|row| load_pair(v4, a, b, row));
    let columns = transpose8x8x2(v4, rows);
    let row_pass_out = row_pass(v4, coef, columns);
    let intermediate_rows = transpose8x8x2(v4, row_pass_out);
    let columns_out = column_pass(v4, coef, intermediate_rows);
    for (row, result) in columns_out.iter().enumerate() {
        store_pair(v4, a, b, row, *result);
    }
}

#[cfg(any(feature = "avx512-tests", feature = "simd-benches"))]
pulp::v4_fn! {
    pub fn dct_inverse_8x8_pair(v4: V4, a: &mut [f32; 64], b: &mut [f32; 64]) {
        let coef = Coefficients::new(v4);
        inverse_pair(v4, &coef, a, b);
    }
}

pulp::v4_fn! {
    /// Batched inverse DCT: processes blocks 2 at a time through the AVX-512
    /// kernel; a trailing odd block (if `blocks` has an odd length) falls back to
    /// the scalar autovectorized kernel -- this is a Stage-1 prototype for
    /// benchmarking/correctness only, not yet wired into the real dispatch chain
    /// (see `discrete_cosine_transform/x86/mod.rs`), so the odd-block case is
    /// deliberately simple rather than reaching for another SIMD tier.
    pub fn dct_inverse_8x8_batch<'a>(v4: V4, blocks: impl Iterator<Item = &'a mut [f32; 64]>) {
        let coef = Coefficients::new(v4);
        let mut iter = blocks;
        loop {
            let Some(first) = iter.next() else { break };
            match iter.next() {
                Some(second) => inverse_pair(v4, &coef, first, second),
                None => {
                    super::super::dct_inverse_8x8_autovectorized(first);
                    break;
                }
            }
        }
    }
}

/// Two independent `inverse_pair` chains (4 spatial blocks, 2 zmm registers),
/// stage-interleaved by hand: both chains' transpose runs before either
/// chain's row_pass, etc. Zen5 has two full-width 512-bit execution ports; a
/// single `inverse_pair` chain is a strict transpose -> row_pass -> transpose
/// -> column_pass dependency chain with nothing else in flight to fill the
/// second port while each stage's latency drains. Placing a second,
/// data-independent chain's same-stage instructions immediately adjacent in
/// the source gives the scheduler two ready, port-fillable instruction
/// streams instead of one, a source-level microbenchmark-only prototype,
/// not wired into any dispatch path. See `dct_inverse_bench_avx512_batch_quad`
/// in `benches/dct.rs` for the A/B against back-to-back `inverse_pair` calls
/// (which is what today's `dct_inverse_8x8_batch` loop already produces, one
/// pair per iteration, whatever ILP LLVM's own loop unrolling finds on its own).
#[inline(always)]
pub(crate) fn inverse_quad(
    v4: V4,
    coef: &Coefficients,
    a0: &mut [f32; 64],
    b0: &mut [f32; 64],
    a1: &mut [f32; 64],
    b1: &mut [f32; 64],
) {
    let rows0: [f32x16; 8] = std::array::from_fn(|row| load_pair(v4, a0, b0, row));
    let rows1: [f32x16; 8] = std::array::from_fn(|row| load_pair(v4, a1, b1, row));

    let columns0 = transpose8x8x2(v4, rows0);
    let columns1 = transpose8x8x2(v4, rows1);

    let row_pass_out0 = row_pass(v4, coef, columns0);
    let row_pass_out1 = row_pass(v4, coef, columns1);

    let intermediate_rows0 = transpose8x8x2(v4, row_pass_out0);
    let intermediate_rows1 = transpose8x8x2(v4, row_pass_out1);

    let columns_out0 = column_pass(v4, coef, intermediate_rows0);
    let columns_out1 = column_pass(v4, coef, intermediate_rows1);

    for (row, result) in columns_out0.iter().enumerate() {
        store_pair(v4, a0, b0, row, *result);
    }
    for (row, result) in columns_out1.iter().enumerate() {
        store_pair(v4, a1, b1, row, *result);
    }
}

#[cfg(any(feature = "avx512-tests", feature = "simd-benches"))]
pulp::v4_fn! {
    pub fn dct_inverse_8x8_quad(
        v4: V4,
        a0: &mut [f32; 64],
        b0: &mut [f32; 64],
        a1: &mut [f32; 64],
        b1: &mut [f32; 64],
    ) {
        let coef = Coefficients::new(v4);
        inverse_quad(v4, &coef, a0, b0, a1, b1);
    }
}

pulp::v4_fn! {
    /// Same batching contract as `dct_inverse_8x8_batch`, but processes 4
    /// blocks (2 pairs) per iteration through the hand-interleaved
    /// `inverse_quad` kernel; a short trailing remainder (1-3 blocks) falls
    /// back to `inverse_pair`/scalar. Microbenchmark-only prototype (see
    /// `inverse_quad`'s doc comment), not wired into the real dispatch chain.
    pub fn dct_inverse_8x8_batch_quad<'a>(v4: V4, blocks: impl Iterator<Item = &'a mut [f32; 64]>) {
        let coef = Coefficients::new(v4);
        let mut iter = blocks;
        loop {
            let Some(a0) = iter.next() else { break };
            let Some(b0) = iter.next() else {
                super::super::dct_inverse_8x8_autovectorized(a0);
                break;
            };
            let Some(a1) = iter.next() else {
                inverse_pair(v4, &coef, a0, b0);
                break;
            };
            let Some(b1) = iter.next() else {
                inverse_pair(v4, &coef, a0, b0);
                super::super::dct_inverse_8x8_autovectorized(a1);
                break;
            };
            inverse_quad(v4, &coef, a0, b0, a1, b1);
        }
    }
}

/// One RGB spatial pair as held by the fused AVX-512 decode step: two
/// blocks × three components. Microbench / test fixture type only.
pub type RgbPairBlocks = ([[f32; 64]; 3], [[f32; 64]; 3]);

// Baseline shape of today's `decode_pair_dct_csc` DCT loop: three sequential
// `inverse_pair` calls (R, then G, then B) on one spatial pair. No zigzag /
// CSC / write, pure DCT middle step, RGB-shaped so the component-quad A/B
// below measures the dual-port idea *without* growing the spatial working set.
#[cfg(any(feature = "avx512-tests", feature = "simd-benches"))]
pulp::v4_fn! {
    pub fn dct_inverse_rgb_pair_components_seq(v4: V4, pairs: &mut [RgbPairBlocks]) {
        let coef = Coefficients::new(v4);
        for (a, b) in pairs.iter_mut() {
            for c in 0..3 {
                inverse_pair(v4, &coef, &mut a[c], &mut b[c]);
            }
        }
    }
}

// Dual-port candidate for the same RGB pair: hand-interleave R and G through
// `inverse_quad` (two independent 512-bit chains, same spatial pair, buffers
// already L1-resident in the fused path), then B alone via `inverse_pair`.
// Same total arithmetic as `dct_inverse_rgb_pair_components_seq`; only the
// issue order changes.
#[cfg(any(feature = "avx512-tests", feature = "simd-benches"))]
pulp::v4_fn! {
    pub fn dct_inverse_rgb_pair_components_quad(v4: V4, pairs: &mut [RgbPairBlocks]) {
        let coef = Coefficients::new(v4);
        for (a, b) in pairs.iter_mut() {
            // split_at_mut: disjoint R/G mut refs (indexing alone won't borrow-check).
            let (a_r, a_rest) = a.split_at_mut(1);
            let (a_g, a_b) = a_rest.split_at_mut(1);
            let (b_r, b_rest) = b.split_at_mut(1);
            let (b_g, b_b) = b_rest.split_at_mut(1);
            inverse_quad(v4, &coef, &mut a_r[0], &mut b_r[0], &mut a_g[0], &mut b_g[0]);
            inverse_pair(v4, &coef, &mut a_b[0], &mut b_b[0]);
        }
    }
}

// TEMPORARY (2026-07-28 pulp VBMI2 smoke test, not for shipping): exercises
// the newly-added `pulp::x86::V4Vbmi2` capability type and its
// `mask_compress_u16x32`/`mask_expand_u16x32` wrappers (vendored fork,
// `V4-vectorize-in-target-feature` branch) against a scalar reference, to
// confirm the new fork infrastructure actually executes correctly on real
// hardware and not just compiles.
#[cfg(all(test, feature = "avx512-tests"))]
mod vbmi2_probe {
    use pulp::{b32, cast, x86::V4Vbmi2};

    pulp::v4_vbmi2_fn! {
        fn compress(v4: V4Vbmi2, mask: u32, a: [u16; 32]) -> [u16; 32] {
            cast!(v4.mask_compress_u16x32(b32(mask), cast!(a)))
        }
    }

    pulp::v4_vbmi2_fn! {
        fn expand(v4: V4Vbmi2, mask: u32, a: [u16; 32]) -> [u16; 32] {
            cast!(v4.mask_expand_u16x32(b32(mask), cast!(a)))
        }
    }

    fn scalar_compress(mask: u32, a: [u16; 32]) -> [u16; 32] {
        let mut out = [0u16; 32];
        let mut dst = 0;
        for i in 0..32 {
            if (mask >> i) & 1 == 1 {
                out[dst] = a[i];
                dst += 1;
            }
        }
        out
    }

    fn scalar_expand(mask: u32, a: [u16; 32]) -> [u16; 32] {
        let mut out = [0u16; 32];
        let mut src = 0;
        for i in 0..32 {
            if (mask >> i) & 1 == 1 {
                out[i] = a[src];
                src += 1;
            }
        }
        out
    }

    #[test]
    fn vbmi2_compress_expand_match_scalar_reference() {
        let Some(v4) = V4Vbmi2::try_new() else {
            // Skylake-X/Cascade Lake-class hosts have V4 but not VBMI2
            // this is exactly the case V4Vbmi2 exists to guard against.
            return;
        };

        let mut random = rand::rngs::StdRng::seed_from_u64(0x7645_1e3e);

        for _ in 0..4096 {
            let mask: u32 = random.random();
            let mut a = [0u16; 32];
            for slot in a.iter_mut() {
                *slot = random.random();
            }

            assert_eq!(compress(v4, mask, a), scalar_compress(mask, a));
            assert_eq!(expand(v4, mask, a), scalar_expand(mask, a));
        }
    }

    use rand::{RngExt, SeedableRng};
}
