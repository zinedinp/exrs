// AVX2 V3 tier: OpenEXR's "dctInverse8x8_avx_0". Each pass runs all 8
// rows/columns of the block in parallel, one 8-wide register per position.
//
// `dct_inverse_8x8_batch` runs the kernel through `V3::vectorize` rather
// than calling it as an ordinary function.

use std::arch::x86_64::__m256;

use pulp::{cast, f32x8, x86::V3};

use super::forward_basis;

// Transposes 8 contiguous row-vectors into 8 column-vectors entirely in
// registers (unpacklo/unpackhi + shuffle + permute2f128), replacing a scalar
// strided-gather construction (8 independent `data[k], data[8+k], ...`
// scalar reads per lane). Measured standalone (not in this crate) at ~32%
// faster to build and ~25% faster once fused with the row_pass math that
// consumes it -> LLVM compiles the scalar-gather construction into a chain of
// `vinsertps`-per-scalar instructions rather than a handful of wide shuffles.
#[inline(always)] // must fuse into the `vectorize` closure -> see `Coefficients::new`
fn transpose8x8(v3: V3, rows: [f32x8; 8]) -> [f32x8; 8] {
    let avx = v3.avx;
    let r: [__m256; 8] = rows.map(|row| cast!(row));

    let t0 = avx._mm256_unpacklo_ps(r[0], r[1]);
    let t1 = avx._mm256_unpackhi_ps(r[0], r[1]);
    let t2 = avx._mm256_unpacklo_ps(r[2], r[3]);
    let t3 = avx._mm256_unpackhi_ps(r[2], r[3]);
    let t4 = avx._mm256_unpacklo_ps(r[4], r[5]);
    let t5 = avx._mm256_unpackhi_ps(r[4], r[5]);
    let t6 = avx._mm256_unpacklo_ps(r[6], r[7]);
    let t7 = avx._mm256_unpackhi_ps(r[6], r[7]);

    let tt0 = avx._mm256_shuffle_ps::<0x44>(t0, t2);
    let tt1 = avx._mm256_shuffle_ps::<0xEE>(t0, t2);
    let tt2 = avx._mm256_shuffle_ps::<0x44>(t1, t3);
    let tt3 = avx._mm256_shuffle_ps::<0xEE>(t1, t3);
    let tt4 = avx._mm256_shuffle_ps::<0x44>(t4, t6);
    let tt5 = avx._mm256_shuffle_ps::<0xEE>(t4, t6);
    let tt6 = avx._mm256_shuffle_ps::<0x44>(t5, t7);
    let tt7 = avx._mm256_shuffle_ps::<0xEE>(t5, t7);

    [
        cast!(avx._mm256_permute2f128_ps::<0x20>(tt0, tt4)),
        cast!(avx._mm256_permute2f128_ps::<0x20>(tt1, tt5)),
        cast!(avx._mm256_permute2f128_ps::<0x20>(tt2, tt6)),
        cast!(avx._mm256_permute2f128_ps::<0x20>(tt3, tt7)),
        cast!(avx._mm256_permute2f128_ps::<0x31>(tt0, tt4)),
        cast!(avx._mm256_permute2f128_ps::<0x31>(tt1, tt5)),
        cast!(avx._mm256_permute2f128_ps::<0x31>(tt2, tt6)),
        cast!(avx._mm256_permute2f128_ps::<0x31>(tt3, tt7)),
    ]
}

// OpenEXRs hardcoded AVX basis constants ("sAvxCoef").
const A: f32 = 3.535536e-1;
const B: f32 = 4.903927e-1;
const C: f32 = 4.619398e-1;
const D: f32 = 4.157349e-1;
const E: f32 = 2.777855e-1;
const F: f32 = 1.913422e-1;
const G: f32 = 9.754573e-2;

// Public to the crate so the fused lossy-DCT decode path can build the
// constants once per `vectorize` trampoline and reuse them across every
// block's in-register iDCT (see `inverse_one`).
pub(crate) struct Coefficients {
    a: f32x8,
    na: f32x8,
    b: f32x8,
    nb: f32x8,
    c: f32x8,
    nc: f32x8,
    d: f32x8,
    // no "nd": the AVX never multiplies by -D
    e: f32x8,
    ne: f32x8,
    f: f32x8,
    nf: f32x8,
    g: f32x8,
    ng: f32x8,
}

impl Coefficients {
    // This, `row_pass`, and `column_pass` must inline into the
    // `vectorize` closure below for their ops to fuse into avx2
    // instructions; LLVM inlining heuristics aren't reliable
    // enough to guarantee that on their own
    #[inline(always)]
    pub(crate) fn new(v3: V3) -> Self {
        // Negated splats are exact (sign flip), so "x * na == -(x * a)".
        Self {
            a: v3.splat_f32x8(A),
            na: v3.splat_f32x8(-A),
            b: v3.splat_f32x8(B),
            nb: v3.splat_f32x8(-B),
            c: v3.splat_f32x8(C),
            nc: v3.splat_f32x8(-C),
            d: v3.splat_f32x8(D),
            e: v3.splat_f32x8(E),
            ne: v3.splat_f32x8(-E),
            f: v3.splat_f32x8(F),
            nf: v3.splat_f32x8(-F),
            g: v3.splat_f32x8(G),
            ng: v3.splat_f32x8(-G),
        }
    }
}

// OpenEXRs "IDCT_AVX_MMULT_ROWS" + "EO_TO_ROW_HALVES"
#[inline(always)] // must fuse into the `vectorize` closure --> see `Coefficients::new`
fn row_pass(v3: V3, coef: &Coefficients, input: [f32x8; 8]) -> [f32x8; 8] {
    let mul = |a, b| v3.mul_f32x8(a, b);
    let add = |a, b| v3.add_f32x8(a, b);
    let sub = |a, b| v3.sub_f32x8(a, b);

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

// The column transform from the back half of "dctInverse8x8_avx_0".
#[inline(always)] // must fuse into the `vectorize` closure --> see `Coefficients::new`
fn column_pass(v3: V3, coef: &Coefficients, input: [f32x8; 8]) -> [f32x8; 8] {
    let mul = |a, b| v3.mul_f32x8(a, b);
    let add = |a, b| v3.add_f32x8(a, b);
    let sub = |a, b| v3.sub_f32x8(a, b);

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

#[cfg(any(feature = "avx2-tests", feature = "simd-benches"))]
pub fn dct_inverse_8x8(v3: V3, data: &mut [f32; 64]) {
    dct_inverse_8x8_batch(v3, std::iter::once(data));
}

// `V3::vectorize` runs a `FnOnce()` closure inside pulps own
// `#[target_feature(enable = "avx2,fma")]` trampoline; passing the
// kernel as a closure, rather than calling it as an ordinary function,
// is what lets that closures body inline and fuse into avx2
// instructions.
//
// One 8x8 inverse DCT. Must be called from inside a `V3::vectorize`
// trampoline (or another `#[target_feature(enable = "avx2,fma")]` body)
// so the ops lower to AVX2; the fused decode path relies on that.
//
// Full iDCT stays in registers: load 8 rows -> transpose to the column-major
// shape row_pass wants -> row_pass -> transpose back to rows for column_pass
// -> column_pass -> store. Scatter-storing the row-pass result and reloading
// it was pure intermediate L1 traffic for a 256-byte block that already
// fits in registers.
#[inline(always)]
pub(crate) fn inverse_one(v3: V3, coef: &Coefficients, data: &mut [f32; 64]) {
    let rows: [f32x8; 8] = std::array::from_fn(|row| {
        let b = row * 8;
        f32x8(
            data[b],
            data[b + 1],
            data[b + 2],
            data[b + 3],
            data[b + 4],
            data[b + 5],
            data[b + 6],
            data[b + 7],
        )
    });
    let columns = transpose8x8(v3, rows);
    let row_pass_out = row_pass(v3, &coef, columns);
    // row_pass_out[col].lane[row] = intermediate[row][col]; the 8x8
    // transpose is an involution, so one more pass yields
    // intermediate_rows[row].lane[col] for column_pass.
    let intermediate_rows = transpose8x8(v3, row_pass_out);
    let columns_out = column_pass(v3, &coef, intermediate_rows);
    for (row, result) in columns_out.iter().enumerate() {
        let b = row * 8;
        data[b] = result.0;
        data[b + 1] = result.1;
        data[b + 2] = result.2;
        data[b + 3] = result.3;
        data[b + 4] = result.4;
        data[b + 5] = result.5;
        data[b + 6] = result.6;
        data[b + 7] = result.7;
    }
}

// `vectorize` fixed overhead per call
pub fn dct_inverse_8x8_batch<'a>(v3: V3, blocks: impl Iterator<Item = &'a mut [f32; 64]>) {
    v3.vectorize(move || {
        let coef = Coefficients::new(v3);
        for data in blocks {
            inverse_one(v3, &coef, data);
        }
    });
}

struct ForwardCoefficients {
    terms: [f32x8; 8],
}

impl ForwardCoefficients {
    #[inline(always)]
    fn new(_v3: V3) -> Self {
        let basis = forward_basis();
        Self {
            terms: std::array::from_fn(|input| {
                let row = basis[input];
                f32x8(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
            }),
        }
    }
}

#[inline(always)]
fn forward_pass(v3: V3, coef: &ForwardCoefficients, input: [f32; 8]) -> f32x8 {
    let mul = |a, b| v3.mul_f32x8(a, b);
    let add = |a, b| v3.add_f32x8(a, b);
    let splat = |value: f32| v3.splat_f32x8(value);

    let mut out = v3.splat_f32x8(0.0);
    for index in 0..8 {
        out = add(out, mul(splat(input[index]), coef.terms[index]));
    }
    out
}

// TODO just #[test]
#[cfg(any(feature = "avx2-tests", feature = "simd-benches"))]
pub fn dct_forward_8x8(v3: V3, data: &mut [f32; 64]) {
    dct_forward_8x8_batch(v3, std::iter::once(data));
}

pub fn dct_forward_8x8_batch<'a>(v3: V3, blocks: impl Iterator<Item = &'a mut [f32; 64]>) {
    v3.vectorize(move || {
        let coef = ForwardCoefficients::new(v3);
        let basis = forward_basis();

        for data in blocks {
            // Row pass: each row's own 8 values are already contiguous, so
            // this needs no gather.
            for row in 0..8 {
                let base = row * 8;
                let input = [
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                    data[base + 4],
                    data[base + 5],
                    data[base + 6],
                    data[base + 7],
                ];
                let out = forward_pass(v3, &coef, input);
                data[base] = out.0;
                data[base + 1] = out.1;
                data[base + 2] = out.2;
                data[base + 3] = out.3;
                data[base + 4] = out.4;
                data[base + 5] = out.5;
                data[base + 6] = out.6;
                data[base + 7] = out.7;
            }

            // Column pass: batched across all 8 columns via SIMD lanes
            // instead of gathering one column at a time with a stride-8
            // read. Each row is loaded contiguously once and fanned into 8
            // per-frequency accumulators (one lane per column), which are
            // then stored back contiguously per output row.
            let mut outputs = [v3.splat_f32x8(0.0); 8];
            for row in 0..8 {
                let base = row * 8;
                let row_vec = f32x8(
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                    data[base + 4],
                    data[base + 5],
                    data[base + 6],
                    data[base + 7],
                );
                for v in 0..8 {
                    let coefficient = v3.splat_f32x8(basis[row][v]);
                    outputs[v] = v3.add_f32x8(outputs[v], v3.mul_f32x8(coefficient, row_vec));
                }
            }

            for (v, out) in outputs.iter().enumerate() {
                let base = v * 8;
                data[base] = out.0;
                data[base + 1] = out.1;
                data[base + 2] = out.2;
                data[base + 3] = out.3;
                data[base + 4] = out.4;
                data[base + 5] = out.5;
                data[base + 6] = out.6;
                data[base + 7] = out.7;
            }
        }
    });
}
