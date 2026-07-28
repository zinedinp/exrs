// AVX-512 (V4) tier: CSC is a fixed per-element linear combination of three
// same-length arrays with no cross-lane shuffles, so unlike the DCT's
// transpose this ports to 2-blocks-per-register by pure mechanical widening.
// load block A's 8-wide chunk into lanes 0-7 and block B's into lanes 8-15,
// run the same elementwise math, store back. No permute/shuffle needed at all.
// (DCT needs insert/extract load packing; pure elementwise CSC is fine with
// the scalar-lane `f32x16`
use pulp::{f32x16, x86::V4};

#[inline(always)]
fn load_pair(a: &[f32; 64], b: &[f32; 64], base: usize) -> f32x16 {
    f32x16(
        a[base],
        a[base + 1],
        a[base + 2],
        a[base + 3],
        a[base + 4],
        a[base + 5],
        a[base + 6],
        a[base + 7],
        b[base],
        b[base + 1],
        b[base + 2],
        b[base + 3],
        b[base + 4],
        b[base + 5],
        b[base + 6],
        b[base + 7],
    )
}

#[inline(always)]
fn store_pair(a: &mut [f32; 64], b: &mut [f32; 64], base: usize, value: f32x16) {
    a[base] = value.0;
    a[base + 1] = value.1;
    a[base + 2] = value.2;
    a[base + 3] = value.3;
    a[base + 4] = value.4;
    a[base + 5] = value.5;
    a[base + 6] = value.6;
    a[base + 7] = value.7;
    b[base] = value.8;
    b[base + 1] = value.9;
    b[base + 2] = value.10;
    b[base + 3] = value.11;
    b[base + 4] = value.12;
    b[base + 5] = value.13;
    b[base + 6] = value.14;
    b[base + 7] = value.15;
}

#[cfg(any(feature = "avx512-tests", feature = "simd-benches"))]
pub fn csc709_forward_8x8_pair(v4: V4, a: &mut [[f32; 64]; 3], b: &mut [[f32; 64]; 3]) {
    v4.vectorize(move || {
        forward_pair(v4, a, b);
    });
}

#[inline(always)]
pub(crate) fn forward_pair(v4: V4, a: &mut [[f32; 64]; 3], b: &mut [[f32; 64]; 3]) {
    let c_r = v4.splat_f32x16(0.2126);
    let c_g = v4.splat_f32x16(0.7152);
    let c_b = v4.splat_f32x16(0.0722);
    let inv_by = v4.splat_f32x16(1.0 / 1.8556);
    let inv_ry = v4.splat_f32x16(1.0 / 1.5747);

    let mul = |x, y| v4.mul_f32x16(x, y);
    let add = |x, y| v4.add_f32x16(x, y);
    let sub = |x, y| v4.sub_f32x16(x, y);

    let [ar, ag, ab] = a;
    let [br, bg, bb] = b;
    for chunk in 0..8 {
        let base = chunk * 8;
        let rv = load_pair(ar, br, base);
        let gv = load_pair(ag, bg, base);
        let bv = load_pair(ab, bb, base);

        let y = add(add(mul(rv, c_r), mul(gv, c_g)), mul(bv, c_b));
        let by = mul(sub(bv, y), inv_by);
        let ry = mul(sub(rv, y), inv_ry);

        store_pair(ar, br, base, y);
        store_pair(ag, bg, base, by);
        store_pair(ab, bb, base, ry);
    }
}

#[cfg(any(feature = "avx512-tests", feature = "simd-benches"))]
pub fn csc709_inverse_8x8_pair(v4: V4, a: &mut [[f32; 64]; 3], b: &mut [[f32; 64]; 3]) {
    v4.vectorize(move || {
        inverse_pair(v4, a, b);
    });
}

/// One 8x8 inverse CSC for each of two blocks at once. Must run inside a
/// `V4::vectorize` trampoline.
#[inline(always)]
pub(crate) fn inverse_pair(v4: V4, a: &mut [[f32; 64]; 3], b: &mut [[f32; 64]; 3]) {
    let c_ry = v4.splat_f32x16(1.5747);
    let c_by_g = v4.splat_f32x16(0.1873);
    let c_ry_g = v4.splat_f32x16(0.4682);
    let c_by = v4.splat_f32x16(1.8556);

    let mul = |x, y| v4.mul_f32x16(x, y);
    let add = |x, y| v4.add_f32x16(x, y);
    let sub = |x, y| v4.sub_f32x16(x, y);

    let [a0, a1, a2] = a;
    let [b0, b1, b2] = b;
    for chunk in 0..8 {
        let base = chunk * 8;
        let y = load_pair(a0, b0, base);
        let by = load_pair(a1, b1, base);
        let ry = load_pair(a2, b2, base);

        let r = add(y, mul(ry, c_ry));
        let g = sub(sub(y, mul(by, c_by_g)), mul(ry, c_ry_g));
        let b_out = add(y, mul(by, c_by));

        store_pair(a0, b0, base, r);
        store_pair(a1, b1, base, g);
        store_pair(a2, b2, base, b_out);
    }
}

pub fn csc709_inverse_8x8_batch<'a>(
    v4: V4,
    blocks: impl Iterator<Item = &'a mut [[f32; 64]; 3]>,
) {
    v4.vectorize(move || {
        let mut iter = blocks;
        loop {
            let Some(first) = iter.next() else { break };
            match iter.next() {
                Some(second) => inverse_pair(v4, first, second),
                None => {
                    super::super::csc709_inverse_8x8_autovectorized(first);
                    break;
                }
            }
        }
    });
}

pub fn csc709_forward_8x8_batch<'a>(
    v4: V4,
    blocks: impl Iterator<Item = &'a mut [[f32; 64]; 3]>,
) {
    v4.vectorize(move || {
        let mut iter = blocks;
        loop {
            let Some(first) = iter.next() else { break };
            match iter.next() {
                Some(second) => forward_pair(v4, first, second),
                None => {
                    super::super::csc709_forward_8x8_autovectorized(first);
                    break;
                }
            }
        }
    });
}
