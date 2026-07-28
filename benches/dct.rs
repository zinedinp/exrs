#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[macro_use]
extern crate bencher;

extern crate exr;

use bencher::Bencher;
use exr::compression::dwa::discrete_cosine_transform::{x86::*, *};
use pulp::x86::{V1, V3, V4};

fn dct_forward_bench_autovectorized(bench: &mut Bencher) {
    let mut blocks = bench_blocks();

    bench.iter(|| {
        for block in blocks.iter_mut() {
            dct_forward_8x8_autovectorized(block);
        }

        bencher::black_box(&mut blocks);
    })
}

fn dct_forward_bench_sse2(bench: &mut Bencher) {
    let mut blocks = bench_blocks();
    let v1 = expect_sse2();

    bench.iter(|| {
        for block in blocks.iter_mut() {
            sse2::dct_forward_8x8(v1, block);
        }

        bencher::black_box(&mut blocks);
    })
}

fn dct_forward_bench_avx2(bench: &mut Bencher) {
    let mut blocks = bench_blocks();
    let v3 = expect_avx2();

    bench.iter(|| {
        for block in blocks.iter_mut() {
            avx2::dct_forward_8x8(v3, block);
        }

        bencher::black_box(&mut blocks);
    })
}

fn dct_forward_bench_avx2_batch(bench: &mut Bencher) {
    let mut blocks = bench_blocks();
    let v3 = expect_avx2();

    bench.iter(|| {
        avx2::dct_forward_8x8_batch(v3, blocks.iter_mut());

        bencher::black_box(&mut blocks);
    })
}

fn dct_inverse_bench_autovectorized(bench: &mut Bencher) {
    let mut blocks = bench_blocks();

    bench.iter(|| {
        for block in blocks.iter_mut() {
            dct_inverse_8x8_autovectorized(block);
        }

        bencher::black_box(&mut blocks);
    })
}

fn dct_inverse_bench_sse2(bench: &mut Bencher) {
    let mut blocks = bench_blocks();
    let v1 = expect_sse2();

    bench.iter(|| {
        for block in blocks.iter_mut() {
            sse2::dct_inverse_8x8(v1, block);
        }

        bencher::black_box(&mut blocks);
    })
}

fn dct_inverse_bench_avx2(bench: &mut Bencher) {
    let mut blocks = bench_blocks();
    let v3 = expect_avx2();

    bench.iter(|| {
        for block in blocks.iter_mut() {
            avx2::dct_inverse_8x8(v3, block);
        }

        bencher::black_box(&mut blocks);
    })
}

fn dct_inverse_bench_avx2_batch(bench: &mut Bencher) {
    let mut blocks = bench_blocks();
    let v3 = expect_avx2();

    bench.iter(|| {
        avx2::dct_inverse_8x8_batch(v3, blocks.iter_mut());

        bencher::black_box(&mut blocks);
    })
}

fn dct_inverse_bench_avx512_batch(bench: &mut Bencher) {
    let mut blocks = bench_blocks();
    let v4 = expect_avx512();

    bench.iter(|| {
        avx512::dct_inverse_8x8_batch(v4, blocks.iter_mut());

        bencher::black_box(&mut blocks);
    })
}

/// A/B against `dct_inverse_bench_avx512_batch`: same 1-pair-per-iteration
/// loop, but relies solely on LLVM's own loop unrolling for any ILP across
/// iterations -> see `avx512::inverse_quad`'s doc comment for what "quad"
/// changes.
fn dct_inverse_bench_avx512_batch_quad(bench: &mut Bencher) {
    let mut blocks = bench_blocks();
    let v4 = expect_avx512();

    bench.iter(|| {
        avx512::dct_inverse_8x8_batch_quad(v4, blocks.iter_mut());

        bencher::black_box(&mut blocks);
    })
}

/// RGB-shaped baseline: 3 sequential `inverse_pair`s per spatial pair
/// (mirrors `decode_pair_dct_csc`'s DCT loop). A/B partner is
/// `dct_inverse_bench_avx512_rgb_comp_quad`.
fn dct_inverse_bench_avx512_rgb_comp_seq(bench: &mut Bencher) {
    let mut pairs = bench_rgb_pairs();
    let v4 = expect_avx512();

    bench.iter(|| {
        avx512::dct_inverse_rgb_pair_components_seq(v4, &mut pairs);
        bencher::black_box(&mut pairs);
    })
}

/// RGB-shaped dual-port candidate: `inverse_quad(R,G)` + `inverse_pair(B)`.
/// Same buffers / arithmetic as `_rgb_comp_seq`; only issue order changes.
/// No extra spatial working set vs the fused pair path.
fn dct_inverse_bench_avx512_rgb_comp_quad(bench: &mut Bencher) {
    let mut pairs = bench_rgb_pairs();
    let v4 = expect_avx512();

    bench.iter(|| {
        avx512::dct_inverse_rgb_pair_components_quad(v4, &mut pairs);
        bencher::black_box(&mut pairs);
    })
}

fn bench_blocks() -> Vec<[f32; 64]> {
    test::pseudo_random_blocks(4096)
}

/// 1024 RGB spatial pairs = 6144 blocks of work, same order of magnitude as
/// the flat 4096-block inverse benches (those process 4096 single blocks;
/// each RGB pair does 6).
fn bench_rgb_pairs() -> Vec<avx512::RgbPairBlocks> {
    let blocks = test::pseudo_random_blocks(1024 * 6);
    blocks
        .chunks_exact(6)
        .map(|c| ([c[0], c[1], c[2]], [c[3], c[4], c[5]]))
        .collect()
}

fn expect_avx2() -> V3 {
    V3::try_new().expect("AVX2 SIMD mode requested, but the AVX2/FMA tier is unavailable")
}

fn expect_sse2() -> V1 {
    V1::try_new().expect("SSE2 SIMD mode requested, but the SSE2 tier is unavailable")
}

fn expect_avx512() -> V4 {
    V4::try_new().expect("AVX-512 SIMD mode requested, but the AVX-512 tier is unavailable")
}

benchmark_group!(
    dct,
    dct_forward_bench_autovectorized,
    dct_forward_bench_sse2,
    dct_forward_bench_avx2,
    dct_forward_bench_avx2_batch,
    dct_inverse_bench_autovectorized,
    dct_inverse_bench_sse2,
    dct_inverse_bench_avx2,
    dct_inverse_bench_avx2_batch,
    dct_inverse_bench_avx512_batch,
    dct_inverse_bench_avx512_batch_quad,
    dct_inverse_bench_avx512_rgb_comp_seq,
    dct_inverse_bench_avx512_rgb_comp_quad
);

benchmark_main!(dct);
