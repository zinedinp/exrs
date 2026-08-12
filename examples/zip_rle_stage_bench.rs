//! Isolated ZIP/RLE reconstruct stage microbench: scalar vs SSE vs AVX2 vs AVX-512.
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example zip_rle_stage_bench

use std::hint::black_box;
use std::time::Instant;

use exr::compression::optimize_bytes::{
    differences_to_samples, differences_to_samples_scalar, interleave_byte_blocks,
    interleave_byte_blocks_scalar, samples_to_differences, samples_to_differences_scalar,
    separate_bytes_fragments, separate_bytes_fragments_scalar,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use exr::compression::optimize_bytes::x86::{avx2, avx512bw as avx512, ssse3 as sse};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use miraculix::x86::detect_features;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use miraculix::x86::ops::avx::avx2::Avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use miraculix::x86::ops::avx512::{avx512bw::Avx512Bw, avx512f::Avx512f};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use miraculix::x86::ops::sse::{sse2::Sse2, ssse3::Ssse3};

fn gbs(n: usize, secs: f64) -> f64 {
    (n as f64 / secs) / 1e9
}

fn time_recon(_n: usize, reps: usize, base: &[u8], mut f: impl FnMut(&mut [u8])) -> f64 {
    let mut b = base.to_vec();
    for _ in 0..4 {
        f(&mut b);
        b.copy_from_slice(base);
    }
    let t = Instant::now();
    for i in 0..reps {
        if i % 8 == 0 {
            b.copy_from_slice(base);
        }
        f(&mut b);
        black_box(&b);
    }
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    println!(
        "ZIP/RLE reconstruct A/B: scalar | sse16 | avx2-lane | avx2-sse-tail | avx512-lane | avx512-masked | production\n"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let features = detect_features();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let (sse2, ssse3, avx2_token, avx512f, avx512bw) = (
        Sse2::from_features(features),
        Ssse3::from_features(features),
        Avx2::from_features(features),
        Avx512f::from_features(features),
        Avx512Bw::from_features(features),
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let v2 = sse2.zip(ssse3);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let v3 = avx2_token.zip(sse2).zip(ssse3).map(|((a, s), t)| (a, s, t));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let v4 = avx512f.zip(avx512bw).zip(v3).map(|((f, bw), (a, s, t))| (f, bw, a, s, t));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    println!(
        "CPU tokens: SSE={}  AVX2={}  AVX512={}\n",
        v2.is_some(),
        v3.is_some(),
        v4.is_some()
    );

    // Include non-multiple-of-64 sizes so SSE-tail / masked-tail differences show up.
    for &n in &[
        48usize, // <64, remainder-heavy
        80,      // 64+16
        96,      // 64+32
        64 * 1024 + 48,
        256 * 1024 + 17,
        1024 * 1024,
        4 * 1024 * 1024,
    ] {
        let base: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(17) as u8).wrapping_add(3)).collect();
        let reps = ((64 * 1024 * 1024) / n).max(32);

        let pred_scalar = time_recon(n, reps, &base, |b| differences_to_samples_scalar(b));
        let pred_dispatch = time_recon(n, reps, &base, |b| differences_to_samples(b));

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let pred_sse = v2.map(|(sse2, ssse3)| {
            time_recon(n, reps, &base, |b| sse::differences_to_samples(sse2, ssse3, b))
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let pred_lane = v3.map(|(avx2_t, sse2, ssse3)| {
            time_recon(n, reps, &base, |b| {
                avx2::differences_to_samples_lane(avx2_t, sse2, ssse3, b)
            })
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let pred_sse_tail = v3.map(|(avx2_t, sse2, ssse3)| {
            time_recon(n, reps, &base, |b| {
                avx2::differences_to_samples_lane_sse_tail(avx2_t, sse2, ssse3, b)
            })
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let pred_full = v3.map(|(avx2_t, sse2, ssse3)| {
            time_recon(n, reps, &base, |b| {
                avx2::differences_to_samples_full(avx2_t, sse2, ssse3, b)
            })
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let pred_avx512 = v4.map(|(f, bw, avx2_t, sse2, ssse3)| {
            time_recon(n, reps, &base, |b| {
                avx512::differences_to_samples_lane(f, bw, avx2_t, sse2, ssse3, b)
            })
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let pred_avx512_m = v4.map(|(f, bw, avx2_t, sse2, ssse3)| {
            time_recon(n, reps, &base, |b| {
                avx512::differences_to_samples_lane_masked(f, bw, avx2_t, sse2, ssse3, b)
            })
        });

        // Encode predictor + interleave still scalar-only (regression ports commented).
        let enc = time_recon(n, reps, &base, |b| samples_to_differences(b));
        let enc_s = time_recon(n, reps, &base, |b| samples_to_differences_scalar(b));

        let ireps = reps.min(128);
        let mut tmp = base.clone();
        let mut scratch = vec![0u8; n];
        for _ in 0..4 {
            tmp.copy_from_slice(&base);
            interleave_byte_blocks_scalar(&tmp, &mut scratch);
            tmp.copy_from_slice(&scratch);
        }
        let t = Instant::now();
        for _ in 0..ireps {
            tmp.copy_from_slice(&base);
            interleave_byte_blocks_scalar(&tmp, &mut scratch);
            tmp.copy_from_slice(&scratch);
            black_box(&tmp);
        }
        let inter_scalar = t.elapsed().as_secs_f64() / ireps as f64;

        let mut tmp = base.clone();
        for _ in 0..4 {
            tmp.copy_from_slice(&base);
            interleave_byte_blocks(&mut tmp);
        }
        let t = Instant::now();
        for _ in 0..ireps {
            tmp.copy_from_slice(&base);
            interleave_byte_blocks(&mut tmp);
            black_box(&tmp);
        }
        let inter_prod = t.elapsed().as_secs_f64() / ireps as f64;

        let mut tmp = base.clone();
        for _ in 0..4 {
            tmp.copy_from_slice(&base);
            separate_bytes_fragments_scalar(&tmp, &mut scratch);
            tmp.copy_from_slice(&scratch);
        }
        let t = Instant::now();
        for _ in 0..ireps {
            tmp.copy_from_slice(&base);
            separate_bytes_fragments_scalar(&tmp, &mut scratch);
            tmp.copy_from_slice(&scratch);
            black_box(&tmp);
        }
        let sep_scalar = t.elapsed().as_secs_f64() / ireps as f64;

        let mut tmp = base.clone();
        for _ in 0..4 {
            tmp.copy_from_slice(&base);
            separate_bytes_fragments(&mut tmp);
        }
        let t = Instant::now();
        for _ in 0..ireps {
            tmp.copy_from_slice(&base);
            separate_bytes_fragments(&mut tmp);
            black_box(&tmp);
        }
        let sep_prod = t.elapsed().as_secs_f64() / ireps as f64;

        println!("n={:>8} ({:>7.1} KiB)  reps={reps}", n, n as f64 / 1024.0);
        println!("  reconstruct  scalar       {:5.1} GB/s", gbs(n, pred_scalar));
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(t) = pred_sse {
                println!(
                    "  reconstruct  sse16        {:5.1} GB/s  ({:.2}x vs scalar)",
                    gbs(n, t),
                    pred_scalar / t
                );
            }
            if let Some(t) = pred_lane {
                println!(
                    "  reconstruct  avx2-lane    {:5.1} GB/s  ({:.2}x vs scalar, {:.2}x vs sse)",
                    gbs(n, t),
                    pred_scalar / t,
                    pred_sse.unwrap_or(t) / t
                );
            }
            if let Some(t) = pred_sse_tail {
                println!(
                    "  reconstruct  avx2-sseTail {:5.1} GB/s  ({:.2}x vs scalar, {:.2}x vs avx2-lane)",
                    gbs(n, t),
                    pred_scalar / t,
                    pred_lane.unwrap_or(t) / t
                );
            }
            if let Some(t) = pred_full {
                println!(
                    "  reconstruct  avx2-full    {:5.1} GB/s  ({:.2}x vs scalar, {:.2}x vs sse)",
                    gbs(n, t),
                    pred_scalar / t,
                    pred_sse.unwrap_or(t) / t
                );
            }
            if let Some(t) = pred_avx512 {
                println!(
                    "  reconstruct  avx512-lane  {:5.1} GB/s  ({:.2}x vs scalar, {:.2}x vs avx2-lane)",
                    gbs(n, t),
                    pred_scalar / t,
                    pred_lane.unwrap_or(t) / t
                );
            }
            if let Some(t) = pred_avx512_m {
                println!(
                    "  reconstruct  avx512-mask  {:5.1} GB/s  ({:.2}x vs scalar, {:.2}x vs avx2-lane)",
                    gbs(n, t),
                    pred_scalar / t,
                    pred_lane.unwrap_or(t) / t
                );
            }
        }
        println!(
            "  reconstruct  dispatch     {:5.1} GB/s  ({:.2}x vs scalar)",
            gbs(n, pred_dispatch),
            pred_scalar / pred_dispatch
        );
        println!(
            "  samples_to_diff (scalar only) {:5.1} GB/s  (dispatch/scalar={:.2}x — expect ~1)",
            gbs(n, enc),
            enc_s / enc
        );
        println!(
            "  interleave     scalar={:5.1}  prod={:5.1} GB/s  ({:.2}x)",
            gbs(n, inter_scalar),
            gbs(n, inter_prod),
            inter_scalar / inter_prod
        );
        println!(
            "  separate       scalar={:5.1}  prod={:5.1} GB/s  ({:.2}x)",
            gbs(n, sep_scalar),
            gbs(n, sep_prod),
            sep_scalar / sep_prod
        );
    }
}
