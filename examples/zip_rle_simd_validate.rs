//! Validate that ZIP/RLE reconstruct stays SIMD under LLVM, and measure cache/TLB.
//!
//! Assembly check (symbol must contain paddb / pshufb / pslldq, not only scalar adds):
//!   RUSTFLAGS="-C target-cpu=native" cargo build --release --example zip_rle_simd_validate
//!   llvm-objdump -d --no-show-raw-insn target/release/examples/zip_rle_simd_validate \
//!     | rg 'paddb|pshufb|pslldq|vpaddb|vpshufb|vpslldq|vextracti32x4|vpermb'
//!
//! Cache / TLB A/B (needs `perf`):
//!   perf stat -e cycles,instructions,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
//!     dTLB-load-misses,page-faults -- \
//!     taskset -c 0 ./target/release/examples/zip_rle_simd_validate dispatch 4194304 200
//!
//! Usage: zip_rle_simd_validate <scalar|sse|avx2|avx512|dispatch> <nbytes> <reps>

use std::env;
use std::hint::black_box;

use exr::compression::optimize_bytes::{differences_to_samples, differences_to_samples_scalar};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use exr::compression::optimize_bytes::x86::{avx2, avx512, sse};

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "dispatch".into());
    let nbytes: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4 * 1024 * 1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);

    // Pseudo-random diffs so the integrate chain is not all zeros (keeps deps live).
    let mut base = vec![0u8; nbytes];
    for (i, b) in base.iter_mut().enumerate() {
        *b = (i.wrapping_mul(131) as u8).wrapping_add((i >> 8) as u8).wrapping_add(3);
    }

    let mut buf = base.clone();
    // Touch once so page faults land in setup, not in the timed path.
    for b in &mut buf {
        *b = b.wrapping_add(1);
    }
    buf.copy_from_slice(&base);

    match mode.as_str() {
        "scalar" => {
            for i in 0..reps {
                if i % 4 == 0 {
                    buf.copy_from_slice(&base);
                }
                differences_to_samples_scalar(&mut buf);
                black_box(&buf);
            }
        }
        "sse" | "simd" => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if let Some(v2) = pulp::x86::V2::try_new() {
                    for i in 0..reps {
                        if i % 4 == 0 {
                            buf.copy_from_slice(&base);
                        }
                        sse::differences_to_samples(v2, &mut buf);
                        black_box(&buf);
                    }
                    finish(&mode, nbytes, reps, &buf);
                    return;
                }
            }
            eprintln!("V2 unavailable; falling back to dispatch");
            run_dispatch(&mut buf, &base, reps);
        }
        "avx2" => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if let Some(v3) = pulp::x86::V3::try_new() {
                    for i in 0..reps {
                        if i % 4 == 0 {
                            buf.copy_from_slice(&base);
                        }
                        avx2::differences_to_samples(v3, &mut buf);
                        black_box(&buf);
                    }
                    finish(&mode, nbytes, reps, &buf);
                    return;
                }
            }
            eprintln!("V3 unavailable; falling back to dispatch");
            run_dispatch(&mut buf, &base, reps);
        }
        "avx512" => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if let Some(v4) = pulp::x86::V4::try_new() {
                    for i in 0..reps {
                        if i % 4 == 0 {
                            buf.copy_from_slice(&base);
                        }
                        avx512::differences_to_samples(v4, &mut buf);
                        black_box(&buf);
                    }
                    finish(&mode, nbytes, reps, &buf);
                    return;
                }
            }
            eprintln!("V4 unavailable; falling back to dispatch");
            run_dispatch(&mut buf, &base, reps);
        }
        "dispatch" => {
            run_dispatch(&mut buf, &base, reps);
        }
        other => {
            eprintln!("unknown mode {other:?}; use scalar|sse|avx2|avx512|dispatch");
            std::process::exit(2);
        }
    }

    finish(&mode, nbytes, reps, &buf);
}

fn run_dispatch(buf: &mut [u8], base: &[u8], reps: usize) {
    for i in 0..reps {
        if i % 4 == 0 {
            buf.copy_from_slice(base);
        }
        differences_to_samples(buf);
        black_box(&buf);
    }
}

fn finish(mode: &str, nbytes: usize, reps: usize, buf: &[u8]) {
    eprintln!(
        "mode={mode} nbytes={nbytes} reps={reps} checksum={}",
        buf.iter().fold(0u64, |a, &b| a.wrapping_add(b as u64))
    );
}
