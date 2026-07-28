//! Widening `rle`'s encoder run-detection scan from a byte-at-a-time compare
//! to a wordwise (u64) SWAR compare. Follow-up to
//! `dwa-compression-rle-branch-miss-investigation` and
//! `rle-encoder-swar-widening-idea`
//!
//! Bench: `cargo test --release -p exr -- --ignored --nocapture bench_rle_encode_widen`.
#![allow(dead_code)]

use super::rle::run_length_at as run_length_at_widened;

// Mirrors the private constants in `rle.rs` (not reachable from a sibling
// module)
const MIN_RUN_LENGTH: usize = 3;
const MAX_RUN_LENGTH: usize = 127;

/// Historical reference: how `rle::pack_rle_tokens`'s run-length scan worked
/// before widening (one byte per iteration).
fn run_length_at_scalar(data: &[u8], start: usize) -> usize {
    let target = data[start];
    let limit = (data.len() - start).min(MAX_RUN_LENGTH + 1);
    let mut len = 1;
    while len < limit && data[start + len] == target {
        len += 1;
    }
    len
}

/// `rle::pack_rle_tokens`, reimplemented here so it can be parameterized by
/// which run-length function to use.
fn pack_rle_tokens_with(data_le: &[u8], run_length_at: impl Fn(&[u8], usize) -> usize) -> Vec<u8> {
    let mut compressed_le = Vec::with_capacity(data_le.len());
    let mut run_start = 0;

    while run_start < data_le.len() {
        let run_len = run_length_at(data_le, run_start);
        let run_end = run_start + run_len;

        if run_len >= MIN_RUN_LENGTH {
            compressed_le.push((run_len as i32 - 1) as u8);
            compressed_le.push(data_le[run_start]);
            run_start = run_end;
        } else {
            // Literal-run scan: unchanged from production, not part of this
            // experiment
            let mut run_end = run_end;
            while run_end < data_le.len()
                && (run_end + 1 >= data_le.len()
                    || data_le[run_end] != data_le[run_end + 1]
                    || run_end + 2 >= data_le.len()
                    || data_le[run_end + 1] != data_le[run_end + 2])
                && run_end - run_start < MAX_RUN_LENGTH
            {
                run_end += 1;
            }

            compressed_le.push((run_start as i32 - run_end as i32) as u8);
            compressed_le.extend_from_slice(&data_le[run_start..run_end]);
            run_start = run_end;
        }
    }

    compressed_le
}

/// Synthetic but content-shaped corpus: real delta-coded (post
/// `samples_to_differences`) photographic data clusters near byte value 128
/// (small +/- deltas) with occasional long flat runs (constant-color
/// regions -- sky, alpha=1, borders). Mixes both so the run-length scan sees
/// a realistic mix of "exits after 0-2 bytes" and "runs for a while".
///
/// `flat_fraction` is the approximate share of output bytes that fall inside
/// a long flat run (length 3..=127) rather than noisy single-byte deltas.
fn realistic_corpus(len: usize, flat_fraction: f64, seed: u64) -> Vec<u8> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(len);

    while out.len() < len {
        if rng.random_range(0.0..1.0) < flat_fraction {
            let run_len = rng.random_range(MIN_RUN_LENGTH..=MAX_RUN_LENGTH);
            let value = 128u8.wrapping_add(rng.random_range(-4i16..=4i16) as u8);
            out.extend(std::iter::repeat(value).take(run_len));
        } else {
            let noisy_len = rng.random_range(1..=24);
            for _ in 0..noisy_len {
                out.push(128u8.wrapping_add(rng.random_range(-20i16..=20i16) as u8));
            }
        }
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{RngExt, SeedableRng};

    /// The old scalar reference and the shipped widened kernel must agree on
    /// every start position, across edge cases (near buffer end, near the
    /// MAX_RUN_LENGTH boundary, all-same-byte, no-repeats) and randomized
    /// realistic content.
    #[test]
    fn run_length_functions_agree() {
        let cases: Vec<Vec<u8>> = vec![
            vec![5],
            vec![5, 5],
            vec![5, 5, 5],
            vec![5; 200],
            (0..200).map(|i| i as u8).collect(),
            realistic_corpus(10_000, 0.3, 1),
            realistic_corpus(10_000, 0.05, 2),
            realistic_corpus(10_000, 0.9, 3),
        ];

        for data in &cases {
            for start in 0..data.len() {
                let scalar = run_length_at_scalar(data, start);
                let widened = run_length_at_widened(data, start);
                assert_eq!(
                    scalar, widened,
                    "mismatch at start={start} len={} data around: {:?}",
                    data.len(),
                    &data[start..(start + 8).min(data.len())]
                );
            }
        }
    }

    /// The parameterized reimplementation must match the real production
    /// `rle::pack_rle_tokens` byte-for-byte with either run-length function;
    /// regression guard now that production itself uses the widened one.
    #[test]
    fn matches_production_output() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..20 {
            let len = rng.random_range(0..5000);
            let flat_fraction = rng.random_range(0.0..1.0);
            let data = realistic_corpus(len, flat_fraction, rng.random());

            let production = super::super::rle::pack_rle_tokens(&data);
            let scalar_helper = pack_rle_tokens_with(&data, run_length_at_scalar);
            let widened_helper = pack_rle_tokens_with(&data, run_length_at_widened);

            assert_eq!(&production[..], &scalar_helper[..], "scalar helper diverged, len={len}");
            assert_eq!(&production[..], &widened_helper[..], "widened helper diverged, len={len}");
        }
    }

    /// Isolated A/B that gated shipping: old scalar run-length scan vs the
    /// widened kernel now in production, composed into the full encoder
    /// (kernel-only timing was tried first and discarded
    /// `cargo test --release -p exr -- --ignored --nocapture bench_rle_encode_widen`
    #[test]
    #[ignore]
    fn bench_rle_encode_widen() {
        const LEN: usize = 16 * 1024 * 1024;
        const ROUNDS: usize = 6;

        for &flat_fraction in &[0.05, 0.3, 0.7] {
            let data = realistic_corpus(LEN, flat_fraction, 7);

            let mut full_scalar_ms = Vec::new();
            let mut full_widened_ms = Vec::new();

            for _ in 0..ROUNDS {
                let start = std::time::Instant::now();
                let out = pack_rle_tokens_with(&data, run_length_at_scalar);
                full_scalar_ms.push(start.elapsed().as_secs_f64());
                std::hint::black_box(&out);

                let start = std::time::Instant::now();
                let out = pack_rle_tokens_with(&data, run_length_at_widened);
                full_widened_ms.push(start.elapsed().as_secs_f64());
                std::hint::black_box(&out);
            }

            let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let fs = avg(&full_scalar_ms);
            let fw = avg(&full_widened_ms);

            eprintln!(
                "flat_fraction={flat_fraction:.2} ({LEN} bytes x {ROUNDS} rounds):  scalar {:>8.3} ms | widened {:>8.3} ms ({:+.2}%)",
                fs * 1e3,
                fw * 1e3,
                (fw / fs - 1.0) * 100.0,
            );
        }
    }
}
