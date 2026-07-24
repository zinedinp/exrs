extern crate exr;

use std::time::Instant;

use exr::prelude::*;

// Matches bench_openexr.cpp's bithash: whole R plane, then whole G plane,
// then whole B plane (row-major), so hashes are directly comparable across
// the exrs and OpenEXR C++ implementations for the same file.
fn bithash(pixels: &[Vec<[f32; 4]>], mut h: u64) -> u64 {
    for channel_index in 0..3 {
        for row in pixels {
            for pixel in row {
                let bits = half::f16::from_f32(pixel[channel_index]).to_bits() as u64;
                h ^=
                    bits.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(h << 6).wrapping_add(h >> 2);
            }
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <file> <0|1 parallel> <iters> [0|1 parallel_pixels]", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let parallel = args[2] == "1";
    let iters: usize = args[3].parse().expect("iters must be a number");
    // Use `collect_pixels_in_parallel` (writes pixels from the decompression
    // worker threads) instead of the default `collect_pixels` (writes
    // pixels serially on the driving thread as blocks arrive). Only affects
    // anything when `parallel` is also set.
    let parallel_pixels = args.get(4).map(|a| a == "1").unwrap_or(false);

    let mut total = std::time::Duration::ZERO;
    let mut hash = 0u64;

    #[cfg(feature = "dwa-profile")]
    exr::compression::dwa::profile::reset();

    for _ in 0..iters {
        let start = Instant::now();

        let pixels = if parallel_pixels {
            let mut reader = read()
                .no_deep_data()
                .largest_resolution_level()
                .specific_channels()
                .required("R")
                .required("G")
                .required("B")
                .optional("A", 1.0f32)
                .collect_pixels_in_parallel(
                    |resolution, _| vec![vec![[0.0f32; 4]; resolution.width()]; resolution.height()],
                    |row, x, (r, g, b, a): (f32, f32, f32, f32)| {
                        row[x] = [r, g, b, a];
                    },
                )
                .first_valid_layer()
                .all_attributes();

            if !parallel {
                reader = reader.non_parallel();
            }

            reader.from_file(path).expect("failed to read exr file").layer_data.channel_data.pixels
        } else {
            let mut reader = read()
                .no_deep_data()
                .largest_resolution_level()
                .rgba_channels(
                    |resolution, _| vec![vec![[0.0f32; 4]; resolution.width()]; resolution.height()],
                    |pixels, position, (r, g, b, a): (f32, f32, f32, f32)| {
                        pixels[position.y()][position.x()] = [r, g, b, a];
                    },
                )
                .first_valid_layer()
                .all_attributes();

            if !parallel {
                reader = reader.non_parallel();
            }

            reader.from_file(path).expect("failed to read exr file").layer_data.channel_data.pixels
        };

        total += start.elapsed();
        hash = bithash(&pixels, hash);
    }

    println!(
        "file={} parallel={} parallel_pixels={} iters={} avg_ms={:.3} bithash={:016x}",
        path,
        parallel,
        parallel_pixels,
        iters,
        total.as_secs_f64() * 1000.0 / iters as f64,
        hash
    );

    #[cfg(feature = "dwa-profile")]
    exr::compression::dwa::profile::report(iters as u64);
}
