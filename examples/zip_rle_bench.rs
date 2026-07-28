//! Whole-pipeline ZIP/RLE decode wall time.
//!
//!   cargo run --release --example zip_rle_bench -- <file.exr> <0|1 parallel> <iters>

use std::time::Instant;

use exr::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <file> <0|1 parallel> <iters>", args[0]);
        std::process::exit(1);
    }
    let path = &args[1];
    let parallel = args[2] == "1";
    let iters: usize = args[3].parse().expect("iters");

    // Load once into memory so the loop is not disk-bound.
    let bytes = std::fs::read(path).expect("read file");

    let decode = |data: &[u8]| {
        let reader = read()
            .no_deep_data()
            .largest_resolution_level()
            .rgba_channels(
                |res, _| vec![vec![[0f32; 4]; res.width()]; res.height()],
                |pixels, pos, (r, g, b, a): (f32, f32, f32, f32)| {
                    pixels[pos.y()][pos.x()] = [r, g, b, a];
                },
            )
            .all_layers()
            .all_attributes();
        if parallel {
            reader.from_buffered(std::io::Cursor::new(data)).expect("decode")
        } else {
            reader.non_parallel().from_buffered(std::io::Cursor::new(data)).expect("decode")
        }
    };

    // Warmup
    let warm = decode(&bytes);
    let layer = &warm.layer_data[0];
    let pixels = &layer.channel_data.pixels;
    let width = layer.size.width();
    let mut h = 0u64;
    for row in pixels {
        for px in row {
            for &c in px {
                let bits = c.to_bits() as u64;
                h ^= bits.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(h << 6).wrapping_add(h >> 2);
            }
        }
    }
    let _ = width;

    let t0 = Instant::now();
    for _ in 0..iters {
        let img = decode(&bytes);
        std::hint::black_box(img);
    }
    let avg_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!(
        "file={} parallel={} iters={} avg_ms={:.3} bithash={:016x}",
        path, parallel, iters, avg_ms, h
    );
}
