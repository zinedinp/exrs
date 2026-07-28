extern crate exr;

use std::time::{Duration, Instant};

use exr::prelude::*;

// Same rolling xor/mix hash shape as `dwa_bench.rs`'s `bithash_*` -- cheap,
// order-sensitive, good enough to prove two reads produced identical pixels
// without keeping a second full-size buffer around.
fn bithash(width: usize, pixels: &[[f32; 4]], mut h: u64) -> u64 {
    for row in pixels.chunks_exact(width) {
        for pixel in row {
            for &channel in pixel {
                let bits = channel.to_bits() as u64;
                h ^= bits.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(h << 6).wrapping_add(h >> 2);
            }
        }
    }
    h
}

/// Generates a large tiled RGBA file if it doesn't already exist. Tiles are
/// small (32x32) relative to the image so there are many of them (a
/// 4096x4096 image has 128x128 = 16384 tiles)
fn generate_if_missing(path: &str, width: usize, height: usize) {
    if std::path::Path::new(path).exists() {
        return;
    }

    eprintln!("generating {path} ({width}x{height}, tiled 32x32, uncompressed)...");
    let channels = SpecificChannels::rgba(|Vec2(x, y)| {
        (
            (x as f32) / (width as f32),
            (y as f32) / (height as f32),
            ((x + y) as f32) / ((width + height) as f32),
            1.0f32,
        )
    });

    let image = Image::from_encoded_channels(
        (width, height),
        Encoding {
            compression: Compression::Uncompressed,
            blocks: Blocks::Tiles(Vec2(32, 32)),
            line_order: LineOrder::Unspecified,
        },
        channels,
    );

    image.write().to_file(path).expect("failed to write generated tiled test file");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <iters> [width] [height]", args[0]);
        std::process::exit(1);
    }

    let iters: usize = args[1].parse().expect("iters must be a number");
    let width: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(4096);
    let height: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(4096);

    let path = format!("/tmp/claude-1000/tiled_bench_{width}x{height}.exr");
    generate_if_missing(&path, width, height);

    let mut total = Duration::ZERO;
    let mut hash = 0u64;

    for _ in 0..iters {
        let start = Instant::now();

        let pixels = read()
            .no_deep_data()
            .largest_resolution_level()
            .specific_channels()
            .required("R")
            .required("G")
            .required("B")
            .optional("A", 1.0f32)
            .collect_pixels_in_parallel(
                |resolution, _channels| FlatRowMajorPixelStorage {
                    width: resolution.width(),
                    pixels: vec![[0.0f32; 4]; resolution.width() * resolution.height()],
                },
                |row: &mut [[f32; 4]], x, (r, g, b, a): (f32, f32, f32, f32)| {
                    row[x] = [r, g, b, a];
                },
            )
            .first_valid_layer()
            .all_attributes()
            .from_file(&path)
            .expect("failed to read tiled exr file")
            .layer_data
            .channel_data
            .pixels;

        total += start.elapsed();
        hash = bithash(pixels.width, &pixels.pixels, hash);
    }

    println!(
        "iters={iters} width={width} height={height} total_ms={:.3} mean_ms={:.3} hash={hash:016x}",
        total.as_secs_f64() * 1000.0,
        total.as_secs_f64() * 1000.0 / iters as f64
    );
}
