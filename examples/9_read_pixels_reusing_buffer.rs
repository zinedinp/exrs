extern crate exr;

use std::cell::Cell;

/// Read the same image multiple times (standing in for, say, successive
/// frames of a sequence at the same resolution), reusing one pixel buffer
/// across every read instead of letting each read allocate its own.
///
/// Preferred API: `collect_flat_pixels` (flat buffer + `PixelSink`).
/// `create_pixels` runs once per `.from_file(..)` and is a plain `Fn`, so it
/// cannot own a `&mut` to external state directly
fn main() {
    use exr::prelude::*;

    let buffer: Cell<Vec<[f32; 4]>> = Cell::new(Vec::new());

    let reader = read()
        .no_deep_data()
        .largest_resolution_level()
        .specific_channels()
        .required("R")
        .required("G")
        .required("B")
        .optional("A", 1.0)
        .collect_flat_pixels(
            |resolution, _channels| {
                let mut pixels = buffer.take();
                pixels.resize(resolution.width() * resolution.height(), [0.0; 4]);
                FlatRowMajorPixelStorage { width: resolution.width(), pixels }
            },
            |row: &mut [[f32; 4]], x, (r, g, b, a): (f32, f32, f32, f32)| {
                row[x] = [r, g, b, a];
            },
        )
        .first_valid_layer()
        .all_attributes();

    for frame_index in 0..3 {
        let image = reader
            .clone()
            .from_file("generated_rgba_with_meta.exr")
            .expect("run example `1a_write_rgba_with_metadata` to generate this image file");

        let pixels = image.layer_data.channel_data.pixels;
        println!("frame {frame_index}: top left pixel = {:?}", pixels.pixels[0]);

        buffer.set(pixels.pixels); // hand the buffer back for the next iteration's `create_pixels`
    }
}
