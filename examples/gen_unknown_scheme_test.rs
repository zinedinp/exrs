// Scratch generator: DWA test fixture with a channel that falls into
// `CompressorScheme::Unknown`
// only R/G/B/Y/BY/RY (F16/F32) and A (U32/F16/F32) match a rule; anything
// else, like a "Z" depth channel, is classified Unknown and goes through
// `decode_unknown_section`'s plain zlib section instead of lossy DCT or RLE).
extern crate exr;

use exr::prelude::*;

fn main() {
    let size = (4096, 4096);
    let (w, h) = size;
    let n = w * h;

    let mut r = vec![f16::ZERO; n];
    let mut g = vec![f16::ZERO; n];
    let mut b = vec![f16::ZERO; n];
    let mut a = vec![f16::from_f32(1.0); n];
    let mut z = vec![0f32; n];

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            // some texture so the DCT/AC streams aren't trivially empty
            let noise = ((x * 37 + y * 101) % 997) as f32 / 997.0;
            r[i] = f16::from_f32((fx * 0.7 + noise * 0.3).clamp(0.0, 1.0));
            g[i] = f16::from_f32((fy * 0.7 + noise * 0.3).clamp(0.0, 1.0));
            b[i] = f16::from_f32(((1.0 - fx) * 0.5 + (1.0 - fy) * 0.3).clamp(0.0, 1.0));
            // plausible depth field: smooth-ish with some per-pixel jitter,
            // like a real Z-AOV -- not so smooth that zlib trivially wins
            let depth = 10.0 + fx * 40.0 + fy * 20.0 + noise * 5.0;
            z[i] = depth;
        }
    }
    let _ = &mut a; // constant alpha, cheap RLE like the real fixtures

    let channels = AnyChannels::sort(smallvec::smallvec![
        AnyChannel::new("R", FlatSamples::F16(r)),
        AnyChannel::new("G", FlatSamples::F16(g)),
        AnyChannel::new("B", FlatSamples::F16(b)),
        AnyChannel::new("A", FlatSamples::F16(a)),
        AnyChannel::new("Z", FlatSamples::F32(z)),
    ]);

    let layer = Layer::new(
        size,
        LayerAttributes::named("unknown-scheme-test"),
        Encoding::default(),
        channels,
    );

    for (name, compression) in
        [("unknown_scheme_test_dwaa", Compression::DWAA(None)), ("unknown_scheme_test_dwab", Compression::DWAB(None))]
    {
        let mut layer = layer.clone();
        layer.encoding.compression = compression;
        let image = Image::from_layer(layer);
        let path = format!("../{name}.exr");
        image.write().to_file(&path).unwrap();
        println!("wrote {path}");
    }
}
