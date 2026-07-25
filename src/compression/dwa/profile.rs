// Per-stage wall-clock timing for the DWA decode pipeline, gated behind the
// `dwa-profile` feature. Stages match `DwaCompressor_uncompress` in
// OpenEXR's `internal_dwa_compressor.h` for direct comparison:
// unknown-section inflate, AC Huffman decode, DC inflate+reconstruct,
// RLE inflate+unpack, lossy DCT decode (dequant+IDCT+CSC+to-linear), and
// final scanline assembly. Atomics so parallel (rayon) decode can accumulate
// from multiple threads; under parallel decode the totals are summed CPU
// time across threads, not wall time.
//
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub static UNKNOWN_NS: AtomicU64 = AtomicU64::new(0);
pub static AC_NS: AtomicU64 = AtomicU64::new(0);
pub static DC_NS: AtomicU64 = AtomicU64::new(0);
pub static RLE_NS: AtomicU64 = AtomicU64::new(0);
// RLE sub-stages, mirroring the two calls OpenEXR's timed t3 block makes
// (`exr_uncompress_buffer` then `internal_rle_decompress`), so the 2.8x
// stage-level gap can be attributed to one or the other.
pub static RLE_INFLATE_NS: AtomicU64 = AtomicU64::new(0);
pub static RLE_ALLOC_NS: AtomicU64 = AtomicU64::new(0);
pub static RLE_UNPACK_NS: AtomicU64 = AtomicU64::new(0);
pub static DCT_NS: AtomicU64 = AtomicU64::new(0);
pub static ASSEMBLE_NS: AtomicU64 = AtomicU64::new(0);
// Whole-`decompress` time and the output buffer allocation inside it, to see
// how much of the wall time falls outside the stages above (chunk parsing,
// channel classification, the per-chunk output allocation)
pub static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
pub static OUT_ALLOC_NS: AtomicU64 = AtomicU64::new(0);
/// Not a duration: total bytes of output buffer allocated, reported as MiB per
/// iteration, so the allocation volume behind `OUT_ALLOC_NS` is visible
pub static OUT_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
pub struct Timer(Instant);

pub fn start() -> Timer { Timer(Instant::now()) }

impl Timer {
    pub fn stop(self, counter: &AtomicU64) {
        counter.fetch_add(self.0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

pub fn reset() {
    for counter in [
        &UNKNOWN_NS,
        &AC_NS,
        &DC_NS,
        &RLE_NS,
        &RLE_INFLATE_NS,
        &RLE_ALLOC_NS,
        &RLE_UNPACK_NS,
        &DCT_NS,
        &ASSEMBLE_NS,
        &TOTAL_NS,
        &OUT_ALLOC_NS,
        &OUT_BYTES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print accumulated totals divided by `divisor` (e.g. iteration count), in milliseconds.
pub fn report(divisor: u64) {
    let ms = |counter: &AtomicU64| {
        counter.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64 / 1_000_000.0
    };
    eprintln!(
        "dwa-profile (avg ms/iter): unknown={:.3} ac_huffman={:.3} dc={:.3} rle={:.3} (rle_inflate={:.3} rle_alloc={:.3} rle_unpack={:.3}) lossy_dct={:.3} assemble={:.3} out_alloc={:.3} total={:.3}",
        ms(&UNKNOWN_NS),
        ms(&AC_NS),
        ms(&DC_NS),
        ms(&RLE_NS),
        ms(&RLE_INFLATE_NS),
        ms(&RLE_ALLOC_NS),
        ms(&RLE_UNPACK_NS),
        ms(&DCT_NS),
        ms(&ASSEMBLE_NS),
        ms(&OUT_ALLOC_NS),
        ms(&TOTAL_NS),
    );
    eprintln!(
        "dwa-profile: out_alloc volume = {:.1} MiB/iter",
        OUT_BYTES.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64 / (1024.0 * 1024.0),
    );
}
