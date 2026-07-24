// Per-stage wall-clock timing for the DWA decode pipeline, gated behind the
// `dwa-profile` feature. Stages match `DwaCompressor_uncompress` in
// OpenEXR's `internal_dwa_compressor.h` for direct comparison:
// unknown-section inflate, AC Huffman decode, DC inflate+reconstruct,
// RLE inflate+unpack, lossy DCT decode (dequant+IDCT+CSC+to-linear), and
// final scanline assembly. Atomics so parallel (rayon) decode can accumulate
// from multiple threads; under parallel decode the totals are summed CPU
// time across threads, not wall time.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub static UNKNOWN_NS: AtomicU64 = AtomicU64::new(0);
pub static AC_NS: AtomicU64 = AtomicU64::new(0);
pub static DC_NS: AtomicU64 = AtomicU64::new(0);
pub static RLE_NS: AtomicU64 = AtomicU64::new(0);
pub static DCT_NS: AtomicU64 = AtomicU64::new(0);
pub static ASSEMBLE_NS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
pub struct Timer(Instant);

pub fn start() -> Timer { Timer(Instant::now()) }

impl Timer {
    pub fn stop(self, counter: &AtomicU64) {
        counter.fetch_add(self.0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

pub fn reset() {
    for counter in [&UNKNOWN_NS, &AC_NS, &DC_NS, &RLE_NS, &DCT_NS, &ASSEMBLE_NS] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print accumulated totals divided by `divisor` (e.g. iteration count), in milliseconds.
pub fn report(divisor: u64) {
    let ms = |counter: &AtomicU64| {
        counter.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64 / 1_000_000.0
    };
    eprintln!(
        "dwa-profile (avg ms/iter): unknown={:.3} ac_huffman={:.3} dc={:.3} rle={:.3} lossy_dct={:.3} assemble={:.3}",
        ms(&UNKNOWN_NS),
        ms(&AC_NS),
        ms(&DC_NS),
        ms(&RLE_NS),
        ms(&DCT_NS),
        ms(&ASSEMBLE_NS),
    );
}
