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
/// AVX-512 fused RGB pair steps that entered `decode_pair_dct_csc` (components==3).
pub static DCT_RGB_PAIR_STEPS: AtomicU64 = AtomicU64::new(0);
/// Of those, how many used component-level `inverse_quad` (any two (true,true) comps).
pub static DCT_RGB_COMP_QUAD: AtomicU64 = AtomicU64::new(0);
pub static ASSEMBLE_NS: AtomicU64 = AtomicU64::new(0);
// Whole-`decompress` time and the output buffer allocation inside it, to see
// how much of the wall time falls outside the stages above (chunk parsing,
// channel classification, the per-chunk output allocation)
pub static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
pub static OUT_ALLOC_NS: AtomicU64 = AtomicU64::new(0);
/// Not a duration: total bytes of output buffer allocated, reported as MiB per
/// iteration, so the allocation volume behind `OUT_ALLOC_NS` is visible
pub static OUT_BYTES: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Block-scheduling / thread-wait counters.
//
// Two topologies feed these counters, and they stall differently:
//
//   * `block::reader::ParallelBlockDecompressor` -- a real main thread reads
//     chunks, `spawn`s them, and blocks in `recv()` for results, which it then
//     converts itself. Its stall is consumer starvation: `SCHED_RECV_NS`.
//   * `specific_channels`' `collect_pixels_in_parallel` -- `ThreadPool::scope`,
//     so the feeding closure runs *on a pool worker* and the tasks convert
//     pixels themselves. There is no `recv`; after the last `spawn` the feeder
//     work-steals until the scope closes (`SCHED_DRAIN_NS`).
// ---------------------------------------------------------------------------

/// Chunks handed to the thread pool.
pub static SCHED_TASKS: AtomicU64 = AtomicU64::new(0);
/// Sum over tasks of (worker start - `spawn` call): queueing delay.
pub static SCHED_QUEUE_NS: AtomicU64 = AtomicU64::new(0);
/// Sum over tasks of the actual `decompress_chunk` duration: worker busy time.
pub static SCHED_TASK_NS: AtomicU64 = AtomicU64::new(0);
/// Main-thread time spent reading compressed chunks and spawning them.
pub static SCHED_FEED_NS: AtomicU64 = AtomicU64::new(0);
/// Main-thread time blocked in `recv()` waiting for any worker to finish.
/// Channel-based path (`block::reader`) only: a true idle block.
pub static SCHED_RECV_NS: AtomicU64 = AtomicU64::new(0);
/// Rayon-scope path only: time from the last `spawn` until the scope closed.
/// *Not* idle time -- rayon work-steals on the scope latch, so the feeding
/// thread keeps executing tasks during this window. It measures the tail: how
/// much of the run happens after there is nothing left to hand out, which is
/// where load imbalance between the last chunks shows up.
pub static SCHED_DRAIN_NS: AtomicU64 = AtomicU64::new(0);
/// Wall-clock span of the whole parallel decompression (decompressor lifetime).
pub static SCHED_WALL_NS: AtomicU64 = AtomicU64::new(0);
/// Worker threads in the pool, for the utilization denominator.
pub static SCHED_THREADS: AtomicU64 = AtomicU64::new(0);

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
        &DCT_RGB_PAIR_STEPS,
        &DCT_RGB_COMP_QUAD,
        &ASSEMBLE_NS,
        &TOTAL_NS,
        &OUT_ALLOC_NS,
        &OUT_BYTES,
        &SCHED_TASKS,
        &SCHED_QUEUE_NS,
        &SCHED_TASK_NS,
        &SCHED_FEED_NS,
        &SCHED_RECV_NS,
        &SCHED_DRAIN_NS,
        &SCHED_WALL_NS,
        &SCHED_THREADS,
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
    // Scheduling / thread-wait view of the same run. Only meaningful for a
    // parallel read; a serial read never touches the thread pool, so all of
    // these stay zero and the line is skipped.
    let tasks = SCHED_TASKS.load(Ordering::Relaxed);
    if tasks > 0 {
        let threads = SCHED_THREADS.load(Ordering::Relaxed).max(1);
        let wall_ns = SCHED_WALL_NS.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64;
        let busy_ns = SCHED_TASK_NS.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64;
        // Every worker is available for the whole span, so this is the pool's
        // capacity; whatever the tasks don't fill is idle worker time.
        let feed_ns = SCHED_FEED_NS.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64;
        let wait_ns = SCHED_RECV_NS.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64;
        let drain_ns = SCHED_DRAIN_NS.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64;
        let queue_ns = SCHED_QUEUE_NS.load(Ordering::Relaxed) as f64 / divisor.max(1) as f64;
        let capacity_ns = wall_ns * threads as f64;
        // The feeding thread is itself one of the pool's threads (`ThreadPool::
        // scope` runs its closure on a worker), so its time is occupied, not
        // idle. What's left over is genuinely unused core-time.
        let idle_ns = (capacity_ns - busy_ns - feed_ns).max(0.0);

        eprintln!(
            "dwa-profile sched (avg ms/iter): wall={:.3} threads={} worker_busy={:.3} feed={:.3} idle={:.3} util={:.1}% recv_wait={:.3} drain={:.3} ({:.1}% of wall) queue_delay={:.3} tasks={:.0} queue_delay_per_task_us={:.2}",
            wall_ns / 1e6,
            threads,
            busy_ns / 1e6,
            feed_ns / 1e6,
            idle_ns / 1e6,
            if capacity_ns > 0.0 { 100.0 * (busy_ns + feed_ns) / capacity_ns } else { 0.0 },
            wait_ns / 1e6,
            drain_ns / 1e6,
            if wall_ns > 0.0 { 100.0 * drain_ns / wall_ns } else { 0.0 },
            queue_ns / 1e6,
            tasks as f64 / divisor.max(1) as f64,
            queue_ns * divisor.max(1) as f64 / tasks as f64 / 1e3,
        );
    }

    let pair_steps = DCT_RGB_PAIR_STEPS.load(Ordering::Relaxed);
    let comp_quad = DCT_RGB_COMP_QUAD.load(Ordering::Relaxed);
    if pair_steps > 0 {
        eprintln!(
            "dwa-profile: dct rgb pair steps={}  component-quad hits={}  ({:.1}% of pair steps)",
            pair_steps as f64 / divisor.max(1) as f64,
            comp_quad as f64 / divisor.max(1) as f64,
            100.0 * comp_quad as f64 / pair_steps as f64,
        );
    }
}
