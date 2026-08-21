//! Reuse of `Vec<u8>` buffers for compressed chunks and decompressed blocks.
//!
//! Keeping a buffer and recycling it is cheaper than allocating a fresh one
//! for every chunk. Take with `take_zeroed` / `take_with_capacity`, give back
//! with `recycle`. Skipping `recycle` just drops the buffer.

use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicBool, Ordering},
};

/// Cap on pooled buffers: one per thread of the default decompress pool.
/// `ParallelBlockDecompressor` keeps at most `max_threads` blocks in flight.
const MAX_POOLED_BUFFERS: usize = 8;

/// Cap on retained bytes. Large tiles give up some reuse rather than keep more.
const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024;

static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static POOL_IS_USED: AtomicBool = AtomicBool::new(false);

fn lock_pool() -> MutexGuard<'static, Vec<Vec<u8>>> {
    // Do not poison the pool on panic; a half-updated pool only costs a missed
    // reuse.
    POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Zeroed buffer of exactly `size` bytes.
/// Reuses a pooled allocation when one is large enough; otherwise allocates.
/// Equivalent to `vec![0; size]`.
pub fn take_zeroed(size: usize) -> Vec<u8> {
    POOL_IS_USED.store(true, Ordering::Relaxed);

    match take_best_fit(size) {
        // Capacity is already enough, so resize zeroes without reallocating.
        Some(mut buffer) => {
            buffer.clear();
            buffer.resize(size, 0);
            buffer
        }
        None => vec![0; size],
    }
}

/// Empty buffer with at least `min_capacity` bytes of capacity.
/// Unlike `take_zeroed`, does not resize or zero; callers that grow it
/// (`Write::write_all`, `extend_from_slice`, `push`) write into already-faulted
/// pages.
pub fn take_with_capacity(min_capacity: usize) -> Vec<u8> {
    POOL_IS_USED.store(true, Ordering::Relaxed);

    // Recycled buffers are always cleared, so no `clear()` here.
    take_best_fit(min_capacity).unwrap_or_else(|| Vec::with_capacity(min_capacity))
}

/// Smallest pooled buffer with capacity >= `min_capacity`.
/// Best-fit so a small compressed take cannot steal a large decompressed
/// buffer.
fn take_best_fit(min_capacity: usize) -> Option<Vec<u8>> {
    let mut pool = lock_pool();
    let index = pool
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.capacity() >= min_capacity)
        .min_by_key(|(_, buffer)| buffer.capacity())
        .map(|(index, _)| index)?;
    Some(pool.swap_remove(index))
}

/// Offer a finished buffer back to the pool instead of dropping it.
/// Typical call sites: `block.data` at the end of `read_block`, or a compressed
/// chunk once its bytes are written. Oversized buffers are dropped as usual.
pub fn recycle(mut buffer: Vec<u8>) {
    if buffer.capacity() == 0 || !POOL_IS_USED.load(Ordering::Relaxed) {
        return;
    }

    let mut pool = lock_pool();
    let pooled_bytes: usize = pool.iter().map(Vec::capacity).sum();

    if pool.len() < MAX_POOLED_BUFFERS && pooled_bytes + buffer.capacity() <= MAX_POOLED_BYTES {
        buffer.clear();
        pool.push(buffer);
    }
}

/// Drop every retained buffer. Use after decoding to return memory to the
/// allocator.
pub fn release() {
    lock_pool().clear();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn recycles_zeroed_buffers_within_bounds() {
        release();
        POOL_IS_USED.store(false, Ordering::Relaxed);

        recycle(vec![0; 1024]);
        assert_eq!(lock_pool().len(), 0, "unused pool must not retain buffers");

        let mut buffer = take_zeroed(1024);
        assert_eq!(buffer.len(), 1024);
        assert!(buffer.iter().all(|&byte| byte == 0));

        buffer.fill(0xAB);
        let address = buffer.as_ptr() as usize;
        recycle(buffer);

        let smaller = take_zeroed(512);
        assert_eq!(smaller.len(), 512);
        assert!(smaller.iter().all(|&byte| byte == 0), "reused buffer was not zeroed");
        assert_eq!(smaller.as_ptr() as usize, address, "buffer was not reused");
        drop(smaller);

        release();
        for _ in 0..MAX_POOLED_BUFFERS * 2 {
            recycle(vec![0; 1024]);
        }
        assert_eq!(lock_pool().len(), MAX_POOLED_BUFFERS);

        release();
        recycle(vec![0; MAX_POOLED_BYTES + 1]);
        assert_eq!(lock_pool().len(), 0, "buffer over the byte limit must not be retained");

        release();
    }

    #[test]
    fn take_prefers_smallest_sufficient_buffer() {
        release();
        POOL_IS_USED.store(true, Ordering::Relaxed);

        let large = vec![0u8; 64 * 1024];
        let small = vec![0u8; 1024];
        let large_addr = large.as_ptr() as usize;
        let small_addr = small.as_ptr() as usize;
        recycle(large);
        recycle(small);

        let taken_small = take_with_capacity(512);
        assert_eq!(taken_small.as_ptr() as usize, small_addr, "small take stole the large buffer");
        assert!(taken_small.capacity() < 64 * 1024);

        let taken_large = take_zeroed(32 * 1024);
        assert_eq!(taken_large.as_ptr() as usize, large_addr, "large take missed the large buffer");
        assert_eq!(taken_large.len(), 32 * 1024);
        assert!(taken_large.iter().all(|&byte| byte == 0));

        release();
    }
}
