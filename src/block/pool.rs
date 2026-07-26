//! Reuse of the byte buffers that decompressed blocks are handed out in, and of
//! the compressed-chunk output buffers produced while writing.
//!
//! Every decompressed chunk gets a fresh output buffer, which is dropped again
//! as soon as the reader has copied the samples out of it. On a 4k DWA image
//! that is 128 MiB of freshly mapped pages per decoded frame, and the kernel
//! faults in every one of those pages on the first write -- ~11 ms per frame,
//! charged to whichever decode stage happens to touch a page first, plus the
//! address-translation misses that come with never seeing the same page twice.
//! OpenEXR never pays this: it keeps one unpacked buffer alive for the whole
//! file.
//!
//! This module is how the same effect is available here without changing the
//! type of `UncompressedBlock::data`: a decompressor takes its output buffer
//! from the pool, and the reader that consumes the block hands the buffer back
//! once it is done with it. Everything is a plain `Vec<u8>`, so a reader that
//! does not recycle simply drops the buffer, as before.
//!
//! The write side has the same shape in reverse: every compressed chunk (and,
//! for multi-section formats like DWA, every section inside it) is built into
//! its own freshly allocated `Vec<u8>` that lives only until its bytes have
//! been copied into the file or into the next buffer up the chain, then it is
//! dropped. `take_with_capacity`/`recycle` let compressors pull a
//! previously-used buffer's pages back instead of faulting in new ones for
//! every chunk.
//!
//! Buffers are only ever retained once something has actually asked for
//! one, so an image whose compression method does not use the pool never makes
//! it hold on to memory.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

/// One buffer per thread of the default decompression thread pool.
/// `ParallelBlockDecompressor` keeps at most `max_threads` blocks in flight, so
/// this is enough to serve every decompression thread out of the pool.
const MAX_POOLED_BUFFERS: usize = 8;

/// Upper bound on the memory the pool retains, whichever limit is hit first.
/// Large-tile images give up some reuse rather than holding on to more.
const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024;

static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static POOL_IS_USED: AtomicBool = AtomicBool::new(false);

fn lock_pool() -> MutexGuard<'static, Vec<Vec<u8>>> {
    // a panic while the pool is locked must not poison it for the rest of the
    // process: the worst a half-updated pool can cost is a missed reuse
    POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A zeroed buffer of exactly `size` bytes, reusing a previously recycled
/// allocation where one is large enough, and allocating otherwise.
/// Reused buffers are zeroed again, so the result is indistinguishable from
/// `vec![0; size]`.
pub fn take_zeroed(size: usize) -> Vec<u8> {
    POOL_IS_USED.store(true, Ordering::Relaxed);

    let recycled = {
        let mut pool = lock_pool();
        pool.iter()
            .position(|buffer| buffer.capacity() >= size)
            .map(|index| pool.swap_remove(index))
    };

    match recycled {
        // capacity is known to be sufficient, so this zeroes without reallocating
        Some(mut buffer) => {
            buffer.clear();
            buffer.resize(size, 0);
            buffer
        }
        None => vec![0; size],
    }
}

/// An empty buffer with at least `min_capacity` bytes of capacity, reusing a
/// previously recycled allocation where one is large enough, and allocating
/// otherwise. Unlike `take_zeroed`, the buffer is not resized or zeroed, so
/// callers that grow it themselves (`Write::write_all`, `extend_from_slice`,
/// `push`) start writing into pages that were already faulted
pub fn take_with_capacity(min_capacity: usize) -> Vec<u8> {
    POOL_IS_USED.store(true, Ordering::Relaxed);

    let recycled = {
        let mut pool = lock_pool();
        pool.iter()
            .position(|buffer| buffer.capacity() >= min_capacity)
            .map(|index| pool.swap_remove(index))
    };

    // buffers are always stored cleared (`recycle`), so no `clear()` needed here
    recycled.unwrap_or_else(|| Vec::with_capacity(min_capacity))
}

/// Offer a block's buffer to the next decompression or compression that
/// needs one, instead of dropping it. Call this only with a buffer that is no
/// longer referenced, typically `block.data` at the end of a `read_block`
/// implementation, or a compressed chunk's bytes once they have been written
/// out. Buffers beyond what the pool retains are dropped as usual.
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

/// Drop every buffer the pool currently holds. Only useful to return the
/// retained memory to the allocator after decoding is done; decoding more
/// images afterwards simply starts filling the pool again.
pub fn release() {
    lock_pool().clear();
}

#[cfg(test)]
mod test {
    use super::*;

    /// A recycled buffer must come back zeroed and with the exact length that
    /// was asked for, no matter what the previous user left in it. Also covers
    /// the retention bounds, because the pool is global state and separate
    /// tests would race each other.
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
}
