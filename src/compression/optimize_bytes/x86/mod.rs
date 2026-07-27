//! Runtime x86 SIMD dispatch for ZIP/RLE byte reconstruct.
//!
//! Production order (stage A/B on Zen 5 / AVX-512):
//! `V4` AVX-512 lane -> `V3` AVX2 lane + SSE remainder -> `V2` SSE -> scalar.
//!
//! Layout mirrors DWA DCT (`discrete_cosine_transform/x86/`): one file per
//! tier, shared dispatch here. Modules are `doc(hidden)`-public so stage
//! benches can A/B kernels without going through production dispatch.

use pulp::x86::{V2, V3, V4};

// public only for benchmarking / correctness tests
#[doc(hidden)]
pub mod avx2;

// public only for benchmarking / correctness tests
#[doc(hidden)]
pub mod avx512;

// public only for benchmarking / correctness tests
#[doc(hidden)]
pub mod sse;

/// Below one full 64-byte AVX-512 lane, `avx512::differences_to_samples`
/// never fills a chunk: it still pays for the pre-bias store, the empty
/// `n_chunks` loop, and the undo-bias-then-delegate-to-AVX2 remainder path
/// (`avx512.rs`'s `finish_hierarchical`, `done == 0` branch). Measured a
/// reproducible ~0.86x vs scalar at n=48 from that overhead alone -- below
/// this threshold, go straight to AVX2 (whose own SSE tail is what AVX-512
/// would have delegated to anyway, minus the wasted bias round-trip).
const AVX512_MIN_LEN: usize = 64;

/// Try AVX-512 (buffers >= 64B only), then AVX2, then SSE reconstruct.
/// Returns `true` if a SIMD path ran.
#[inline]
pub(super) fn try_differences_to_samples(buffer: &mut [u8]) -> bool {
    if buffer.len() >= AVX512_MIN_LEN {
        if let Some(v4) = V4::try_new() {
            avx512::differences_to_samples(v4, buffer);
            return true;
        }
    }
    if let Some(v3) = V3::try_new() {
        avx2::differences_to_samples(v3, buffer);
        return true;
    }
    if let Some(v2) = V2::try_new() {
        sse::differences_to_samples(v2, buffer);
        return true;
    }
    false
}
