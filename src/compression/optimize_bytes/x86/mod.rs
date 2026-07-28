//! Runtime x86 SIMD dispatch for ZIP/RLE byte reconstruct.
//!
//! Production order (stage A/B on Zen 5 / AVX-512):
//! `V4` AVX-512 lane -> `V3` AVX2 lane + SSE remainder -> `V2` SSE -> scalar.
//!
//! Layout mirrors DWA DCT (`discrete_cosine_transform/x86/`): one file per
//! tier, shared dispatch here. Modules are `doc(hidden)`-public so stage
//! benches can A/B kernels without going through production dispatch.

use std::sync::OnceLock;

use pulp::x86::{V2, V3, V4};

use crate::compression::simd_tier::x86::{v2 as tier_v2, v3 as tier_v3, v4 as tier_v4};

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

/// The best tier this CPU supports, resolved once (see [`resolved_tier`]).
#[derive(Clone, Copy)]
enum Tier {
    Avx512(V4),
    Avx2(V3),
    Sse(V2),
    Scalar,
}

/// CPU features don't change at runtime, but every call to `differences_to_samples`
/// (once per DWA DC section / ZIP·RLE chunk, hundreds of times per image) was
/// re-running the full `V4 -> V3 -> V2::try_new()` cascade. Resolve it once per
/// process instead and reuse the token: `V4: Deref<Target = V3>` (and `V3 -> V2`),
/// so the cached top tier alone gives every lower tier for free, no extra
/// `try_new`/CPUID calls at all
fn resolved_tier() -> Tier {
    static TIER: OnceLock<Tier> = OnceLock::new();
    *TIER.get_or_init(|| {
        tier_v4()
            .map(Tier::Avx512)
            .or_else(|| tier_v3().map(Tier::Avx2))
            .or_else(|| tier_v2().map(Tier::Sse))
            .unwrap_or(Tier::Scalar)
    })
}

/// Dispatch to the best available tier (resolved once, see [`resolved_tier`]).
/// AVX-512 is only used for buffers >= 64B; smaller buffers on an AVX-512 host
/// drop to AVX2 via the same cached token (`*v4` deref, no second `try_new`).
/// Returns `true` if a SIMD path ran.
#[inline]
pub(super) fn try_differences_to_samples(buffer: &mut [u8]) -> bool {
    match resolved_tier() {
        Tier::Avx512(v4) => {
            if buffer.len() >= AVX512_MIN_LEN {
                avx512::differences_to_samples(v4, buffer);
            } else {
                avx2::differences_to_samples(*v4, buffer);
            }
            true
        }
        Tier::Avx2(v3) => {
            avx2::differences_to_samples(v3, buffer);
            true
        }
        Tier::Sse(v2) => {
            sse::differences_to_samples(v2, buffer);
            true
        }
        Tier::Scalar => false,
    }
}
