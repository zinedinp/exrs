//! Runtime x86 SIMD dispatch for ZIP/RLE byte reconstruct.
//!
//! Production order (stage A/B on Zen 5 / AVX-512):
//! AVX-512 lane -> AVX2 lane + SSE remainder -> SSE -> scalar.
//!
//! Layout mirrors DWA DCT (`discrete_cosine_transform/x86/`): one file per
//! tier, shared dispatch here. Modules are `doc(hidden)`-public so stage
//! benches can A/B kernels without going through production dispatch.

use std::sync::OnceLock;

use miraculix::x86::ops::avx::avx2::Avx2;
use miraculix::x86::ops::avx512::avx512bw::Avx512Bw;
use miraculix::x86::ops::avx512::avx512f::Avx512f;
use miraculix::x86::ops::sse::sse2::Sse2;
use miraculix::x86::ops::sse::ssse3::Ssse3;

use crate::compression::simd_tier::x86::miraculix_x86;

// public only for benchmarking / correctness tests
#[doc(hidden)]
pub mod avx2;
use self::avx2 as avx2_dispatch;

// public only for benchmarking / correctness tests. Named `avx512bw`:
// `Avx512Bw` is the most restrictive of this file's 5 tokens (byte/word
// integer ops need BW, not just base `Avx512f`).
#[doc(hidden)]
pub mod avx512bw;

// public only for benchmarking / correctness tests. Named `ssse3`, not
// `sse`: this file's carry-propagation shuffle needs `Ssse3`, not just the
// base `Sse2` token.
#[doc(hidden)]
pub mod ssse3;

/// Below one full 64-byte AVX-512 lane, `avx512bw::differences_to_samples`
/// never fills a chunk: it still pays for the pre-bias store, the empty
/// `n_chunks` loop, and the undo-bias-then-delegate-to-AVX2 remainder path
/// (`avx512.rs`'s `finish_hierarchical`, `done == 0` branch). Measured a
/// reproducible ~0.86x vs scalar at n=48 from that overhead alone -- below
/// this threshold, go straight to AVX2 (whose own SSE tail is what AVX-512
/// would have delegated to anyway, minus the wasted bias round-trip).
const AVX512_MIN_LEN: usize = 64;

/// The best tier this CPU supports, resolved once (see [`resolved_tier`]).
/// Each rung carries every token its own kernel (and its fallbacks) need -
/// miraculix hands out one token per CPU feature rather than pulp's one
/// bundled struct per tier, and there is no `Deref` chain between them.
#[derive(Clone, Copy)]
enum Tier {
    Avx512 {
        f: Avx512f,
        bw: Avx512Bw,
        avx2: Avx2,
        sse2: Sse2,
        ssse3: Ssse3,
    },
    Avx2 {
        avx2: Avx2,
        sse2: Sse2,
        ssse3: Ssse3,
    },
    Sse {
        sse2: Sse2,
        ssse3: Ssse3,
    },
    Scalar,
}

/// CPU features don't change at runtime, but every call to `differences_to_samples`
/// (once per DWA DC section / ZIP·RLE chunk, hundreds of times per image) would
/// re-run the full tier cascade otherwise. Resolve it once per process instead
/// and reuse the tokens (each `from_features` call is just a cached-bitset
/// check, not a fresh CPUID probe, but the tier match itself is worth caching
/// too).
fn resolved_tier() -> Tier {
    static TIER: OnceLock<Tier> = OnceLock::new();
    *TIER.get_or_init(|| {
        if let (Some(f), Some(bw), Some(avx2), Some(sse2), Some(ssse3)) = (
            miraculix_x86::avx512f(),
            miraculix_x86::avx512bw(),
            miraculix_x86::avx2(),
            miraculix_x86::sse2(),
            miraculix_x86::ssse3(),
        ) {
            return Tier::Avx512 {
                f,
                bw,
                avx2,
                sse2,
                ssse3,
            };
        }
        if let (Some(avx2), Some(sse2), Some(ssse3)) =
            (miraculix_x86::avx2(), miraculix_x86::sse2(), miraculix_x86::ssse3())
        {
            return Tier::Avx2 {
                avx2,
                sse2,
                ssse3,
            };
        }
        if let (Some(sse2), Some(ssse3)) = (miraculix_x86::sse2(), miraculix_x86::ssse3()) {
            return Tier::Sse {
                sse2,
                ssse3,
            };
        }
        Tier::Scalar
    })
}

/// Dispatch to the best available tier (resolved once, see [`resolved_tier`]).
/// AVX-512 is only used for buffers >= 64B; smaller buffers on an AVX-512 host
/// drop straight to AVX2 (the cached tokens, no second feature probe).
/// Returns `true` if a SIMD path ran.
#[inline]
pub(super) fn try_differences_to_samples(buffer: &mut [u8]) -> bool {
    match resolved_tier() {
        Tier::Avx512 {
            f,
            bw,
            avx2,
            sse2,
            ssse3,
        } => {
            if buffer.len() >= AVX512_MIN_LEN {
                avx512bw::differences_to_samples(f, bw, avx2, sse2, ssse3, buffer);
            } else {
                avx2_dispatch::differences_to_samples(avx2, sse2, ssse3, buffer);
            }
            true
        }
        Tier::Avx2 {
            avx2,
            sse2,
            ssse3,
        } => {
            avx2_dispatch::differences_to_samples(avx2, sse2, ssse3, buffer);
            true
        }
        Tier::Sse {
            sse2,
            ssse3,
        } => {
            self::ssse3::differences_to_samples(sse2, ssse3, buffer);
            true
        }
        Tier::Scalar => false,
    }
}
