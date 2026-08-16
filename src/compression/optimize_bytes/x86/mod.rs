//! Runtime x86 SIMD dispatch for ZIP/RLE byte reconstruct.
//! Production order: AVX-512 lane -> AVX2 lane + SSE rem -> SSE -> scalar.

use std::sync::OnceLock;

use miraculix::x86::ops::avx::avx2::Avx2;
use miraculix::x86::ops::avx512::avx512bw::Avx512Bw;
use miraculix::x86::ops::avx512::avx512f::Avx512f;
use miraculix::x86::ops::sse::sse2::Sse2;
use miraculix::x86::ops::sse::ssse3::Ssse3;

use crate::compression::simd_tier::x86::miraculix_x86;

// A/B benches and correctness tests.
#[doc(hidden)]
pub mod avx2;
use self::avx2 as avx2_dispatch;

// Named `avx512bw`: BW is the most restrictive of this file's 5 tokens
// (byte/word integer ops need BW, not just base `Avx512f`).
#[doc(hidden)]
pub mod avx512bw;

// Named `ssse3`, not `sse`: carry-propagation shuffle needs `Ssse3`, not
// just the base `Sse2` token.
#[doc(hidden)]
pub mod ssse3;

/// Skip AVX-512 below one full 64B lane: empty-loop + undo-bias overhead
/// measured ~0.86x vs scalar at n=48. Drop to AVX2 (same hierarchical tail).
const AVX512_MIN_LEN: usize = 64;

/// Best supported tier, tokens included. One miraculix token per feature
/// (no pulp-style bundled tier struct / `Deref` chain).
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

/// Resolve once per process: features are static; cascade runs once per
/// DWA DC / ZIP/RLE chunk otherwise.
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

/// Best tier once ([`resolved_tier`]). AVX-512 only for `len >= 64`; smaller
/// buffers on AVX-512 hosts drop to cached AVX2 tokens. `true` if SIMD ran.
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
