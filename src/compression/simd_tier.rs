//! Optional runtime cap on which x86 SIMD tier the compression codecs may use.

/// Rungs of the x86 SIMD ladder, ordered so that `<=` means "at most this tier".
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    /// No SIMD tokens at all; codecs take their scalar/autovectorized fallbacks.
    Scalar = 0,
    /// 128-bit SSE tiers (`V1`, `V2`).
    Sse = 1,
    /// 256-bit AVX2 tier (`V3`) plus `F16c`, which exrs only ever uses alongside `V3`.
    Avx2 = 2,
    /// 512-bit AVX-512 tier (`V4`) and everything below it.
    Avx512 = 3,
}

#[cfg(not(feature = "simd-tier-env"))]
#[inline(always)]
pub fn cap() -> Tier {
    Tier::Avx512
}

/// The cap in force for this process, parsed once from `EXRS_SIMD_TIER`.
/// An unset or unrecognized value means "no cap".
#[cfg(feature = "simd-tier-env")]
pub fn cap() -> Tier {
    use std::sync::OnceLock;

    static CAP: OnceLock<Tier> = OnceLock::new();
    *CAP.get_or_init(|| match std::env::var("EXRS_SIMD_TIER") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "scalar" | "none" | "off" => Tier::Scalar,
            "sse" | "sse2" | "v1" | "v2" => Tier::Sse,
            "avx" | "avx2" | "v3" => Tier::Avx2,
            "avx512" | "v4" | "max" | "" => Tier::Avx512,
            other => {
                eprintln!(
                    "exrs: ignoring unknown EXRS_SIMD_TIER={:?} (expected scalar|sse|avx2|avx512)",
                    other
                );
                Tier::Avx512
            }
        },
        Err(_) => Tier::Avx512,
    })
}

/// Human-readable name of the tier actually in force, for benchmark reports.
pub fn cap_name() -> &'static str {
    match cap() {
        Tier::Scalar => "scalar",
        Tier::Sse => "sse",
        Tier::Avx2 => "avx2",
        Tier::Avx512 => "avx512",
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86 {
    pub(crate) mod miraculix_x86 {
        use miraculix::x86::detect_features;
        use miraculix::x86::ops::avx::avx::Avx;
        use miraculix::x86::ops::avx::avx2::Avx2;
        use miraculix::x86::ops::avx::f16c::F16c;
        use miraculix::x86::ops::avx512::avx512bw::Avx512Bw;
        use miraculix::x86::ops::avx512::avx512dq::Avx512Dq;
        use miraculix::x86::ops::avx512::avx512f::Avx512f;
        use miraculix::x86::ops::sse::sse::Sse;
        use miraculix::x86::ops::sse::sse2::Sse2;
        use miraculix::x86::ops::sse::sse41::Sse41;
        use miraculix::x86::ops::sse::ssse3::Ssse3;

        use super::super::{Tier, cap};

        /// Base SSE token (f32 ops only), unless the process is capped below
        /// the SSE tier. `color_space_conversion`'s and
        /// `discrete_cosine_transform`'s SSE-tier kernels only ever touch
        /// `f32` arithmetic, so they need this rather than [`sse2`].
        #[inline(always)]
        pub(crate) fn sse() -> Option<Sse> {
            if cap() >= Tier::Sse {
                Sse::from_features(detect_features())
            } else {
                None
            }
        }

        /// SSE2 baseline token, unless the process is capped below the SSE
        /// tier. Pairs with [`ssse3`] for the SSE-tier kernels.
        #[inline(always)]
        pub(crate) fn sse2() -> Option<Sse2> {
            if cap() >= Tier::Sse {
                Sse2::from_features(detect_features())
            } else {
                None
            }
        }

        /// SSSE3 token, unless the process is capped below the SSE tier.
        #[inline(always)]
        pub(crate) fn ssse3() -> Option<Ssse3> {
            if cap() >= Tier::Sse {
                Ssse3::from_features(detect_features())
            } else {
                None
            }
        }

        /// SSE4.1 token, capped at the AVX2 tier rather than its own lower
        /// `Tier::Sse` rung: `lossy_dct`'s zigzag-unpack kernel only ever
        /// reaches for this alongside an AVX2-tier ([`f16c`]) kernel, same
        /// rationale already established for [`avx`].
        #[inline(always)]
        pub(crate) fn sse41() -> Option<Sse41> {
            if cap() >= Tier::Avx2 {
                Sse41::from_features(detect_features())
            } else {
                None
            }
        }

        /// F16C token, unless the process is capped below the AVX2 tier.
        /// Capped with AVX2 rather than its own lower rung for the same
        /// reason [`avx`]/[`sse41`] are: exrs never uses F16C without an
        /// accompanying AVX2-tier kernel.
        #[inline(always)]
        pub(crate) fn f16c() -> Option<F16c> {
            if cap() >= Tier::Avx2 {
                F16c::from_features(detect_features())
            } else {
                None
            }
        }

        /// AVX2 token, unless the process is capped below the AVX2 tier.
        #[inline(always)]
        pub(crate) fn avx2() -> Option<Avx2> {
            if cap() >= Tier::Avx2 {
                Avx2::from_features(detect_features())
            } else {
                None
            }
        }

        /// Base AVX token (f32 ops only), capped at the AVX2 tier rather
        /// than its own lower `Tier::Sse` rung: exrs only ever reaches for
        /// this alongside an AVX2-tier kernel (same rationale already
        /// established for [`super::f16c`]).
        #[inline(always)]
        pub(crate) fn avx() -> Option<Avx> {
            if cap() >= Tier::Avx2 {
                Avx::from_features(detect_features())
            } else {
                None
            }
        }

        /// AVX-512F token, unless the process is capped below the AVX-512
        /// tier. Pairs with [`avx512bw`]/[`avx512dq`] for the AVX-512-tier
        /// kernels.
        #[inline(always)]
        pub(crate) fn avx512f() -> Option<Avx512f> {
            if cap() >= Tier::Avx512 {
                Avx512f::from_features(detect_features())
            } else {
                None
            }
        }

        /// AVX-512BW token, unless the process is capped below the AVX-512
        /// tier.
        #[inline(always)]
        pub(crate) fn avx512bw() -> Option<Avx512Bw> {
            if cap() >= Tier::Avx512 {
                Avx512Bw::from_features(detect_features())
            } else {
                None
            }
        }

        /// AVX-512DQ token, unless the process is capped below the AVX-512
        /// tier. `discrete_cosine_transform`'s AVX-512 tier needs this for
        /// its `extract_f32x8_from_x16`/`insert_f32x8_into_x16` transpose
        /// recombine step (`vextractf32x8`/`vinsertf32x8`, DQ-only).
        #[inline(always)]
        pub(crate) fn avx512dq() -> Option<Avx512Dq> {
            if cap() >= Tier::Avx512 {
                Avx512Dq::from_features(detect_features())
            } else {
                None
            }
        }
    }
}
