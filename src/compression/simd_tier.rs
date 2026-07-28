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
    use super::{cap, Tier};
    use pulp::core_arch::x86::F16c;
    use pulp::x86::{V1, V2, V3, V4};

    /// `V1::try_new()`, unless the process is capped below the SSE tier.
    #[inline(always)]
    pub(crate) fn v1() -> Option<V1> {
        if cap() >= Tier::Sse {
            V1::try_new()
        } else {
            None
        }
    }

    /// `V2::try_new()`, unless the process is capped below the SSE tier.
    #[inline(always)]
    pub(crate) fn v2() -> Option<V2> {
        if cap() >= Tier::Sse {
            V2::try_new()
        } else {
            None
        }
    }

    /// `V3::try_new()`, unless the process is capped below the AVX2 tier.
    #[inline(always)]
    pub(crate) fn v3() -> Option<V3> {
        if cap() >= Tier::Avx2 {
            V3::try_new()
        } else {
            None
        }
    }

    /// `V4::try_new()`, unless the process is capped below the AVX-512 tier.
    #[inline(always)]
    pub(crate) fn v4() -> Option<V4> {
        if cap() >= Tier::Avx512 {
            V4::try_new()
        } else {
            None
        }
    }

    /// `F16c::try_new()`. Capped with `V3` rather than on its own: exrs never
    /// uses F16C without an accompanying `V3` token, so letting it survive into
    /// the SSE tier would describe a configuration the crate cannot actually run.
    #[inline(always)]
    pub(crate) fn f16c() -> Option<F16c> {
        if cap() >= Tier::Avx2 {
            F16c::try_new()
        } else {
            None
        }
    }
}
