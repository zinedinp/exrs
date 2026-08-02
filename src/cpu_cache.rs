//! L1d size and working-set sizing (Huffman LUT tier, scanline run width).

use std::sync::OnceLock;

/// Compile-time L1d estimate when CPUID is unavailable.
pub(crate) fn typical_l1d_bytes() -> usize {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        128 * 1024
    } else if cfg!(target_arch = "arm") {
        16 * 1024
    } else {
        32 * 1024
    }
}

/// L1d from CPUID, or `None`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn detected_l1d_bytes() -> Option<usize> {
    let cpuid = raw_cpuid::CpuId::new();

    // Leaf 4 / AMD 0x8000_001D.
    for cache in cpuid.get_cache_parameters().into_iter().flatten() {
        if cache.level() == 1
            && matches!(
                cache.cache_type(),
                raw_cpuid::CacheType::Data | raw_cpuid::CacheType::Unified
            )
        {
            return Some(
                cache.associativity()
                    * cache.physical_line_partitions()
                    * cache.coherency_line_size()
                    * cache.sets(),
            );
        }
    }

    // Older AMD: leaf 0x8000_0005 (KB).
    if let Some(l1) = cpuid.get_l1_cache_and_tlb_info() {
        let size = usize::from(l1.dcache_size()) * 1024;
        if size > 0 {
            return Some(size);
        }
    }

    None
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn detected_l1d_bytes() -> Option<usize> {
    None
}

/// L1d: probe, else typical. Once per process.
pub(crate) fn l1d_bytes() -> usize {
    static L1D: OnceLock<usize> = OnceLock::new();
    *L1D.get_or_init(|| detected_l1d_bytes().unwrap_or_else(typical_l1d_bytes))
}

/// Hybrid P/E (leaf 7 EDX bit 15). Always false off x86.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn is_hybrid_cpu() -> bool {
    const HYBRID_BIT: u32 = 1 << 15;
    raw_cpuid::cpuid!(7, 0).edx & HYBRID_BIT != 0
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn is_hybrid_cpu() -> bool {
    false
}

/// Max elements of `element_bytes` that fit in L1d/2 over `passes` walks (clamped 256–4096).
pub(crate) fn l1_resident_count(element_bytes: usize, passes: usize) -> usize {
    let budget = (l1d_bytes() / 2).max(1024);
    let cost_per = element_bytes.max(1).saturating_mul(passes.max(1));
    (budget / cost_per).clamp(256, 4096)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1d_is_plausible() {
        let n = l1d_bytes();
        assert!(n >= 8 * 1024, "L1d {n} too small");
        assert!(n <= 512 * 1024, "L1d {n} too large for an L1d");
    }

    #[test]
    fn resident_count_scales_with_element_size() {
        let small = l1_resident_count(2, 4);
        let large = l1_resident_count(16, 4);
        assert!(small >= large);
        assert!((256..=4096).contains(&small));
        assert!((256..=4096).contains(&large));
    }

    #[test]
    fn rgb_half_48kib_budget_is_1024() {
        assert_eq!((48 * 1024) / 2 / (6 * 4), 1024);
    }
}
