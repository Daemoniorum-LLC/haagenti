//! # Haagenti SIMD
//!
//! SIMD-accelerated primitives for compression algorithms.
//!
//! Provides optimized implementations of common compression operations
//! using platform-specific SIMD instructions.
//!
//! ## Supported Architectures
//!
//! - **x86_64**: SSE4.2, AVX2, AVX-512
//! - **aarch64**: NEON, SVE
//! - **Fallback**: Scalar implementation
//!
//! ## Example
//!
//! ```ignore
//! use haagenti_simd::{detect_simd, SimdLevel};
//!
//! match detect_simd() {
//!     SimdLevel::Avx512 => println!("Using AVX-512"),
//!     SimdLevel::Avx2 => println!("Using AVX2"),
//!     SimdLevel::Neon => println!("Using NEON"),
//!     SimdLevel::None => println!("Scalar fallback"),
//! }
//! ```

mod hash;
mod histogram;
mod match_finder;
mod memops;

pub use hash::{hash4_scalar, hash4x4, hash4x8};
pub use histogram::{byte_histogram, byte_histogram_simd};
pub use match_finder::{find_match_length, find_match_length_safe};
pub use memops::{copy_match, copy_within_extend, fill_repeat};

/// SIMD feature level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SimdLevel {
    /// No SIMD available.
    #[default]
    None,
    /// SSE4.2 (x86_64).
    Sse42,
    /// AVX2 (x86_64).
    Avx2,
    /// AVX-512 (x86_64).
    Avx512,
    /// NEON (aarch64).
    Neon,
    /// SVE (aarch64).
    Sve,
}

impl SimdLevel {
    /// Get string representation.
    pub fn as_str(self) -> &'static str {
        match self {
            SimdLevel::None => "none",
            SimdLevel::Sse42 => "sse4.2",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Avx512 => "avx512",
            SimdLevel::Neon => "neon",
            SimdLevel::Sve => "sve",
        }
    }

    /// Get vector width in bytes.
    pub fn vector_width(self) -> usize {
        match self {
            SimdLevel::None => 1,
            SimdLevel::Sse42 => 16,
            SimdLevel::Avx2 => 32,
            SimdLevel::Avx512 => 64,
            SimdLevel::Neon => 16,
            SimdLevel::Sve => 64, // Variable, assume max
        }
    }

    /// Check if this level supports the given operation width efficiently.
    pub fn supports_width(self, bytes: usize) -> bool {
        bytes <= self.vector_width()
    }
}

/// Detect available SIMD level at runtime.
pub fn detect_simd() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
        if is_x86_feature_detected!("sse4.2") {
            return SimdLevel::Sse42;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return SimdLevel::Neon;
    }

    SimdLevel::None
}

/// Global SIMD level (cached on first call).
static SIMD_LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();

/// Get the current SIMD level (cached).
pub fn simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(detect_simd)
}

/// Check if AVX2 is available.
#[inline]
pub fn has_avx2() -> bool {
    simd_level() >= SimdLevel::Avx2
}

/// Check if AVX-512 is available.
#[inline]
pub fn has_avx512() -> bool {
    simd_level() >= SimdLevel::Avx512
}

/// Check if NEON is available.
#[inline]
pub fn has_neon() -> bool {
    simd_level() == SimdLevel::Neon || simd_level() == SimdLevel::Sve
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_simd() {
        let level = detect_simd();
        println!("Detected SIMD level: {:?} ({})", level, level.as_str());

        // Should always get some result
        assert!(level.as_str().len() > 0);
    }

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::None < SimdLevel::Sse42);
        assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
    }

    #[test]
    fn test_vector_width() {
        assert_eq!(SimdLevel::None.vector_width(), 1);
        assert_eq!(SimdLevel::Sse42.vector_width(), 16);
        assert_eq!(SimdLevel::Avx2.vector_width(), 32);
        assert_eq!(SimdLevel::Avx512.vector_width(), 64);
        assert_eq!(SimdLevel::Neon.vector_width(), 16);
    }

    #[test]
    fn test_cached_simd_level() {
        let level1 = simd_level();
        let level2 = simd_level();
        assert_eq!(level1, level2);
    }
}
