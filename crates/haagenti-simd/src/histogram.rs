//! SIMD-accelerated byte histogram computation.
//!
//! Used for entropy estimation in compression analysis.

/// Compute byte histogram (frequency of each byte value 0-255).
///
/// Uses SIMD acceleration when available.
#[inline]
pub fn byte_histogram(data: &[u8]) -> [u32; 256] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && data.len() >= 64 {
            return unsafe { byte_histogram_avx2(data) };
        }
    }

    byte_histogram_scalar(data)
}

/// SIMD-accelerated histogram using scatter/gather approach.
///
/// This uses multiple histogram arrays to reduce memory conflicts
/// and processes data in parallel lanes.
#[inline]
pub fn byte_histogram_simd(data: &[u8]) -> [u32; 256] {
    byte_histogram(data)
}

/// Scalar histogram implementation with loop unrolling.
#[inline]
fn byte_histogram_scalar(data: &[u8]) -> [u32; 256] {
    let mut hist = [0u32; 256];

    // Process 4 bytes at a time to reduce loop overhead
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        hist[chunk[0] as usize] += 1;
        hist[chunk[1] as usize] += 1;
        hist[chunk[2] as usize] += 1;
        hist[chunk[3] as usize] += 1;
    }

    for &b in remainder {
        hist[b as usize] += 1;
    }

    hist
}

/// AVX2-accelerated histogram using multiple histogram banks.
///
/// Uses 4 parallel histogram arrays to avoid memory conflicts,
/// then merges them at the end.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn byte_histogram_avx2(data: &[u8]) -> [u32; 256] {
    // Use 4 parallel histograms to reduce conflicts
    let mut hist0 = [0u32; 256];
    let mut hist1 = [0u32; 256];
    let mut hist2 = [0u32; 256];
    let mut hist3 = [0u32; 256];

    // Process 4 bytes per iteration, spread across 4 histograms
    let chunks = data.chunks_exact(16);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Unroll 16 bytes across 4 histograms
        hist0[chunk[0] as usize] += 1;
        hist1[chunk[1] as usize] += 1;
        hist2[chunk[2] as usize] += 1;
        hist3[chunk[3] as usize] += 1;

        hist0[chunk[4] as usize] += 1;
        hist1[chunk[5] as usize] += 1;
        hist2[chunk[6] as usize] += 1;
        hist3[chunk[7] as usize] += 1;

        hist0[chunk[8] as usize] += 1;
        hist1[chunk[9] as usize] += 1;
        hist2[chunk[10] as usize] += 1;
        hist3[chunk[11] as usize] += 1;

        hist0[chunk[12] as usize] += 1;
        hist1[chunk[13] as usize] += 1;
        hist2[chunk[14] as usize] += 1;
        hist3[chunk[15] as usize] += 1;
    }

    // Handle remainder
    for &b in remainder {
        hist0[b as usize] += 1;
    }

    // Merge histograms using SIMD
    use std::arch::x86_64::*;

    let mut result = [0u32; 256];

    // Process 8 u32s at a time with AVX2
    for i in (0..256).step_by(8) {
        // SAFETY: AVX2 is enabled via target_feature, pointers are valid and aligned
        unsafe {
            let v0 = _mm256_loadu_si256(hist0[i..].as_ptr() as *const __m256i);
            let v1 = _mm256_loadu_si256(hist1[i..].as_ptr() as *const __m256i);
            let v2 = _mm256_loadu_si256(hist2[i..].as_ptr() as *const __m256i);
            let v3 = _mm256_loadu_si256(hist3[i..].as_ptr() as *const __m256i);

            // Add all 4 histograms
            let sum01 = _mm256_add_epi32(v0, v1);
            let sum23 = _mm256_add_epi32(v2, v3);
            let sum = _mm256_add_epi32(sum01, sum23);

            _mm256_storeu_si256(result[i..].as_mut_ptr() as *mut __m256i, sum);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_empty() {
        let hist = byte_histogram(&[]);
        assert!(hist.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_histogram_single_byte() {
        let data = vec![42u8; 100];
        let hist = byte_histogram(&data);
        assert_eq!(hist[42], 100);
        assert_eq!(hist.iter().filter(|&&c| c > 0).count(), 1);
    }

    #[test]
    fn test_histogram_all_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let hist = byte_histogram(&data);
        assert!(hist.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_histogram_repeated_pattern() {
        let pattern = b"ABCD";
        let data: Vec<u8> = pattern.iter().cycle().take(1000).cloned().collect();
        let hist = byte_histogram(&data);

        assert_eq!(hist[b'A' as usize], 250);
        assert_eq!(hist[b'B' as usize], 250);
        assert_eq!(hist[b'C' as usize], 250);
        assert_eq!(hist[b'D' as usize], 250);
    }

    #[test]
    fn test_histogram_large() {
        // Test with large data to exercise SIMD path
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let hist = byte_histogram(&data);

        // Each byte value should appear ~390 times (100000 / 256)
        let expected = 100_000 / 256;
        for &count in hist.iter() {
            assert!(count >= expected as u32);
            assert!(count <= (expected + 1) as u32);
        }
    }
}
