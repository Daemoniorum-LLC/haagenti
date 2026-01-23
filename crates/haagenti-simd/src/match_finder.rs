//! Match finding primitives for LZ algorithms.
//!
//! These functions are used to find how many bytes match between two
//! memory locations, which is critical for LZ77-style compression.

/// Find the length of matching bytes between two slices.
///
/// This is the core operation for LZ77 compression - given a match position
/// and current position, find how many bytes are identical.
///
/// # Safety
///
/// This function uses unsafe pointer operations for performance. The caller
/// must ensure that both slices have at least `max_len` bytes available.
///
/// # Arguments
///
/// * `src` - The source (match) position
/// * `cur` - The current position
/// * `max_len` - Maximum length to check
///
/// # Returns
///
/// Number of matching bytes, up to `max_len`.
#[inline]
pub fn find_match_length(src: &[u8], cur: &[u8], max_len: usize) -> usize {
    let len = src.len().min(cur.len()).min(max_len);

    if len == 0 {
        return 0;
    }

    // Use SIMD-accelerated comparison on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX-512 for 2x throughput vs AVX2
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && len >= 64
        {
            // Safety: we've checked that AVX-512 is available and len >= 64
            return unsafe { find_match_length_avx512(src, cur, len) };
        }

        if is_x86_feature_detected!("avx2") && len >= 32 {
            // Safety: we've checked that AVX2 is available and len >= 32
            return unsafe { find_match_length_avx2(src, cur, len) };
        }
    }

    // Fallback to optimized scalar implementation
    find_match_length_scalar(src, cur, len)
}

/// Safe version that doesn't use unsafe operations.
/// Slightly slower but guaranteed safe.
#[inline]
pub fn find_match_length_safe(src: &[u8], cur: &[u8], max_len: usize) -> usize {
    let len = src.len().min(cur.len()).min(max_len);
    src.iter()
        .zip(cur.iter())
        .take(len)
        .take_while(|(a, b)| a == b)
        .count()
}

/// Scalar implementation using word-at-a-time comparison.
#[inline]
fn find_match_length_scalar(src: &[u8], cur: &[u8], max_len: usize) -> usize {
    let mut matched = 0;

    // Compare 8 bytes at a time using u64
    while matched + 8 <= max_len {
        // Safety: we've verified length above
        let src_word = u64::from_le_bytes(src[matched..matched + 8].try_into().unwrap_or([0; 8]));
        let cur_word = u64::from_le_bytes(cur[matched..matched + 8].try_into().unwrap_or([0; 8]));

        let diff = src_word ^ cur_word;
        if diff != 0 {
            // Find first differing byte
            matched += (diff.trailing_zeros() / 8) as usize;
            return matched;
        }
        matched += 8;
    }

    // Compare remaining bytes one at a time
    while matched < max_len && src[matched] == cur[matched] {
        matched += 1;
    }

    matched
}

/// AVX-512 accelerated match length finding.
///
/// Processes 64 bytes per iteration, ~2x throughput vs AVX2 for long matches.
/// Falls back to AVX2 for the 32-63 byte tail, then scalar for remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn find_match_length_avx512(src: &[u8], cur: &[u8], max_len: usize) -> usize {
    use std::arch::x86_64::*;

    let mut matched = 0;

    // Process 64 bytes at a time with AVX-512
    unsafe {
        while matched + 64 <= max_len {
            let src_vec = _mm512_loadu_si512(src[matched..].as_ptr() as *const __m512i);
            let cur_vec = _mm512_loadu_si512(cur[matched..].as_ptr() as *const __m512i);

            // Compare for equality - returns 64-bit mask
            let mask = _mm512_cmpeq_epi8_mask(src_vec, cur_vec);

            if mask != 0xFFFFFFFFFFFFFFFF {
                // Found a mismatch - count trailing ones in the equality mask
                matched += mask.trailing_ones() as usize;
                return matched;
            }

            matched += 64;
        }

        // Handle 32-63 remaining bytes with AVX2
        if matched + 32 <= max_len {
            let src_vec = _mm256_loadu_si256(src[matched..].as_ptr() as *const __m256i);
            let cur_vec = _mm256_loadu_si256(cur[matched..].as_ptr() as *const __m256i);

            let cmp = _mm256_cmpeq_epi8(src_vec, cur_vec);
            let mask = _mm256_movemask_epi8(cmp) as u32;

            if mask != 0xFFFFFFFF {
                matched += (!mask).trailing_zeros() as usize;
                return matched;
            }
            matched += 32;
        }
    }

    // Handle remaining bytes with scalar code
    while matched < max_len && src[matched] == cur[matched] {
        matched += 1;
    }

    matched
}

/// AVX2-accelerated match length finding.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_match_length_avx2(src: &[u8], cur: &[u8], max_len: usize) -> usize {
    use std::arch::x86_64::*;

    let mut matched = 0;

    // Process 32 bytes at a time with AVX2
    unsafe {
        while matched + 32 <= max_len {
            let src_vec = _mm256_loadu_si256(src[matched..].as_ptr() as *const __m256i);
            let cur_vec = _mm256_loadu_si256(cur[matched..].as_ptr() as *const __m256i);

            // Compare for equality
            let cmp = _mm256_cmpeq_epi8(src_vec, cur_vec);
            let mask = _mm256_movemask_epi8(cmp) as u32;

            if mask != 0xFFFFFFFF {
                // Found a mismatch - count trailing ones
                matched += (!mask).trailing_zeros() as usize;
                return matched;
            }

            matched += 32;
        }
    }

    // Handle remaining bytes with scalar code
    while matched < max_len && src[matched] == cur[matched] {
        matched += 1;
    }

    matched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_match_length_empty() {
        assert_eq!(find_match_length(&[], &[], 0), 0);
        assert_eq!(find_match_length(&[1, 2, 3], &[], 3), 0);
        assert_eq!(find_match_length(&[], &[1, 2, 3], 3), 0);
    }

    #[test]
    fn test_find_match_length_no_match() {
        assert_eq!(find_match_length(&[1, 2, 3], &[4, 5, 6], 3), 0);
    }

    #[test]
    fn test_find_match_length_partial() {
        assert_eq!(find_match_length(&[1, 2, 3, 4], &[1, 2, 5, 6], 4), 2);
        assert_eq!(find_match_length(&[1, 2, 3, 4], &[1, 2, 3, 5], 4), 3);
    }

    #[test]
    fn test_find_match_length_full() {
        assert_eq!(find_match_length(&[1, 2, 3, 4], &[1, 2, 3, 4], 4), 4);
    }

    #[test]
    fn test_find_match_length_max_limit() {
        assert_eq!(find_match_length(&[1, 2, 3, 4], &[1, 2, 3, 4], 2), 2);
    }

    #[test]
    fn test_find_match_length_long() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        assert_eq!(find_match_length(&data, &data, 1000), 1000);

        let mut data2 = data.clone();
        data2[500] = 255;
        assert_eq!(find_match_length(&data, &data2, 1000), 500);
    }

    #[test]
    fn test_find_match_length_safe_matches_regular() {
        let data1: Vec<u8> = (0..100).map(|i| (i * 7) as u8).collect();
        let data2: Vec<u8> = (0..100).map(|i| (i * 7) as u8).collect();

        assert_eq!(
            find_match_length(&data1, &data2, 100),
            find_match_length_safe(&data1, &data2, 100)
        );

        let mut data3 = data2.clone();
        data3[50] = 0;

        assert_eq!(
            find_match_length(&data1, &data3, 100),
            find_match_length_safe(&data1, &data3, 100)
        );
    }

    #[test]
    fn test_find_match_length_alignment() {
        // Test various alignments to exercise SIMD edge cases
        for offset in 0..32 {
            let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
            let len = find_match_length(&data[offset..], &data[offset..], 256 - offset);
            assert_eq!(len, 256 - offset);
        }
    }

    #[test]
    fn test_find_match_length_large_for_avx512() {
        // Test with 64+ bytes to exercise AVX-512 path (if available)
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

        // Full match
        assert_eq!(find_match_length(&data, &data, 1024), 1024);

        // Mismatch at various positions to test all code paths
        for mismatch_pos in [0, 1, 31, 32, 63, 64, 65, 127, 128, 500, 1000] {
            let mut data2 = data.clone();
            if mismatch_pos < data2.len() {
                data2[mismatch_pos] = 255;
                assert_eq!(
                    find_match_length(&data, &data2, 1024),
                    mismatch_pos,
                    "Mismatch at position {} not detected correctly",
                    mismatch_pos
                );
            }
        }
    }

    #[test]
    fn test_find_match_length_avx512_boundary() {
        // Test specifically at 64-byte boundaries (AVX-512 vector width)
        for size in [64, 128, 192, 256, 320] {
            let data: Vec<u8> = vec![0xAA; size];
            assert_eq!(
                find_match_length(&data, &data, size),
                size,
                "Full match failed at size {}",
                size
            );

            // Test mismatch at last byte
            let mut data2 = data.clone();
            data2[size - 1] = 0xBB;
            assert_eq!(
                find_match_length(&data, &data2, size),
                size - 1,
                "Mismatch at last byte failed at size {}",
                size
            );
        }
    }
}
