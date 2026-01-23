//! SIMD-accelerated hash computation for LZ77 compression.
//!
//! Provides batch hash computation for multiple consecutive positions,
//! which is useful for updating hash tables during match skip phases.

/// Hash prime constants (same as match_finder in haagenti-zstd).
const HASH_PRIME: u32 = 0x9E3779B9;
const HASH_PRIME2: u32 = 0x85EBCA6B;

/// Compute hash for a single 4-byte sequence (scalar).
#[inline(always)]
pub fn hash4_scalar(data: &[u8], pos: usize) -> u32 {
    debug_assert!(pos + 4 <= data.len());
    let bytes = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let h = bytes.wrapping_mul(HASH_PRIME);
    let h = h ^ (h >> 15);
    h.wrapping_mul(HASH_PRIME2)
}

/// Compute hashes for 4 consecutive positions using best available SIMD.
///
/// Returns hashes for positions [pos, pos+1, pos+2, pos+3].
/// Requires at least 7 bytes of data starting at pos.
#[inline]
pub fn hash4x4(data: &[u8], pos: usize) -> [u32; 4] {
    debug_assert!(
        pos + 7 <= data.len(),
        "Need 7 bytes for 4 overlapping 4-byte windows"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            // Safety: we've checked SSE4.1 is available
            return unsafe { hash4x4_sse(data, pos) };
        }
    }

    // Scalar fallback
    hash4x4_scalar(data, pos)
}

/// Compute hashes for 8 consecutive positions using AVX2.
///
/// Returns hashes for positions [pos, pos+1, ..., pos+7].
/// Requires at least 11 bytes of data starting at pos.
#[inline]
pub fn hash4x8(data: &[u8], pos: usize) -> [u32; 8] {
    debug_assert!(
        pos + 11 <= data.len(),
        "Need 11 bytes for 8 overlapping 4-byte windows"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: we've checked AVX2 is available
            return unsafe { hash4x8_avx2(data, pos) };
        }
    }

    // Scalar fallback
    hash4x8_scalar(data, pos)
}

/// Scalar implementation for 4 hashes.
#[inline]
fn hash4x4_scalar(data: &[u8], pos: usize) -> [u32; 4] {
    [
        hash4_scalar(data, pos),
        hash4_scalar(data, pos + 1),
        hash4_scalar(data, pos + 2),
        hash4_scalar(data, pos + 3),
    ]
}

/// Scalar implementation for 8 hashes.
#[inline]
fn hash4x8_scalar(data: &[u8], pos: usize) -> [u32; 8] {
    [
        hash4_scalar(data, pos),
        hash4_scalar(data, pos + 1),
        hash4_scalar(data, pos + 2),
        hash4_scalar(data, pos + 3),
        hash4_scalar(data, pos + 4),
        hash4_scalar(data, pos + 5),
        hash4_scalar(data, pos + 6),
        hash4_scalar(data, pos + 7),
    ]
}

/// SSE4.1 implementation for 4 hashes at consecutive positions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn hash4x4_sse(data: &[u8], pos: usize) -> [u32; 4] {
    use std::arch::x86_64::*;

    unsafe {
        // Load 7 bytes and create 4 overlapping u32 values
        // We need bytes at: [pos..pos+4], [pos+1..pos+5], [pos+2..pos+6], [pos+3..pos+7]

        // Load as individual u32s (unaligned)
        let ptr = data.as_ptr().add(pos);
        let w0 = std::ptr::read_unaligned(ptr as *const u32);
        let w1 = std::ptr::read_unaligned(ptr.add(1) as *const u32);
        let w2 = std::ptr::read_unaligned(ptr.add(2) as *const u32);
        let w3 = std::ptr::read_unaligned(ptr.add(3) as *const u32);

        // Pack into SSE register
        let words = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);

        // Multiply by HASH_PRIME
        let prime = _mm_set1_epi32(HASH_PRIME as i32);
        let h = _mm_mullo_epi32(words, prime);

        // XOR with h >> 15
        let h_shifted = _mm_srli_epi32(h, 15);
        let h = _mm_xor_si128(h, h_shifted);

        // Multiply by HASH_PRIME2
        let prime2 = _mm_set1_epi32(HASH_PRIME2 as i32);
        let h = _mm_mullo_epi32(h, prime2);

        // Extract results
        let mut result = [0u32; 4];
        _mm_storeu_si128(result.as_mut_ptr() as *mut __m128i, h);
        result
    }
}

/// AVX2 implementation for 8 hashes at consecutive positions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hash4x8_avx2(data: &[u8], pos: usize) -> [u32; 8] {
    use std::arch::x86_64::*;

    unsafe {
        // Load 8 overlapping u32 values from consecutive positions
        let ptr = data.as_ptr().add(pos);

        // Load each 4-byte window individually (unaligned loads)
        let w0 = std::ptr::read_unaligned(ptr as *const u32);
        let w1 = std::ptr::read_unaligned(ptr.add(1) as *const u32);
        let w2 = std::ptr::read_unaligned(ptr.add(2) as *const u32);
        let w3 = std::ptr::read_unaligned(ptr.add(3) as *const u32);
        let w4 = std::ptr::read_unaligned(ptr.add(4) as *const u32);
        let w5 = std::ptr::read_unaligned(ptr.add(5) as *const u32);
        let w6 = std::ptr::read_unaligned(ptr.add(6) as *const u32);
        let w7 = std::ptr::read_unaligned(ptr.add(7) as *const u32);

        // Pack into AVX2 register (note: _mm256_set_epi32 is high-to-low order)
        let words = _mm256_set_epi32(
            w7 as i32, w6 as i32, w5 as i32, w4 as i32, w3 as i32, w2 as i32, w1 as i32, w0 as i32,
        );

        // Multiply by HASH_PRIME
        let prime = _mm256_set1_epi32(HASH_PRIME as i32);
        let h = _mm256_mullo_epi32(words, prime);

        // XOR with h >> 15
        let h_shifted = _mm256_srli_epi32(h, 15);
        let h = _mm256_xor_si256(h, h_shifted);

        // Multiply by HASH_PRIME2
        let prime2 = _mm256_set1_epi32(HASH_PRIME2 as i32);
        let h = _mm256_mullo_epi32(h, prime2);

        // Extract results
        let mut result = [0u32; 8];
        _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, h);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash4_scalar() {
        let data = b"abcdefghijklmnop";
        let h1 = hash4_scalar(data, 0);
        let h2 = hash4_scalar(data, 0);
        assert_eq!(h1, h2, "Same input should produce same hash");

        let h3 = hash4_scalar(data, 1);
        assert_ne!(
            h1, h3,
            "Different input should (usually) produce different hash"
        );
    }

    #[test]
    fn test_hash4x4_correctness() {
        let data = b"abcdefghijklmnop";

        let batch = hash4x4(data, 0);
        let scalar = [
            hash4_scalar(data, 0),
            hash4_scalar(data, 1),
            hash4_scalar(data, 2),
            hash4_scalar(data, 3),
        ];

        assert_eq!(batch, scalar, "Batch hash should match scalar");
    }

    #[test]
    fn test_hash4x8_correctness() {
        let data = b"abcdefghijklmnopqrstuvwxyz";

        let batch = hash4x8(data, 0);
        let scalar = [
            hash4_scalar(data, 0),
            hash4_scalar(data, 1),
            hash4_scalar(data, 2),
            hash4_scalar(data, 3),
            hash4_scalar(data, 4),
            hash4_scalar(data, 5),
            hash4_scalar(data, 6),
            hash4_scalar(data, 7),
        ];

        assert_eq!(batch, scalar, "Batch hash should match scalar");
    }

    #[test]
    fn test_hash4x4_various_offsets() {
        let data: Vec<u8> = (0..100).collect();

        for offset in 0..90 {
            let batch = hash4x4(&data, offset);
            let scalar = [
                hash4_scalar(&data, offset),
                hash4_scalar(&data, offset + 1),
                hash4_scalar(&data, offset + 2),
                hash4_scalar(&data, offset + 3),
            ];
            assert_eq!(batch, scalar, "Mismatch at offset {}", offset);
        }
    }

    #[test]
    fn test_hash4x8_various_offsets() {
        let data: Vec<u8> = (0..100).collect();

        for offset in 0..85 {
            let batch = hash4x8(&data, offset);
            for i in 0..8 {
                assert_eq!(
                    batch[i],
                    hash4_scalar(&data, offset + i),
                    "Mismatch at offset {} position {}",
                    offset,
                    i
                );
            }
        }
    }

    #[test]
    fn test_hash_distribution() {
        // Test that hashes are well-distributed
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let mut hashes = std::collections::HashSet::new();

        for offset in 0..240 {
            let h = hash4_scalar(&data, offset);
            hashes.insert(h);
        }

        // Should have mostly unique hashes (allow some collisions)
        assert!(
            hashes.len() > 200,
            "Hash distribution too poor: only {} unique out of 240",
            hashes.len()
        );
    }
}
