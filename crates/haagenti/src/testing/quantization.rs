//! INT4 quantization utilities for weight compression testing.
//!
//! Provides per-block INT4 quantization with FP16 scales:
//! - Each block of 32 weights gets one FP16 scale factor
//! - Weights are quantized to signed 4-bit integers (-8 to +7)
//! - Stored as packed nibbles with scales prepended

/// Block size for INT4 quantization.
/// Each block of 32 elements shares one FP16 scale factor.
pub const Q4_BLOCK_SIZE: usize = 32;

/// Quantize f32 weights to INT4 with per-block FP16 scaling.
///
/// # Format
///
/// Output structure:
/// - First `num_blocks * 2` bytes: FP16 scale factors
/// - Remaining bytes: packed INT4 nibbles (2 weights per byte)
///
/// Each weight is quantized as: `round(weight / scale)` clamped to [-8, 7]
/// and stored as unsigned nibble (value + 8) in range [0, 15].
///
/// # Arguments
///
/// * `weights` - Input f32 weights to quantize
///
/// # Returns
///
/// Quantized data with scales prepended.
pub fn quantize_int4(weights: &[f32]) -> Vec<u8> {
    if weights.is_empty() {
        return vec![];
    }

    let num_blocks = (weights.len() + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_blocks * 2 + (weights.len() + 1) / 2);

    // First pass: compute and store scales
    let mut scales = Vec::with_capacity(num_blocks);
    for block in weights.chunks(Q4_BLOCK_SIZE) {
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // Scale maps max value to 7 (positive range of INT4)
        let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1.0 };
        scales.push(scale);
        output.extend_from_slice(&f32_to_f16_bytes(scale));
    }

    // Second pass: quantize and pack nibbles
    let mut nibble_buffer = Vec::with_capacity(weights.len());
    for (block_idx, block) in weights.chunks(Q4_BLOCK_SIZE).enumerate() {
        let scale = scales[block_idx];
        for &val in block {
            let q = ((val / scale).round() as i8).clamp(-8, 7);
            // Store as unsigned: -8 -> 0, 7 -> 15
            nibble_buffer.push((q + 8) as u8);
        }
    }

    // Pack nibbles into bytes (low nibble first, then high nibble)
    for pair in nibble_buffer.chunks(2) {
        let byte = if pair.len() == 2 {
            (pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4)
        } else {
            pair[0] & 0x0F
        };
        output.push(byte);
    }

    output
}

/// Dequantize INT4 data back to f32.
///
/// # Arguments
///
/// * `data` - Quantized data (scales + packed nibbles)
/// * `num_elements` - Original number of elements
///
/// # Returns
///
/// Reconstructed f32 weights.
pub fn dequantize_int4(data: &[u8], num_elements: usize) -> Vec<f32> {
    if num_elements == 0 {
        return vec![];
    }

    let num_blocks = (num_elements + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let scales_bytes = num_blocks * 2;

    if data.len() < scales_bytes {
        return vec![];
    }

    // Read FP16 scales
    let scales: Vec<f32> = data[..scales_bytes]
        .chunks_exact(2)
        .map(|c| f16_bytes_to_f32([c[0], c[1]]))
        .collect();

    // Unpack nibbles and dequantize
    let packed_data = &data[scales_bytes..];
    let mut output = Vec::with_capacity(num_elements);

    let mut nibble_idx = 0;
    for block_idx in 0..num_blocks {
        let scale = scales[block_idx];
        let block_size = Q4_BLOCK_SIZE.min(num_elements - block_idx * Q4_BLOCK_SIZE);

        for _ in 0..block_size {
            let byte_idx = nibble_idx / 2;
            let is_high = nibble_idx % 2 == 1;

            if byte_idx >= packed_data.len() {
                break;
            }

            let nibble = if is_high {
                (packed_data[byte_idx] >> 4) & 0x0F
            } else {
                packed_data[byte_idx] & 0x0F
            };

            // Convert back from unsigned: 0 -> -8, 15 -> 7
            let q = (nibble as i8) - 8;
            output.push(q as f32 * scale);
            nibble_idx += 1;
        }
    }

    output
}

/// Convert f32 to FP16 bytes (little-endian).
///
/// Uses manual conversion to avoid requiring the half crate at runtime.
pub fn f32_to_f16_bytes(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;

    let h: u16 = if exp == 0 {
        // Zero or subnormal (flush to zero for simplicity)
        (sign << 15) as u16
    } else if exp == 0xff {
        // Inf or NaN
        ((sign << 15) | (0x1f << 10) | (frac >> 13)) as u16
    } else {
        // Normal number
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            // Overflow to infinity
            ((sign << 15) | (0x1f << 10)) as u16
        } else if new_exp <= 0 {
            // Underflow to zero
            (sign << 15) as u16
        } else {
            ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
        }
    };

    h.to_le_bytes()
}

/// Convert FP16 bytes (little-endian) to f32.
pub fn f16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);

    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = -14i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            let new_exp = ((e + 127) as u32) & 0xff;
            f32::from_bits((sign << 31) | (new_exp << 23) | (f << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
    } else {
        // Normal
        let new_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
    }
}

/// Calculate the compressed size for a given number of elements.
///
/// Returns the total byte size including scales and packed nibbles.
pub fn compressed_size(num_elements: usize) -> usize {
    if num_elements == 0 {
        return 0;
    }
    let num_blocks = (num_elements + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let scales_bytes = num_blocks * 2;
    let nibbles_bytes = (num_elements + 1) / 2;
    scales_bytes + nibbles_bytes
}

/// Calculate the compression ratio for INT4 quantization.
///
/// Compares f32 input to INT4 output size.
pub fn compression_ratio(num_elements: usize) -> f32 {
    if num_elements == 0 {
        return 1.0;
    }
    let original_bytes = num_elements * 4; // f32
    let compressed_bytes = compressed_size(num_elements);
    original_bytes as f32 / compressed_bytes as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();

        let quantized = quantize_int4(&weights);
        let dequantized = dequantize_int4(&quantized, weights.len());

        assert_eq!(dequantized.len(), weights.len());

        // Check that values are close (INT4 has limited precision)
        for (orig, deq) in weights.iter().zip(dequantized.iter()) {
            // INT4 can have up to ~14% error per value
            let max_error = orig.abs() * 0.15 + 0.1;
            assert!(
                (orig - deq).abs() < max_error,
                "Too much error: orig={}, deq={}, diff={}",
                orig,
                deq,
                (orig - deq).abs()
            );
        }
    }

    #[test]
    fn test_quantize_empty() {
        let result = quantize_int4(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_empty() {
        let result = dequantize_int4(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_quantize_single_block() {
        let weights: Vec<f32> = vec![1.0; Q4_BLOCK_SIZE];
        let quantized = quantize_int4(&weights);

        // Should have: 2 bytes scale + 16 bytes data (32 nibbles)
        assert_eq!(quantized.len(), 2 + Q4_BLOCK_SIZE / 2);
    }

    #[test]
    fn test_f16_conversion_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -100.0, 0.001];
        for &val in &values {
            let bytes = f32_to_f16_bytes(val);
            let result = f16_bytes_to_f32(bytes);
            // FP16 has limited precision
            let tolerance = val.abs() * 0.01 + 0.001;
            assert!(
                (val - result).abs() < tolerance,
                "F16 roundtrip failed: {} -> {:?} -> {}",
                val,
                bytes,
                result
            );
        }
    }

    #[test]
    fn test_compressed_size() {
        // 32 elements = 1 block = 2 bytes scale + 16 bytes data = 18 bytes
        assert_eq!(compressed_size(32), 18);

        // 64 elements = 2 blocks = 4 bytes scales + 32 bytes data = 36 bytes
        assert_eq!(compressed_size(64), 36);

        // 33 elements = 2 blocks = 4 bytes scales + 17 bytes data = 21 bytes
        assert_eq!(compressed_size(33), 21);
    }

    #[test]
    fn test_compression_ratio() {
        // 32 elements: 128 bytes f32 / 18 bytes INT4 ≈ 7.1x
        let ratio = compression_ratio(32);
        assert!(
            ratio > 7.0 && ratio < 7.2,
            "Expected ratio ~7.1 for 32 elements, got {}",
            ratio
        );

        // Large arrays: scales overhead becomes negligible
        // 10000 elements: 313 blocks = 626 bytes scales + 5000 bytes data = 5626 bytes
        // Ratio: 40000 / 5626 ≈ 7.11x
        let ratio = compression_ratio(10000);
        assert!(
            ratio > 7.0 && ratio < 7.2,
            "Expected ratio ~7.1 for 10000 elements, got {}",
            ratio
        );

        // Compression ratio is bounded:
        // - Lower bound: ~7x (small arrays with scale overhead)
        // - Upper bound: ~7.27x (theoretical limit as array size → ∞)
        // The ratio should always be positive and reasonable
        let ratio = compression_ratio(1_000_000);
        assert!(
            ratio > 7.0 && ratio < 8.0,
            "Expected ratio between 7-8 for large arrays, got {}",
            ratio
        );
    }

    #[test]
    fn test_quantize_preserves_sign() {
        let weights = vec![-5.0, 5.0, -2.5, 2.5];
        let quantized = quantize_int4(&weights);
        let dequantized = dequantize_int4(&quantized, 4);

        assert!(dequantized[0] < 0.0);
        assert!(dequantized[1] > 0.0);
        assert!(dequantized[2] < 0.0);
        assert!(dequantized[3] > 0.0);
    }

    #[test]
    fn test_quantize_zeros() {
        let weights = vec![0.0; 64];
        let quantized = quantize_int4(&weights);
        let dequantized = dequantize_int4(&quantized, 64);

        for val in dequantized {
            assert!(val.abs() < 1e-6);
        }
    }
}
