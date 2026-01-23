//! Mixed Precision Compression
//!
//! Implements a hybrid precision scheme where high-energy (essential) coefficients
//! are stored at FP16 precision, while detail coefficients use INT4 quantization.
//!
//! ## Rationale
//!
//! - **Essential coefficients** (top 10-20%): Capture the bulk of tensor energy.
//!   Using FP16 preserves their values accurately.
//! - **Detail coefficients** (remaining 80-90%): Small magnitudes that can tolerate
//!   quantization noise. INT4 with per-block scaling is sufficient.
//!
//! ## Storage Format
//!
//! ```text
//! +------------------+--------------------+------------------+
//! | FP16 Essentials  | INT4 Details       | Index Map        |
//! | (high precision) | (4-bit quantized)  | (coefficient pos)|
//! +------------------+--------------------+------------------+
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use haagenti::mixed_precision::{MixedPrecisionEncoder, MixedPrecisionDecoder};
//!
//! let encoder = MixedPrecisionEncoder::new(0.30, 0.20); // 30% retention, 20% as FP16
//! let compressed = encoder.encode(&tensor, width, height)?;
//!
//! let decoder = MixedPrecisionDecoder::new();
//! let reconstructed = decoder.decode(&compressed)?;
//! ```

use haagenti_core::{Error, Result};

/// Block size for INT4 quantization.
const Q4_BLOCK_SIZE: usize = 32;

/// Mixed precision compressed weight.
#[derive(Debug, Clone)]
pub struct MixedPrecisionWeight {
    /// Original tensor dimensions
    pub width: usize,
    pub height: usize,
    /// Total number of retained coefficients
    pub total_coefficients: usize,
    /// Number of FP16 essential coefficients
    pub fp16_count: usize,
    /// Number of INT4 detail coefficients
    pub int4_count: usize,
    /// FP16 essential coefficients (stored as raw bytes)
    pub fp16_data: Vec<u8>,
    /// INT4 detail coefficients with per-block FP16 scales
    pub int4_data: Vec<u8>,
    /// Index map: position of each coefficient in the original DCT array
    pub index_map: Vec<u32>,
    /// DCT coefficient ordering (for reconstruction)
    pub dct_total: usize,
}

impl MixedPrecisionWeight {
    /// Calculate storage size in bytes.
    pub fn storage_bytes(&self) -> usize {
        self.fp16_data.len() + self.int4_data.len() + self.index_map.len() * 4
    }

    /// Calculate original size in bytes (f32 tensor).
    pub fn original_bytes(&self) -> usize {
        self.width * self.height * 4
    }

    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        let orig = self.original_bytes();
        let compressed = self.storage_bytes();
        if compressed == 0 {
            0.0
        } else {
            orig as f32 / compressed as f32
        }
    }

    /// Get fraction of coefficients stored as FP16.
    pub fn fp16_fraction(&self) -> f32 {
        if self.total_coefficients == 0 {
            0.0
        } else {
            self.fp16_count as f32 / self.total_coefficients as f32
        }
    }
}

/// Mixed precision encoder.
///
/// Encodes tensor using DCT, then stores:
/// - Top essential coefficients as FP16
/// - Remaining detail coefficients as INT4
#[derive(Debug, Clone)]
pub struct MixedPrecisionEncoder {
    /// Total retention ratio (fraction of DCT coefficients to keep)
    retention: f32,
    /// Fraction of retained coefficients to store as FP16
    fp16_ratio: f32,
}

impl Default for MixedPrecisionEncoder {
    fn default() -> Self {
        Self::new(0.30, 0.20)
    }
}

impl MixedPrecisionEncoder {
    /// Create encoder with retention and FP16 ratio.
    ///
    /// # Arguments
    /// * `retention` - Fraction of total DCT coefficients to retain (0.0-1.0)
    /// * `fp16_ratio` - Fraction of retained coefficients to store as FP16 (0.0-1.0)
    pub fn new(retention: f32, fp16_ratio: f32) -> Self {
        Self {
            retention: retention.clamp(0.01, 1.0),
            fp16_ratio: fp16_ratio.clamp(0.0, 1.0),
        }
    }

    /// Set retention ratio.
    pub fn with_retention(mut self, r: f32) -> Self {
        self.retention = r.clamp(0.01, 1.0);
        self
    }

    /// Set FP16 ratio.
    pub fn with_fp16_ratio(mut self, r: f32) -> Self {
        self.fp16_ratio = r.clamp(0.0, 1.0);
        self
    }

    /// Encode a 2D tensor.
    pub fn encode(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<MixedPrecisionWeight> {
        let n = width * height;
        if data.len() != n {
            return Err(Error::corrupted("data size mismatch"));
        }

        if n == 0 {
            return Err(Error::corrupted("empty tensor"));
        }

        // Perform 2D DCT
        let dct_coeffs = dct_2d(data, width, height);

        // Calculate how many coefficients to retain
        let total_retain = ((n as f32 * self.retention) as usize).max(1).min(n);
        let fp16_count = ((total_retain as f32 * self.fp16_ratio) as usize)
            .max(1)
            .min(total_retain);
        let int4_count = total_retain - fp16_count;

        // Sort coefficients by magnitude to find the most important ones
        let mut indexed: Vec<(usize, f32)> = dct_coeffs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top `total_retain` coefficients
        let retained: Vec<(usize, f32)> = indexed.into_iter().take(total_retain).collect();

        // Split into FP16 (top fp16_count) and INT4 (rest)
        let fp16_coeffs: Vec<(usize, f32)> = retained[..fp16_count].to_vec();
        let int4_coeffs: Vec<(usize, f32)> = retained[fp16_count..].to_vec();

        // Encode FP16 coefficients
        let fp16_data = encode_fp16(&fp16_coeffs.iter().map(|(_, v)| *v).collect::<Vec<_>>());

        // Encode INT4 coefficients
        let int4_data = quantize_int4(&int4_coeffs.iter().map(|(_, v)| *v).collect::<Vec<_>>());

        // Build index map: indices of retained coefficients in DCT order
        let mut index_map: Vec<u32> = Vec::with_capacity(total_retain);
        for (idx, _) in &fp16_coeffs {
            index_map.push(*idx as u32);
        }
        for (idx, _) in &int4_coeffs {
            index_map.push(*idx as u32);
        }

        Ok(MixedPrecisionWeight {
            width,
            height,
            total_coefficients: total_retain,
            fp16_count,
            int4_count,
            fp16_data,
            int4_data,
            index_map,
            dct_total: n,
        })
    }

    /// Get expected storage size for given dimensions.
    pub fn expected_storage(&self, width: usize, height: usize) -> usize {
        let n = width * height;
        let total_retain = ((n as f32 * self.retention) as usize).max(1);
        let fp16_count = ((total_retain as f32 * self.fp16_ratio) as usize).max(1);
        let int4_count = total_retain - fp16_count;

        // FP16: 2 bytes per coefficient
        let fp16_bytes = fp16_count * 2;

        // INT4: scales + packed nibbles
        let int4_blocks = int4_count.div_ceil(Q4_BLOCK_SIZE);
        let int4_bytes = int4_blocks * 2 + int4_count.div_ceil(2);

        // Index map: 4 bytes per coefficient
        let index_bytes = total_retain * 4;

        fp16_bytes + int4_bytes + index_bytes
    }
}

/// Mixed precision decoder.
#[derive(Debug, Clone, Default)]
pub struct MixedPrecisionDecoder;

impl MixedPrecisionDecoder {
    /// Create a new decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decode mixed precision compressed weight.
    pub fn decode(&self, compressed: &MixedPrecisionWeight) -> Result<Vec<f32>> {
        let n = compressed.width * compressed.height;

        // Decode FP16 coefficients
        let fp16_values = decode_fp16(&compressed.fp16_data, compressed.fp16_count);

        // Decode INT4 coefficients
        let int4_values = dequantize_int4(&compressed.int4_data, compressed.int4_count);

        // Reconstruct DCT coefficient array
        let mut dct_coeffs = vec![0.0f32; n];

        // Place FP16 coefficients
        for (i, &value) in fp16_values.iter().enumerate() {
            if i < compressed.index_map.len() {
                let idx = compressed.index_map[i] as usize;
                if idx < n {
                    dct_coeffs[idx] = value;
                }
            }
        }

        // Place INT4 coefficients
        for (i, &value) in int4_values.iter().enumerate() {
            let map_idx = compressed.fp16_count + i;
            if map_idx < compressed.index_map.len() {
                let idx = compressed.index_map[map_idx] as usize;
                if idx < n {
                    dct_coeffs[idx] = value;
                }
            }
        }

        // Inverse DCT
        let reconstructed = idct_2d(&dct_coeffs, compressed.width, compressed.height);

        Ok(reconstructed)
    }

    /// Decode only FP16 essentials (progressive decompression).
    pub fn decode_essentials_only(&self, compressed: &MixedPrecisionWeight) -> Result<Vec<f32>> {
        let n = compressed.width * compressed.height;

        // Decode FP16 coefficients only
        let fp16_values = decode_fp16(&compressed.fp16_data, compressed.fp16_count);

        // Reconstruct DCT coefficient array with only essentials
        let mut dct_coeffs = vec![0.0f32; n];

        for (i, &value) in fp16_values.iter().enumerate() {
            if i < compressed.index_map.len() {
                let idx = compressed.index_map[i] as usize;
                if idx < n {
                    dct_coeffs[idx] = value;
                }
            }
        }

        // Inverse DCT
        let reconstructed = idct_2d(&dct_coeffs, compressed.width, compressed.height);

        Ok(reconstructed)
    }
}

// =============================================================================
// DCT Implementation (simplified 2D DCT using separable 1D transforms)
// =============================================================================

/// 1D DCT-II transform.
fn dct_1d(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let mut output = vec![0.0f32; n];
    let scale = (2.0 / n as f32).sqrt();

    for k in 0..n {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum +=
                input[i] * (std::f32::consts::PI * ((2 * i + 1) * k) as f32 / (2 * n) as f32).cos();
        }
        output[k] = sum
            * scale
            * if k == 0 {
                1.0 / std::f32::consts::SQRT_2
            } else {
                1.0
            };
    }

    output
}

/// 1D IDCT-II transform (inverse DCT).
fn idct_1d(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let mut output = vec![0.0f32; n];
    let scale = (2.0 / n as f32).sqrt();

    for i in 0..n {
        let mut sum = 0.0f32;
        for k in 0..n {
            let coeff = input[k]
                * if k == 0 {
                    1.0 / std::f32::consts::SQRT_2
                } else {
                    1.0
                };
            sum += coeff * (std::f32::consts::PI * ((2 * i + 1) * k) as f32 / (2 * n) as f32).cos();
        }
        output[i] = sum * scale;
    }

    output
}

/// 2D DCT using separable 1D transforms.
fn dct_2d(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    if width == 0 || height == 0 {
        return vec![];
    }

    // DCT on rows
    let mut temp = vec![0.0f32; width * height];
    for row in 0..height {
        let row_data: Vec<f32> = data[row * width..(row + 1) * width].to_vec();
        let dct_row = dct_1d(&row_data);
        temp[row * width..(row + 1) * width].copy_from_slice(&dct_row);
    }

    // DCT on columns
    let mut output = vec![0.0f32; width * height];
    for col in 0..width {
        let col_data: Vec<f32> = (0..height).map(|row| temp[row * width + col]).collect();
        let dct_col = dct_1d(&col_data);
        for row in 0..height {
            output[row * width + col] = dct_col[row];
        }
    }

    output
}

/// 2D IDCT using separable 1D transforms.
fn idct_2d(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    if width == 0 || height == 0 {
        return vec![];
    }

    // IDCT on columns
    let mut temp = vec![0.0f32; width * height];
    for col in 0..width {
        let col_data: Vec<f32> = (0..height).map(|row| data[row * width + col]).collect();
        let idct_col = idct_1d(&col_data);
        for row in 0..height {
            temp[row * width + col] = idct_col[row];
        }
    }

    // IDCT on rows
    let mut output = vec![0.0f32; width * height];
    for row in 0..height {
        let row_data: Vec<f32> = temp[row * width..(row + 1) * width].to_vec();
        let idct_row = idct_1d(&row_data);
        output[row * width..(row + 1) * width].copy_from_slice(&idct_row);
    }

    output
}

// =============================================================================
// FP16 Encoding/Decoding
// =============================================================================

/// Encode f32 values as FP16 bytes.
fn encode_fp16(values: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let fp16 = half::f16::from_f32(v);
        output.extend_from_slice(&fp16.to_le_bytes());
    }
    output
}

/// Decode FP16 bytes to f32 values.
fn decode_fp16(data: &[u8], count: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * 2;
        if offset + 2 <= data.len() {
            let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
            output.push(half::f16::from_bits(bits).to_f32());
        }
    }
    output
}

// =============================================================================
// INT4 Quantization/Dequantization
// =============================================================================

/// Quantize f32 values to INT4 with per-block FP16 scaling.
fn quantize_int4(values: &[f32]) -> Vec<u8> {
    if values.is_empty() {
        return vec![];
    }

    let num_blocks = values.len().div_ceil(Q4_BLOCK_SIZE);
    let mut output = Vec::with_capacity(num_blocks * 2 + values.len().div_ceil(2));

    // First pass: compute and store scales
    let mut scales = Vec::with_capacity(num_blocks);
    for block in values.chunks(Q4_BLOCK_SIZE) {
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1.0 };
        scales.push(scale);
        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    }

    // Second pass: quantize and pack nibbles
    let mut nibbles = Vec::with_capacity(values.len());
    for (block_idx, block) in values.chunks(Q4_BLOCK_SIZE).enumerate() {
        let scale = scales[block_idx];
        for &val in block {
            let q = ((val / scale).round() as i8).clamp(-8, 7);
            nibbles.push((q + 8) as u8);
        }
    }

    // Pack nibbles into bytes
    for pair in nibbles.chunks(2) {
        let byte = if pair.len() == 2 {
            (pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4)
        } else {
            pair[0] & 0x0F
        };
        output.push(byte);
    }

    output
}

/// Dequantize INT4 back to f32.
fn dequantize_int4(data: &[u8], num_elements: usize) -> Vec<f32> {
    if num_elements == 0 || data.is_empty() {
        return vec![];
    }

    let num_blocks = num_elements.div_ceil(Q4_BLOCK_SIZE);
    let scales_bytes = num_blocks * 2;

    if data.len() < scales_bytes {
        return vec![0.0; num_elements];
    }

    // Read scales
    let scales: Vec<f32> = data[..scales_bytes]
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();

    // Unpack nibbles and dequantize
    let packed_data = &data[scales_bytes..];
    let mut output = Vec::with_capacity(num_elements);

    let mut nibble_idx = 0;
    for block_idx in 0..num_blocks {
        let scale = scales.get(block_idx).copied().unwrap_or(1.0);
        let block_size = Q4_BLOCK_SIZE.min(num_elements - block_idx * Q4_BLOCK_SIZE);

        for _ in 0..block_size {
            let byte_idx = nibble_idx / 2;
            let is_high = nibble_idx % 2 == 1;

            if byte_idx >= packed_data.len() {
                output.push(0.0);
                nibble_idx += 1;
                continue;
            }

            let nibble = if is_high {
                (packed_data[byte_idx] >> 4) & 0x0F
            } else {
                packed_data[byte_idx] & 0x0F
            };

            let q = (nibble as i8) - 8;
            output.push(q as f32 * scale);
            nibble_idx += 1;
        }
    }

    output
}

// =============================================================================
// Quality Metrics
// =============================================================================

/// Calculate MSE between two vectors.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::MAX;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / a.len() as f32
}

/// Calculate cosine similarity.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_basic() {
        let encoder = MixedPrecisionEncoder::new(0.50, 0.20);
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let compressed = encoder.encode(&data, 8, 8).unwrap();

        assert!(compressed.total_coefficients > 0);
        assert!(compressed.fp16_count > 0);
        assert_eq!(
            compressed.total_coefficients,
            compressed.fp16_count + compressed.int4_count
        );
    }

    #[test]
    fn test_mixed_precision_roundtrip() {
        let encoder = MixedPrecisionEncoder::new(0.70, 0.30);
        let decoder = MixedPrecisionDecoder::new();

        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let compressed = encoder.encode(&data, 8, 8).unwrap();
        let reconstructed = decoder.decode(&compressed).unwrap();

        assert_eq!(reconstructed.len(), data.len());

        // Check quality
        let cos = cosine_similarity(&data, &reconstructed);
        assert!(cos > 0.8, "Cosine similarity too low: {}", cos);
    }

    #[test]
    fn test_mixed_precision_vs_all_int4() {
        let encoder_mixed = MixedPrecisionEncoder::new(0.50, 0.30);
        let encoder_all_int4 = MixedPrecisionEncoder::new(0.50, 0.0); // 0% FP16 = all INT4
        let decoder = MixedPrecisionDecoder::new();

        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();

        let compressed_mixed = encoder_mixed.encode(&data, 16, 16).unwrap();
        let compressed_int4 = encoder_all_int4.encode(&data, 16, 16).unwrap();

        let recon_mixed = decoder.decode(&compressed_mixed).unwrap();
        let recon_int4 = decoder.decode(&compressed_int4).unwrap();

        let cos_mixed = cosine_similarity(&data, &recon_mixed);
        let cos_int4 = cosine_similarity(&data, &recon_int4);

        // Mixed precision should be at least as good as all-INT4
        assert!(
            cos_mixed >= cos_int4 * 0.99,
            "Mixed ({}) should be >= INT4 ({})",
            cos_mixed,
            cos_int4
        );
    }

    #[test]
    fn test_progressive_decode() {
        let encoder = MixedPrecisionEncoder::new(0.50, 0.30);
        let decoder = MixedPrecisionDecoder::new();

        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let compressed = encoder.encode(&data, 8, 8).unwrap();

        // Decode essentials only
        let essentials_only = decoder.decode_essentials_only(&compressed).unwrap();
        // Decode full
        let full = decoder.decode(&compressed).unwrap();

        // Full decode should be better than essentials only
        let cos_essentials = cosine_similarity(&data, &essentials_only);
        let cos_full = cosine_similarity(&data, &full);

        assert!(
            cos_full >= cos_essentials,
            "Full ({}) should be >= essentials ({})",
            cos_full,
            cos_essentials
        );
    }

    #[test]
    fn test_compression_ratio() {
        let encoder = MixedPrecisionEncoder::new(0.30, 0.20);
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

        let compressed = encoder.encode(&data, 32, 32).unwrap();

        // Should achieve some compression
        let ratio = compressed.compression_ratio();
        assert!(
            ratio > 1.0,
            "Expected compression ratio > 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_fp16_fraction() {
        let encoder = MixedPrecisionEncoder::new(0.50, 0.25);
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();

        let compressed = encoder.encode(&data, 8, 8).unwrap();

        let fraction = compressed.fp16_fraction();
        assert!(
            (fraction - 0.25).abs() < 0.1,
            "FP16 fraction should be ~0.25, got {}",
            fraction
        );
    }

    #[test]
    fn test_empty_tensor() {
        let encoder = MixedPrecisionEncoder::new(0.50, 0.20);
        let result = encoder.encode(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_size_mismatch() {
        let encoder = MixedPrecisionEncoder::new(0.50, 0.20);
        let data = vec![1.0f32; 10];
        let result = encoder.encode(&data, 8, 8); // Should be 64 elements
        assert!(result.is_err());
    }

    #[test]
    fn test_dct_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let dct = dct_2d(&data, 8, 8);
        let reconstructed = idct_2d(&dct, 8, 8);

        let cos = cosine_similarity(&data, &reconstructed);
        assert!(cos > 0.999, "DCT roundtrip should be near-perfect: {}", cos);
    }

    #[test]
    fn test_int4_roundtrip() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();

        let quantized = quantize_int4(&data);
        let dequantized = dequantize_int4(&quantized, data.len());

        assert_eq!(dequantized.len(), data.len());

        // INT4 has limited precision, but should be reasonable
        let cos = cosine_similarity(&data, &dequantized);
        assert!(cos > 0.9, "INT4 roundtrip cosine: {}", cos);
    }

    #[test]
    fn test_fp16_roundtrip() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();

        let encoded = encode_fp16(&data);
        let decoded = decode_fp16(&encoded, data.len());

        assert_eq!(decoded.len(), data.len());

        // FP16 should be very accurate
        let cos = cosine_similarity(&data, &decoded);
        assert!(cos > 0.9999, "FP16 roundtrip cosine: {}", cos);
    }

    #[test]
    fn test_expected_storage() {
        let encoder = MixedPrecisionEncoder::new(0.30, 0.20);

        let expected = encoder.expected_storage(32, 32);
        assert!(expected > 0);
        assert!(expected < 32 * 32 * 4); // Should be less than original
    }
}
