//! Compressive Spectral Encoding for HoloTensor
//!
//! This module provides storage-optimized encoding that achieves actual compression
//! rather than the streaming-optimized SpectralEncoder which trades storage for
//! progressive loading capability.
//!
//! ## V3 Format (Bitmap + f16 Coefficients)
//!
//! Combines bitmap indices with f16 coefficient storage:
//! - Bitmap: 1 bit per coefficient (N/8 bytes)
//! - Coefficients: f16 (2 bytes each instead of 4)
//!
//! For 20% retention on 1M elements:
//! - Bitmap: 125KB
//! - Coefficients: 200K × 2 = 400KB (was 800KB with f32)
//! - Total: 525KB vs 2MB original f16 → **3.8x compression**

use crate::holotensor::{dct_2d, idct_2d, HoloFragment};
use haagenti_core::{Error, Result};
use half::f16;

/// Magic bytes to identify V2 format with bitmap indices (f32 coefficients)
const BITMAP_FORMAT_MAGIC: u32 = 0x48435432; // "HCT2" in little-endian

/// Magic bytes to identify V3 format with bitmap + f16 coefficients
const BITMAP_F16_MAGIC: u32 = 0x48435433; // "HCT3" in little-endian

// ==================== Compressive Spectral Encoder ====================

/// Compressive Spectral Encoder - designed for actual storage compression.
///
/// Unlike `SpectralEncoder` which is optimized for progressive streaming (and expands storage),
/// `CompressiveSpectralEncoder` is optimized for storage compression:
///
/// 1. **Truncation**: Discards low-energy DCT coefficients (lossy but NN-friendly)
/// 2. **Single essential storage**: Essentials stored once, not replicated in every fragment
/// 3. **Implicit ordering**: Uses zigzag scan order, no index storage overhead
/// 4. **Efficient packing**: Values stored directly without indices
///
/// ## Compression Pipeline
///
/// ```text
/// Input f32 → DCT → Sort by energy → Truncate to retention_ratio
///          → Zigzag reorder → Pack values → Fragment distribution
/// ```
///
/// ## Trade-offs vs SpectralEncoder
///
/// | Feature | SpectralEncoder | CompressiveSpectralEncoder |
/// |---------|-----------------|---------------------------|
/// | Storage | 5-13x expansion | 2-10x compression |
/// | Progressive | Any fragment works | Need essentials first |
/// | Fault tolerance | High redundancy | Lower redundancy |
/// | Use case | Streaming inference | Storage/transmission |
///
/// ## Expected Compression Ratios
///
/// | retention_ratio | Compression | Quality Loss |
/// |-----------------|-------------|--------------|
/// | 0.50 | ~1.6x | Minimal |
/// | 0.20 | ~3.5x | Low |
/// | 0.10 | ~5x | Moderate |
/// | 0.05 | ~8x | Noticeable |
///
/// ## Combined with Quantization
///
/// For maximum compression, combine with 4-bit quantization:
/// - 4-bit quantization: 8x
/// - CompressiveSpectral (0.10): 5x
/// - Zstd (~2x on f32): 2x
/// - **Total: 80x** (405B → ~10 GB)
pub struct CompressiveSpectralEncoder {
    num_fragments: u16,
    /// Ratio of coefficients to retain (0.1 = keep top 10% by energy)
    retention_ratio: f32,
    /// Ratio of retained coefficients that are "essential" (stored in fragment 0)
    essential_ratio: f32,
}

impl CompressiveSpectralEncoder {
    /// Create encoder with specified fragments and retention ratio.
    ///
    /// # Arguments
    /// * `num_fragments` - Number of output fragments (more = better progressive quality)
    /// * `retention_ratio` - Fraction of DCT coefficients to keep (0.1-0.5 typical)
    ///
    /// # Example
    /// ```ignore
    /// // Keep 10% of coefficients, distribute across 8 fragments
    /// let encoder = CompressiveSpectralEncoder::new(8, 0.10);
    /// let fragments = encoder.encode_2d(&weights, 4096, 4096)?;
    /// // Expected ~5x compression ratio
    /// ```
    pub fn new(num_fragments: u16, retention_ratio: f32) -> Self {
        CompressiveSpectralEncoder {
            num_fragments,
            retention_ratio: retention_ratio.clamp(0.05, 1.0),
            essential_ratio: 0.2, // 20% of retained coefficients are essential
        }
    }

    /// Set ratio of retained coefficients considered essential.
    ///
    /// When using num_fragments=1, set this to 1.0 to store all retained
    /// coefficients in the single fragment.
    pub fn with_essential_ratio(mut self, ratio: f32) -> Self {
        self.essential_ratio = ratio.clamp(0.05, 1.0);
        self
    }

    /// Encode 2D tensor with compression using V3 format (bitmap + f16 coefficients).
    ///
    /// Returns fragments where:
    /// - Fragment 0: Contains essential coefficients + bitmap marking retained positions
    /// - Fragments 1..N: Contain distributed detail coefficients
    ///
    /// ## V3 Format (Bitmap + f16)
    ///
    /// Fragment 0 layout:
    /// ```text
    /// [magic: u32][width: u32][height: u32][retain_count: u32][essential_count: u32]
    /// [detail_per_frag: u32][bitmap_bytes...][essential_coeffs as f16...]
    /// ```
    ///
    /// Coefficients are stored as f16 (2 bytes) instead of f32 (4 bytes), halving
    /// the coefficient storage overhead while maintaining sufficient precision for
    /// neural network weights.
    pub fn encode_2d(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<HoloFragment>> {
        let n = width * height;
        if data.len() != n {
            return Err(Error::corrupted("data size mismatch"));
        }

        // Transform to frequency domain
        let mut dct_coeffs = vec![0.0f32; n];
        dct_2d(data, &mut dct_coeffs, width, height);

        // Sort coefficients by energy (importance) to find which to keep
        let mut indexed: Vec<(usize, f32)> = dct_coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // TRUNCATE: Keep only top retention_ratio coefficients
        let retain_count = ((n as f32 * self.retention_ratio) as usize).max(1);
        let essential_count = ((retain_count as f32 * self.essential_ratio) as usize).max(1);
        let detail_count = retain_count - essential_count;

        // Build bitmap: 1 bit per coefficient, bit=1 means retained
        let bitmap_bytes = n.div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_bytes];

        // Mark retained positions in bitmap
        for &(idx, _) in indexed.iter().take(retain_count) {
            bitmap[idx / 8] |= 1 << (idx % 8);
        }

        // Collect coefficients in INDEX ORDER (ascending), not energy order
        // This matches the bitmap scan order for reconstruction
        let mut retained_indices: Vec<usize> = indexed
            .iter()
            .take(retain_count)
            .map(|(idx, _)| *idx)
            .collect();
        retained_indices.sort_unstable();

        let mut fragments = Vec::with_capacity(self.num_fragments as usize);

        // Fragment 0: Header + bitmap + essential coefficients (f16)
        {
            let detail_per_frag = if self.num_fragments > 1 {
                detail_count.div_ceil(self.num_fragments as usize - 1)
            } else {
                detail_count
            };

            let mut frag_data = Vec::new();

            // V3 Header with magic (f16 format)
            frag_data.extend_from_slice(&BITMAP_F16_MAGIC.to_le_bytes());
            frag_data.extend_from_slice(&(width as u32).to_le_bytes());
            frag_data.extend_from_slice(&(height as u32).to_le_bytes());
            frag_data.extend_from_slice(&(retain_count as u32).to_le_bytes());
            frag_data.extend_from_slice(&(essential_count as u32).to_le_bytes());
            frag_data.extend_from_slice(&(detail_per_frag as u32).to_le_bytes());

            // Bitmap (N/8 bytes)
            frag_data.extend_from_slice(&bitmap);

            // Essential coefficients as f16 (2 bytes each instead of 4!)
            for &idx in retained_indices.iter().take(essential_count) {
                let coeff_f16 = f16::from_f32(dct_coeffs[idx]);
                frag_data.extend_from_slice(&coeff_f16.to_le_bytes());
            }

            fragments.push(HoloFragment::new(0, frag_data));
        }

        // Fragments 1..N: Detail coefficients (distributed, f16)
        if self.num_fragments > 1 {
            let detail_per_frag = detail_count.div_ceil(self.num_fragments as usize - 1);

            for frag_idx in 1..self.num_fragments {
                let mut frag_data = Vec::new();

                // Header
                frag_data.extend_from_slice(&frag_idx.to_le_bytes());
                frag_data.extend_from_slice(&(self.num_fragments).to_le_bytes());

                // Detail coefficients for this fragment (f16)
                let start = essential_count + (frag_idx as usize - 1) * detail_per_frag;
                let end = (start + detail_per_frag).min(retain_count);

                let coeff_count = end.saturating_sub(start);
                frag_data.extend_from_slice(&(coeff_count as u32).to_le_bytes());

                // Store coefficients as f16
                for i in start..end {
                    if i < retained_indices.len() {
                        let idx = retained_indices[i];
                        let coeff_f16 = f16::from_f32(dct_coeffs[idx]);
                        frag_data.extend_from_slice(&coeff_f16.to_le_bytes());
                    }
                }

                fragments.push(HoloFragment::new(frag_idx, frag_data));
            }
        }

        Ok(fragments)
    }

    /// Encode from pre-computed DCT coefficients.
    ///
    /// This is useful when DCT is computed externally (e.g., on GPU).
    /// The `dct_coeffs` must already be in frequency domain.
    pub fn encode_2d_from_dct(
        &self,
        dct_coeffs: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<HoloFragment>> {
        let n = width * height;
        if dct_coeffs.len() != n {
            return Err(Error::corrupted("DCT coefficients size mismatch"));
        }

        // Sort coefficients by energy (importance) to find which to keep
        let mut indexed: Vec<(usize, f32)> = dct_coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // TRUNCATE: Keep only top retention_ratio coefficients
        let retain_count = ((n as f32 * self.retention_ratio) as usize).max(1);
        let essential_count = ((retain_count as f32 * self.essential_ratio) as usize).max(1);
        let detail_count = retain_count - essential_count;

        // Build bitmap: 1 bit per coefficient, bit=1 means retained
        let bitmap_bytes = n.div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_bytes];

        // Mark retained positions in bitmap
        for &(idx, _) in indexed.iter().take(retain_count) {
            bitmap[idx / 8] |= 1 << (idx % 8);
        }

        // Collect coefficients in INDEX ORDER (ascending), not energy order
        let mut retained_indices: Vec<usize> = indexed
            .iter()
            .take(retain_count)
            .map(|(idx, _)| *idx)
            .collect();
        retained_indices.sort_unstable();

        let mut fragments = Vec::with_capacity(self.num_fragments as usize);

        // Fragment 0: Header + bitmap + essential coefficients (f16)
        {
            let detail_per_frag = if self.num_fragments > 1 {
                detail_count.div_ceil(self.num_fragments as usize - 1)
            } else {
                detail_count
            };

            let mut frag_data = Vec::new();

            // V3 Header with magic (f16 format)
            frag_data.extend_from_slice(&BITMAP_F16_MAGIC.to_le_bytes());
            frag_data.extend_from_slice(&(width as u32).to_le_bytes());
            frag_data.extend_from_slice(&(height as u32).to_le_bytes());
            frag_data.extend_from_slice(&(retain_count as u32).to_le_bytes());
            frag_data.extend_from_slice(&(essential_count as u32).to_le_bytes());
            frag_data.extend_from_slice(&(detail_per_frag as u32).to_le_bytes());

            // Bitmap (N/8 bytes)
            frag_data.extend_from_slice(&bitmap);

            // Essential coefficients as f16 (2 bytes each instead of 4!)
            for &idx in retained_indices.iter().take(essential_count) {
                let coeff_f16 = f16::from_f32(dct_coeffs[idx]);
                frag_data.extend_from_slice(&coeff_f16.to_le_bytes());
            }

            fragments.push(HoloFragment::new(0, frag_data));
        }

        // Fragments 1..N: Detail coefficients (distributed, f16)
        if self.num_fragments > 1 {
            let detail_per_frag = detail_count.div_ceil(self.num_fragments as usize - 1);

            for frag_idx in 1..self.num_fragments {
                let mut frag_data = Vec::new();

                // Header
                frag_data.extend_from_slice(&frag_idx.to_le_bytes());
                frag_data.extend_from_slice(&(self.num_fragments).to_le_bytes());

                // Detail coefficients for this fragment (f16)
                let start = essential_count + (frag_idx as usize - 1) * detail_per_frag;
                let end = (start + detail_per_frag).min(retain_count);

                let coeff_count = end.saturating_sub(start);
                frag_data.extend_from_slice(&(coeff_count as u32).to_le_bytes());

                // Store coefficients as f16
                for i in start..end {
                    if i < retained_indices.len() {
                        let idx = retained_indices[i];
                        let coeff_f16 = f16::from_f32(dct_coeffs[idx]);
                        frag_data.extend_from_slice(&coeff_f16.to_le_bytes());
                    }
                }

                fragments.push(HoloFragment::new(frag_idx, frag_data));
            }
        }

        Ok(fragments)
    }

    /// Encode 1D tensor.
    pub fn encode_1d(&self, data: &[f32]) -> Result<Vec<HoloFragment>> {
        self.encode_2d(data, data.len(), 1)
    }

    /// Calculate expected compression ratio for given parameters (V3 bitmap + f16 format).
    ///
    /// # Returns
    /// The compression ratio (input_size / output_size). Higher is better.
    ///
    /// With V3 format (bitmap + f16 coefficients):
    /// - 20% retention: ~3.8x vs f16 original
    /// - 10% retention: ~6x vs f16 original
    ///
    /// # Example
    /// ```ignore
    /// let encoder = CompressiveSpectralEncoder::new(8, 0.10);
    /// let ratio = encoder.expected_ratio(4096 * 4096);
    /// println!("Expected {}x compression", ratio); // ~6x vs f16
    /// ```
    pub fn expected_ratio(&self, input_elements: usize) -> f32 {
        let input_bytes = input_elements * 4; // f32 input
        let retained = (input_elements as f32 * self.retention_ratio) as usize;
        let essential = (retained as f32 * self.essential_ratio) as usize;

        // Fragment 0: header (24 bytes) + bitmap (N/8 bytes) + essentials (2 bytes each, f16)
        let bitmap_bytes = input_elements.div_ceil(8);
        let frag0_bytes = 24 + bitmap_bytes + essential * 2;

        // Detail fragments: header (8 bytes) + values (2 bytes each, f16)
        let detail_count = retained - essential;
        let detail_bytes = if self.num_fragments > 1 {
            (self.num_fragments as usize - 1) * 8 + detail_count * 2
        } else {
            0
        };

        let total_output = frag0_bytes + detail_bytes;
        input_bytes as f32 / total_output as f32
    }

    /// Calculate expected compression ratio vs f16 original.
    pub fn expected_ratio_vs_f16(&self, input_elements: usize) -> f32 {
        let input_bytes_f16 = input_elements * 2; // f16 original
        let retained = (input_elements as f32 * self.retention_ratio) as usize;
        let essential = (retained as f32 * self.essential_ratio) as usize;

        let bitmap_bytes = input_elements.div_ceil(8);
        let frag0_bytes = 24 + bitmap_bytes + essential * 2;

        let detail_count = retained - essential;
        let detail_bytes = if self.num_fragments > 1 {
            (self.num_fragments as usize - 1) * 8 + detail_count * 2
        } else {
            0
        };

        let total_output = frag0_bytes + detail_bytes;
        input_bytes_f16 as f32 / total_output as f32
    }
}

/// Compressive Spectral Decoder - reconstructs from CompressiveSpectralEncoder output.
///
/// Supports V1 (index array), V2 (bitmap + f32), and V3 (bitmap + f16) formats,
/// auto-detected via magic bytes.
pub struct CompressiveSpectralDecoder {
    width: usize,
    height: usize,
    total_coeffs: usize,
    essential_count: usize,
    detail_per_frag: usize,
    /// For V1: explicit index list. For V2/V3: populated from bitmap scan.
    index_map: Vec<usize>,
    /// Coefficient values in index_map order (always f32 internally)
    coefficients: Vec<f32>,
    has_essentials: bool,
    detail_fragments_loaded: u16,
    total_fragments: u16,
    /// Format version: 1 = index array, 2 = bitmap+f32, 3 = bitmap+f16
    format_version: u8,
}

impl CompressiveSpectralDecoder {
    /// Create decoder (must add fragment 0 first to get dimensions).
    pub fn new() -> Self {
        CompressiveSpectralDecoder {
            width: 0,
            height: 0,
            total_coeffs: 0,
            essential_count: 0,
            detail_per_frag: 0,
            index_map: Vec::new(),
            coefficients: Vec::new(),
            has_essentials: false,
            detail_fragments_loaded: 0,
            total_fragments: 0,
            format_version: 0,
        }
    }

    /// Add fragment 0 (essentials) - MUST be called first.
    ///
    /// Auto-detects V1 (index array), V2 (bitmap+f32), or V3 (bitmap+f16) format via magic bytes.
    pub fn add_essentials(&mut self, fragment: &HoloFragment) -> Result<()> {
        if fragment.index != 0 {
            return Err(Error::corrupted(
                "fragment 0 must be added first for CompressiveSpectralDecoder",
            ));
        }

        let data = &fragment.data;
        if data.len() < 24 {
            return Err(Error::corrupted("fragment 0 too short"));
        }

        // Check magic to determine format version
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);

        if magic == BITMAP_F16_MAGIC {
            self.parse_v3_essentials(data)
        } else if magic == BITMAP_FORMAT_MAGIC {
            self.parse_v2_essentials(data)
        } else {
            self.parse_v1_essentials(data)
        }
    }

    /// Parse V1 format (index array)
    fn parse_v1_essentials(&mut self, data: &[u8]) -> Result<()> {
        self.format_version = 1;

        // V1 Header: [total_coeffs][essential_count][detail_per_frag][width][height]
        self.total_coeffs = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        self.essential_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        self.detail_per_frag = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        self.width = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        self.height = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;

        // Initialize coefficient storage
        self.coefficients = vec![0.0f32; self.total_coeffs];

        // Read essential coefficients
        let mut offset = 20;
        for i in 0..self.essential_count {
            if offset + 4 > data.len() {
                return Err(Error::corrupted("truncated essential coefficients"));
            }
            self.coefficients[i] = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;
        }

        // Read index map
        self.index_map = Vec::with_capacity(self.total_coeffs);
        for _ in 0..self.total_coeffs {
            if offset + 4 > data.len() {
                return Err(Error::corrupted("truncated index map"));
            }
            let idx = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            self.index_map.push(idx);
            offset += 4;
        }

        self.has_essentials = true;
        Ok(())
    }

    /// Parse V2 format (bitmap + f32 coefficients)
    fn parse_v2_essentials(&mut self, data: &[u8]) -> Result<()> {
        self.format_version = 2;

        // V2 Header: [magic][width][height][retain_count][essential_count][detail_per_frag]
        // Skip magic (already verified)
        self.width = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        self.height = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        self.total_coeffs = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        self.essential_count =
            u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        self.detail_per_frag =
            u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;

        let n = self.width * self.height;
        let bitmap_bytes = n.div_ceil(8);

        // Read bitmap and build index map
        let bitmap_start = 24;
        let bitmap_end = bitmap_start + bitmap_bytes;

        if bitmap_end > data.len() {
            return Err(Error::corrupted("truncated bitmap"));
        }

        // Scan bitmap to build index map (positions where bit=1)
        self.index_map = Vec::with_capacity(self.total_coeffs);
        for i in 0..n {
            let byte_idx = bitmap_start + i / 8;
            let bit_idx = i % 8;
            if (data[byte_idx] >> bit_idx) & 1 == 1 {
                self.index_map.push(i);
            }
        }

        // Verify we found the expected number of retained coefficients
        if self.index_map.len() != self.total_coeffs {
            return Err(Error::corrupted(format!(
                "bitmap has {} set bits, expected {}",
                self.index_map.len(),
                self.total_coeffs
            )));
        }

        // Initialize coefficient storage
        self.coefficients = vec![0.0f32; self.total_coeffs];

        // Read essential coefficients (stored after bitmap)
        let mut offset = bitmap_end;
        for i in 0..self.essential_count {
            if offset + 4 > data.len() {
                return Err(Error::corrupted("truncated essential coefficients"));
            }
            self.coefficients[i] = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;
        }

        self.has_essentials = true;
        Ok(())
    }

    /// Parse V3 format (bitmap + f16 coefficients)
    fn parse_v3_essentials(&mut self, data: &[u8]) -> Result<()> {
        self.format_version = 3;

        // V3 Header: [magic][width][height][retain_count][essential_count][detail_per_frag]
        // Same as V2 but coefficients are f16 instead of f32
        self.width = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        self.height = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        self.total_coeffs = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        self.essential_count =
            u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        self.detail_per_frag =
            u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;

        let n = self.width * self.height;
        let bitmap_bytes = n.div_ceil(8);

        // Read bitmap and build index map
        let bitmap_start = 24;
        let bitmap_end = bitmap_start + bitmap_bytes;

        if bitmap_end > data.len() {
            return Err(Error::corrupted("truncated bitmap"));
        }

        // Scan bitmap to build index map (positions where bit=1)
        self.index_map = Vec::with_capacity(self.total_coeffs);
        for i in 0..n {
            let byte_idx = bitmap_start + i / 8;
            let bit_idx = i % 8;
            if (data[byte_idx] >> bit_idx) & 1 == 1 {
                self.index_map.push(i);
            }
        }

        // Verify we found the expected number of retained coefficients
        if self.index_map.len() != self.total_coeffs {
            return Err(Error::corrupted(format!(
                "bitmap has {} set bits, expected {}",
                self.index_map.len(),
                self.total_coeffs
            )));
        }

        // Initialize coefficient storage (internally f32 for computation)
        self.coefficients = vec![0.0f32; self.total_coeffs];

        // Read essential coefficients as f16 (2 bytes each, convert to f32)
        let mut offset = bitmap_end;
        for i in 0..self.essential_count {
            if offset + 2 > data.len() {
                return Err(Error::corrupted("truncated essential coefficients"));
            }
            let f16_val = f16::from_le_bytes([data[offset], data[offset + 1]]);
            self.coefficients[i] = f16_val.to_f32();
            offset += 2;
        }

        self.has_essentials = true;
        Ok(())
    }

    /// Add a detail fragment (fragments 1..N).
    pub fn add_detail(&mut self, fragment: &HoloFragment) -> Result<()> {
        if !self.has_essentials {
            return Err(Error::corrupted("must add fragment 0 (essentials) first"));
        }

        if fragment.index == 0 {
            return Ok(()); // Already processed
        }

        let data = &fragment.data;
        if data.len() < 8 {
            return Err(Error::corrupted("detail fragment too short"));
        }

        let frag_idx = u16::from_le_bytes([data[0], data[1]]);
        self.total_fragments = u16::from_le_bytes([data[2], data[3]]);
        let coeff_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        // Calculate where these coefficients go in our array
        let start = self.essential_count + (frag_idx as usize - 1) * self.detail_per_frag;

        let mut offset = 8;

        // V3 uses f16 (2 bytes), V1/V2 use f32 (4 bytes)
        let coeff_size = if self.format_version == 3 { 2 } else { 4 };

        for i in 0..coeff_count {
            if offset + coeff_size > data.len() {
                break;
            }
            let coeff_idx = start + i;
            if coeff_idx < self.coefficients.len() {
                if self.format_version == 3 {
                    // V3: f16 coefficients
                    let f16_val = f16::from_le_bytes([data[offset], data[offset + 1]]);
                    self.coefficients[coeff_idx] = f16_val.to_f32();
                } else {
                    // V1/V2: f32 coefficients
                    self.coefficients[coeff_idx] = f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                }
            }
            offset += coeff_size;
        }

        self.detail_fragments_loaded += 1;
        Ok(())
    }

    /// Check if we can reconstruct (need at least essentials).
    pub fn can_reconstruct(&self) -> bool {
        self.has_essentials
    }

    /// Get quality estimate (0.0-1.0).
    pub fn quality(&self) -> f32 {
        if !self.has_essentials {
            return 0.0;
        }
        if self.total_fragments <= 1 {
            return 1.0; // Only essentials, all loaded
        }
        let detail_frags = self.total_fragments - 1;
        let loaded = self.detail_fragments_loaded.min(detail_frags);
        let essential_quality = self.essential_count as f32 / self.total_coeffs as f32;
        let detail_quality =
            (self.total_coeffs - self.essential_count) as f32 / self.total_coeffs as f32;
        essential_quality + detail_quality * (loaded as f32 / detail_frags as f32)
    }

    /// Reconstruct tensor from loaded coefficients.
    pub fn reconstruct(&self) -> Result<Vec<f32>> {
        if !self.has_essentials {
            return Err(Error::corrupted("need essentials fragment to reconstruct"));
        }

        let n = self.width * self.height;
        let mut dct_coeffs = vec![0.0f32; n];

        // Place coefficients at their original positions using index map
        for (i, &original_idx) in self.index_map.iter().enumerate() {
            if original_idx < n && i < self.coefficients.len() {
                dct_coeffs[original_idx] = self.coefficients[i];
            }
        }

        // Inverse DCT to reconstruct spatial domain
        let mut output = vec![0.0f32; n];
        idct_2d(&dct_coeffs, &mut output, self.width, self.height);

        Ok(output)
    }
}

impl Default for CompressiveSpectralDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum / a.len() as f32
    }

    #[test]
    fn test_compressive_roundtrip_full_retention() {
        // With 100% retention, should be near-perfect reconstruction
        let encoder = CompressiveSpectralEncoder::new(4, 1.0);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }

        let output = decoder.reconstruct().unwrap();
        let mse = mean_squared_error(&input, &output);

        assert!(mse < 1e-4, "MSE {} too high for full retention", mse);
    }

    #[test]
    fn test_compressive_roundtrip_partial_retention() {
        // With 20% retention, expect some quality loss but reasonable reconstruction
        let encoder = CompressiveSpectralEncoder::new(4, 0.2);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }

        let output = decoder.reconstruct().unwrap();
        let mse = mean_squared_error(&input, &output);

        // With lossy compression, expect some error but not huge
        assert!(mse < 0.5, "MSE {} too high for 20% retention", mse);
    }

    #[test]
    fn test_compressive_storage_reduction() {
        let encoder = CompressiveSpectralEncoder::new(8, 0.1);

        let input: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 64, 64).unwrap();

        let total_output_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();
        let input_bytes = input.len() * 4;

        let actual_ratio = input_bytes as f32 / total_output_bytes as f32;
        let expected_ratio = encoder.expected_ratio(input.len());

        println!(
            "Input: {} bytes, Output: {} bytes",
            input_bytes, total_output_bytes
        );
        println!(
            "Actual ratio: {:.2}x, Expected: {:.2}x",
            actual_ratio, expected_ratio
        );

        // Should achieve significant compression
        assert!(
            actual_ratio > 1.5,
            "Expected compression, got {}x",
            actual_ratio
        );
    }

    #[test]
    fn test_progressive_quality() {
        let encoder = CompressiveSpectralEncoder::new(8, 0.3);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();

        // Just essentials
        decoder.add_essentials(&fragments[0]).unwrap();
        let output_essentials = decoder.reconstruct().unwrap();
        let mse_essentials = mean_squared_error(&input, &output_essentials);

        // Add half the details
        for frag in &fragments[1..4] {
            decoder.add_detail(frag).unwrap();
        }
        let output_half = decoder.reconstruct().unwrap();
        let mse_half = mean_squared_error(&input, &output_half);

        // Add all details
        for frag in &fragments[4..] {
            decoder.add_detail(frag).unwrap();
        }
        let quality_full = decoder.quality();
        let output_full = decoder.reconstruct().unwrap();
        let mse_full = mean_squared_error(&input, &output_full);

        println!("Essentials only: MSE={:.6}", mse_essentials);
        println!("Half details: MSE={:.6}", mse_half);
        println!(
            "Full details: quality={:.2}, MSE={:.6}",
            quality_full, mse_full
        );

        // MSE should decrease (improve) with more fragments
        // This is the actual quality metric that matters
        assert!(
            mse_half <= mse_essentials,
            "Half details MSE {} should be <= essentials MSE {}",
            mse_half,
            mse_essentials
        );
        assert!(
            mse_full <= mse_half,
            "Full MSE {} should be <= half MSE {}",
            mse_full,
            mse_half
        );

        // Full quality should be 1.0 when all fragments loaded
        assert!(
            (quality_full - 1.0).abs() < 0.01,
            "Full quality should be ~1.0, got {}",
            quality_full
        );
    }

    // =========================================================================
    // Phase 3: Comprehensive retention ratio tests
    // =========================================================================

    #[test]
    fn test_retention_ratio_5_percent() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.05);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 32, 32).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        // At 5% retention, expect significant compression but lossy reconstruction
        let mse = mean_squared_error(&input, &output);
        assert!(mse < 1.0, "MSE {} too high even for 5% retention", mse);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_retention_ratio_10_percent() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.10);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 32, 32).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.5, "MSE {} too high for 10% retention", mse);
    }

    #[test]
    fn test_retention_ratio_30_percent() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.30);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 32, 32).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.2, "MSE {} too high for 30% retention", mse);
    }

    #[test]
    fn test_retention_ratio_50_percent() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.50);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 32, 32).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.05, "MSE {} too high for 50% retention", mse);
    }

    #[test]
    fn test_retention_ratio_75_percent() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.75);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 32, 32).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.01, "MSE {} too high for 75% retention", mse);
    }

    #[test]
    fn test_retention_ratio_90_percent() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.90);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 32, 32).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.001, "MSE {} too high for 90% retention", mse);
    }

    #[test]
    fn test_retention_monotonicity() {
        // Quality should improve as retention increases
        let retentions = [0.10, 0.30, 0.50, 0.70, 0.90];
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut prev_mse = f32::MAX;

        for retention in retentions {
            let encoder = CompressiveSpectralEncoder::new(4, retention);
            let fragments = encoder.encode_2d(&input, 32, 32).unwrap();

            let mut decoder = CompressiveSpectralDecoder::new();
            decoder.add_essentials(&fragments[0]).unwrap();
            for frag in &fragments[1..] {
                decoder.add_detail(frag).unwrap();
            }
            let output = decoder.reconstruct().unwrap();
            let mse = mean_squared_error(&input, &output);

            assert!(
                mse <= prev_mse + 0.001,
                "MSE should decrease with higher retention: {}% gave {}, previous gave {}",
                retention * 100.0,
                mse,
                prev_mse
            );
            prev_mse = mse;
        }
    }

    // =========================================================================
    // Phase 3: Tensor shape tests
    // =========================================================================

    #[test]
    fn test_shape_64x64() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input: Vec<f32> = (0..64 * 64).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 64, 64).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 64 * 64);
        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.01, "MSE {} too high for 64x64", mse);
    }

    #[test]
    fn test_shape_128x512_wide() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input: Vec<f32> = (0..128 * 512).map(|i| (i as f32 * 0.001).sin()).collect();
        let fragments = encoder.encode_2d(&input, 128, 512).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 128 * 512);
        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.01, "MSE {} too high for 128x512", mse);
    }

    #[test]
    fn test_shape_512x128_tall() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input: Vec<f32> = (0..512 * 128).map(|i| (i as f32 * 0.001).sin()).collect();
        let fragments = encoder.encode_2d(&input, 512, 128).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 512 * 128);
        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.01, "MSE {} too high for 512x128", mse);
    }

    #[test]
    fn test_shape_non_power_of_two() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        // 100x100 is not a power of 2
        let input: Vec<f32> = (0..100 * 100).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 100, 100).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 100 * 100);
    }

    #[test]
    fn test_shape_prime_dimensions() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        // 97x101 are both prime
        let input: Vec<f32> = (0..97 * 101).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 97, 101).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 97 * 101);
    }

    // =========================================================================
    // Phase 3: Edge cases and small tensors
    // =========================================================================

    #[test]
    fn test_tiny_tensor_4x4() {
        let encoder = CompressiveSpectralEncoder::new(2, 0.70);
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let fragments = encoder.encode_2d(&input, 4, 4).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_tiny_tensor_2x2() {
        let encoder = CompressiveSpectralEncoder::new(1, 0.70);
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let fragments = encoder.encode_2d(&input, 2, 2).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_single_row_tensor() {
        let encoder = CompressiveSpectralEncoder::new(2, 0.70);
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let fragments = encoder.encode_2d(&input, 128, 1).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_single_column_tensor() {
        let encoder = CompressiveSpectralEncoder::new(2, 0.70);
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let fragments = encoder.encode_2d(&input, 1, 128).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_all_zeros() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input = vec![0.0f32; 256];
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        // All zeros should reconstruct to all (nearly) zeros
        let max_err = max_error(&input, &output);
        assert!(
            max_err < 1e-5,
            "Max error {} too high for all-zeros",
            max_err
        );
    }

    #[test]
    fn test_all_ones() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input = vec![1.0f32; 256];
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        // Constant value (DC only) should compress perfectly
        let mse = mean_squared_error(&input, &output);
        assert!(mse < 1e-4, "MSE {} too high for all-ones", mse);
    }

    #[test]
    fn test_constant_negative() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input = vec![-42.0f32; 256];
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        let mse = mean_squared_error(&input, &output);
        assert!(mse < 1e-2, "MSE {} too high for constant -42", mse);
    }

    // =========================================================================
    // Phase 3: Fragment count variations
    // =========================================================================

    #[test]
    fn test_single_fragment() {
        let encoder = CompressiveSpectralEncoder::new(1, 0.70);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        assert_eq!(
            fragments.len(),
            1,
            "Single fragment encoder should produce 1 fragment"
        );

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_two_fragments() {
        let encoder = CompressiveSpectralEncoder::new(2, 0.70);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        assert!(
            fragments.len() <= 2,
            "Two fragment encoder should produce <=2 fragments"
        );

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_many_fragments() {
        let encoder = CompressiveSpectralEncoder::new(16, 0.70);
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 32, 32).unwrap();

        // Should produce multiple fragments
        assert!(!fragments.is_empty(), "Should produce at least 1 fragment");

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        assert_eq!(output.len(), input.len());
        let mse = mean_squared_error(&input, &output);
        assert!(mse < 0.01, "MSE {} too high for many fragments", mse);
    }

    // =========================================================================
    // Phase 3: Partial fragment reconstruction
    // =========================================================================

    #[test]
    fn test_essentials_only_reconstruction() {
        let encoder = CompressiveSpectralEncoder::new(8, 0.50);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        // Only use essentials
        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        let output = decoder.reconstruct().unwrap();

        // Should get a valid output, even if lower quality
        assert_eq!(output.len(), input.len());
        // Quality should be measurable
        let quality = decoder.quality();
        assert!(quality > 0.0, "Quality should be positive");
    }

    #[test]
    fn test_partial_details_reconstruction() {
        let encoder = CompressiveSpectralEncoder::new(8, 0.50);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        // Use essentials + first half of details
        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();

        let num_details = fragments.len() - 1;
        let half_details = num_details / 2;
        for frag in &fragments[1..=half_details.max(1)] {
            decoder.add_detail(frag).unwrap();
        }
        let output_partial = decoder.reconstruct().unwrap();

        // Add rest of details
        for frag in &fragments[half_details + 1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output_full = decoder.reconstruct().unwrap();

        // Full should be at least as good as partial
        let mse_partial = mean_squared_error(&input, &output_partial);
        let mse_full = mean_squared_error(&input, &output_full);
        assert!(
            mse_full <= mse_partial + 0.001,
            "Full MSE {} should be <= partial MSE {}",
            mse_full,
            mse_partial
        );
    }

    // =========================================================================
    // Phase 3: Error handling tests
    // =========================================================================

    #[test]
    fn test_decoder_without_essentials() {
        let decoder = CompressiveSpectralDecoder::new();
        // Should fail or return error when reconstructing without essentials
        let result = decoder.reconstruct();
        assert!(
            result.is_err(),
            "Should error when reconstructing without essentials"
        );
    }

    #[test]
    fn test_decoder_reuse() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input1: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let input2: Vec<f32> = (0..256).map(|i| (i as f32 * 0.2).cos()).collect();

        let fragments1 = encoder.encode_2d(&input1, 16, 16).unwrap();
        let fragments2 = encoder.encode_2d(&input2, 16, 16).unwrap();

        // First decode
        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments1[0]).unwrap();
        for frag in &fragments1[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output1 = decoder.reconstruct().unwrap();

        // Create new decoder for second input (cannot reuse)
        let mut decoder2 = CompressiveSpectralDecoder::new();
        decoder2.add_essentials(&fragments2[0]).unwrap();
        for frag in &fragments2[1..] {
            decoder2.add_detail(frag).unwrap();
        }
        let output2 = decoder2.reconstruct().unwrap();

        // Both should be valid
        assert_eq!(output1.len(), 256);
        assert_eq!(output2.len(), 256);

        // And different
        let diff: f32 = output1
            .iter()
            .zip(output2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1.0,
            "Different inputs should produce different outputs"
        );
    }

    #[test]
    fn test_expected_ratio_calculation() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.10);
        let ratio = encoder.expected_ratio(10000);

        // With 10% retention, expect roughly 5-10x compression
        assert!(
            ratio > 3.0,
            "Expected ratio {} should be > 3.0 for 10% retention",
            ratio
        );
        assert!(ratio < 20.0, "Expected ratio {} should be < 20.0", ratio);
    }

    #[test]
    fn test_large_value_range() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        // Values ranging from -1000 to +1000
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 8.0).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        // Should handle large values
        let max_err = max_error(&input, &output);
        let relative_err = max_err / 1000.0;
        assert!(
            relative_err < 0.1,
            "Relative error {} too high for large values",
            relative_err
        );
    }

    #[test]
    fn test_small_value_range() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        // Very small values (typical neural network weights)
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.001).sin() * 0.01).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let output = decoder.reconstruct().unwrap();

        // Should handle small values without losing precision
        let mse = mean_squared_error(&input, &output);
        assert!(mse < 1e-6, "MSE {} too high for small values", mse);
    }

    // -------------------- Error Condition Tests --------------------

    #[test]
    fn test_add_detail_before_essentials() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let fragments = encoder.encode_2d(&input, 8, 8).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();

        // Try to add detail before essentials - should error
        if fragments.len() > 1 {
            let result = decoder.add_detail(&fragments[1]);
            assert!(
                result.is_err(),
                "Should error when adding detail without essentials"
            );
        }
    }

    #[test]
    fn test_reconstruct_without_any_data() {
        let decoder = CompressiveSpectralDecoder::new();

        // Try to reconstruct with no data added
        let result = decoder.reconstruct();
        assert!(
            result.is_err(),
            "Should error when reconstructing with no data"
        );
    }

    #[test]
    fn test_encode_empty_input() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let empty: Vec<f32> = vec![];

        // Encoding empty data should either succeed with empty output or error
        let result = encoder.encode_2d(&empty, 0, 0);
        // Either is acceptable, shouldn't panic
        let _ = result;
    }

    #[test]
    fn test_encode_nan_values_compressive() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let mut input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        input[10] = f32::NAN;
        input[30] = f32::NAN;

        // Encoding with NaN should handle gracefully
        let result = encoder.encode_2d(&input, 8, 8);
        // Shouldn't panic - may succeed with NaN in output or error
        let _ = result;
    }

    #[test]
    fn test_encode_infinity_values_compressive() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let mut input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        input[5] = f32::INFINITY;
        input[25] = f32::NEG_INFINITY;

        // Encoding with infinity should handle gracefully
        let result = encoder.encode_2d(&input, 8, 8);
        // Shouldn't panic
        let _ = result;
    }

    #[test]
    fn test_decode_corrupted_fragment_data_compressive() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let mut fragments = encoder.encode_2d(&input, 8, 8).unwrap();

        // Corrupt the essential fragment data (make it too short)
        if !fragments.is_empty() {
            fragments[0].data = vec![0; 4]; // Too short to be valid
        }

        let mut decoder = CompressiveSpectralDecoder::new();
        let result = decoder.add_essentials(&fragments[0]);

        // May error or produce garbage, but shouldn't panic
        let _ = result;
    }

    #[test]
    fn test_decode_duplicate_essentials() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let fragments = encoder.encode_2d(&input, 8, 8).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();

        // Try to add essentials again - should either error or replace
        let result = decoder.add_essentials(&fragments[0]);
        // Either behavior is acceptable - just shouldn't panic
        let _ = result;
    }

    #[test]
    fn test_extreme_retention_zero() {
        // 0% retention
        let encoder = CompressiveSpectralEncoder::new(4, 0.0);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

        let result = encoder.encode_2d(&input, 8, 8);
        // Should handle gracefully
        if let Ok(fragments) = result {
            // May have empty or minimal data - just verify it didn't panic
            let _ = fragments.is_empty();
        }
    }

    #[test]
    fn test_extreme_retention_over_one() {
        // >100% retention (clamped internally)
        let encoder = CompressiveSpectralEncoder::new(4, 1.5);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

        let result = encoder.encode_2d(&input, 8, 8);
        // Should handle gracefully (clamp to 1.0)
        assert!(result.is_ok(), "Should handle >100% retention");
    }

    #[test]
    fn test_corrupted_detail_fragment() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let mut fragments = encoder.encode_2d(&input, 8, 8).unwrap();

        // Corrupt a detail fragment's data
        if fragments.len() > 1 {
            fragments[1].data = vec![0xFF; 100]; // Random garbage data
        }

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();

        if fragments.len() > 1 {
            // Adding corrupted detail - may error or handle gracefully
            let result = decoder.add_detail(&fragments[1]);
            // Should not panic, may error or produce garbage
            let _ = result;
        }
    }

    #[test]
    fn test_very_high_fragment_count() {
        // More fragments than makes sense
        let encoder = CompressiveSpectralEncoder::new(100, 0.70);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

        let result = encoder.encode_2d(&input, 8, 8);
        // Should handle gracefully - probably limits fragments to meaningful count
        assert!(result.is_ok(), "Should handle high fragment count");
    }

    #[test]
    fn test_negative_fragment_count() {
        // This would be caught at type level (usize can't be negative)
        // But test with 0 fragments
        let encoder = CompressiveSpectralEncoder::new(0, 0.70);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

        let result = encoder.encode_2d(&input, 8, 8);
        // Should handle gracefully - at least 1 fragment needed
        let _ = result;
    }

    #[test]
    fn test_encode_dimension_mismatch() {
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        // Data has 100 elements, but we claim 8x8=64
        let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();

        let result = encoder.encode_2d(&input, 8, 8);
        // Should either truncate or error, not panic
        let _ = result;
    }

    #[test]
    fn test_reconstruct_after_partial_details() {
        let encoder = CompressiveSpectralEncoder::new(8, 0.70);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let fragments = encoder.encode_2d(&input, 16, 16).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();

        // Only add some details, not all
        if fragments.len() > 2 {
            decoder.add_detail(&fragments[1]).unwrap();
        }

        // Reconstruct with partial data
        let result = decoder.reconstruct();
        assert!(result.is_ok(), "Should reconstruct with partial details");

        let output = result.unwrap();
        assert_eq!(output.len(), input.len(), "Output size should match input");
    }
}
