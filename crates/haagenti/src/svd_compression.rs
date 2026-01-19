//! SVD-based Compression for Neural Network Weights
//!
//! This module implements low-rank approximation using Singular Value Decomposition (SVD)
//! as an alternative to spectral (DCT) compression for certain tensor types.
//!
//! ## When to Use SVD vs DCT
//!
//! | Tensor Type | Recommended | Reason |
//! |-------------|-------------|--------|
//! | Attention Q/K/V/O | SVD | Natural low-rank structure |
//! | MLP/FFN | DCT | Better for dense, full-rank weights |
//! | Embeddings | DCT | High-frequency patterns |
//! | LayerNorm | None | Keep at full precision |
//!
//! ## Algorithm
//!
//! SVD decomposes matrix A (m×n) into: A = U × S × V^T
//! - U: Left singular vectors (m×k)
//! - S: Singular values (k diagonal)
//! - V^T: Right singular vectors (k×n)
//!
//! For compression, we keep only the top-k singular values (rank-k approximation).
//!
//! ## Usage
//!
//! ```ignore
//! use haagenti::svd_compression::{SvdEncoder, SvdDecoder, SvdCompressedWeight};
//!
//! let encoder = SvdEncoder::new(0.5); // 50% rank retention
//! let compressed = encoder.compress(&weight_matrix, rows, cols)?;
//!
//! let decoder = SvdDecoder::new();
//! let reconstructed = decoder.decompress(&compressed)?;
//! ```

use haagenti_core::{Error, Result};

/// Compressed weight stored as low-rank SVD factors.
///
/// Storage: U (m×k) + S (k) + Vt (k×n) = k×(m+n+1) floats
/// vs original: m×n floats
///
/// Compression ratio: k×(m+n+1) / (m×n) ≈ k/min(m,n) for square-ish matrices
#[derive(Debug, Clone)]
pub struct SvdCompressedWeight {
    /// Original matrix dimensions
    pub rows: usize,
    pub cols: usize,
    /// Rank of the approximation (number of singular values kept)
    pub rank: usize,
    /// Left singular vectors U, stored row-major as (rows × rank)
    pub u: Vec<f32>,
    /// Singular values S (rank elements)
    pub s: Vec<f32>,
    /// Right singular vectors V^T, stored row-major as (rank × cols)
    pub vt: Vec<f32>,
}

impl SvdCompressedWeight {
    /// Calculate storage size in bytes.
    pub fn storage_bytes(&self) -> usize {
        (self.u.len() + self.s.len() + self.vt.len()) * 4
    }

    /// Calculate original size in bytes.
    pub fn original_bytes(&self) -> usize {
        self.rows * self.cols * 4
    }

    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        self.original_bytes() as f32 / self.storage_bytes() as f32
    }

    /// Calculate rank retention ratio.
    pub fn rank_ratio(&self) -> f32 {
        self.rank as f32 / self.rows.min(self.cols) as f32
    }
}

/// SVD Encoder for compressing weight matrices.
///
/// Uses randomized SVD for efficiency with large matrices.
#[derive(Debug, Clone)]
pub struct SvdEncoder {
    /// Target rank as fraction of min(rows, cols)
    rank_ratio: f32,
    /// Minimum rank to keep
    min_rank: usize,
    /// Maximum rank to keep
    max_rank: usize,
    /// Oversampling factor for randomized SVD
    oversampling: usize,
    /// Number of power iterations for accuracy
    power_iterations: usize,
}

impl Default for SvdEncoder {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl SvdEncoder {
    /// Create encoder with target rank ratio.
    ///
    /// # Arguments
    /// * `rank_ratio` - Fraction of singular values to keep (0.0-1.0)
    pub fn new(rank_ratio: f32) -> Self {
        Self {
            rank_ratio: rank_ratio.clamp(0.01, 1.0),
            min_rank: 1,
            max_rank: 4096,
            oversampling: 10,
            power_iterations: 2,
        }
    }

    /// Set minimum rank.
    pub fn with_min_rank(mut self, min: usize) -> Self {
        self.min_rank = min.max(1);
        self
    }

    /// Set maximum rank.
    pub fn with_max_rank(mut self, max: usize) -> Self {
        self.max_rank = max;
        self
    }

    /// Set oversampling factor for randomized SVD.
    pub fn with_oversampling(mut self, oversample: usize) -> Self {
        self.oversampling = oversample;
        self
    }

    /// Set number of power iterations.
    pub fn with_power_iterations(mut self, iters: usize) -> Self {
        self.power_iterations = iters;
        self
    }

    /// Compress a matrix using truncated SVD.
    ///
    /// # Arguments
    /// * `data` - Matrix data in row-major order
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    pub fn compress(&self, data: &[f32], rows: usize, cols: usize) -> Result<SvdCompressedWeight> {
        if data.len() != rows * cols {
            return Err(Error::corrupted("data size mismatch"));
        }

        if rows == 0 || cols == 0 {
            return Err(Error::corrupted("empty matrix"));
        }

        // Calculate target rank
        let max_rank = rows.min(cols);
        let target_rank = ((max_rank as f32 * self.rank_ratio) as usize)
            .clamp(self.min_rank, self.max_rank.min(max_rank));

        // For small matrices, use direct SVD
        if rows * cols < 10000 || target_rank > max_rank / 2 {
            self.direct_svd(data, rows, cols, target_rank)
        } else {
            // For large matrices, use randomized SVD
            self.randomized_svd(data, rows, cols, target_rank)
        }
    }

    /// Direct SVD using power iteration method.
    ///
    /// This is a simplified implementation suitable for moderate-sized matrices.
    fn direct_svd(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
        target_rank: usize,
    ) -> Result<SvdCompressedWeight> {
        // For direct SVD, we use iterative deflation
        let mut a = data.to_vec();
        let mut u_vecs: Vec<Vec<f32>> = Vec::with_capacity(target_rank);
        let mut s_vals: Vec<f32> = Vec::with_capacity(target_rank);
        let mut vt_vecs: Vec<Vec<f32>> = Vec::with_capacity(target_rank);

        for _ in 0..target_rank {
            // Find dominant singular triplet using power iteration
            let (u, sigma, v) = self.power_iteration(&a, rows, cols, 50)?;

            if sigma < 1e-10 {
                break; // Remaining singular values are negligible
            }

            // Store the triplet
            u_vecs.push(u.clone());
            s_vals.push(sigma);
            vt_vecs.push(v.clone());

            // Deflate: A = A - sigma * u * v^T
            for i in 0..rows {
                for j in 0..cols {
                    a[i * cols + j] -= sigma * u[i] * v[j];
                }
            }
        }

        let rank = s_vals.len();

        // Flatten U: rows × rank
        let mut u = Vec::with_capacity(rows * rank);
        for i in 0..rows {
            for k in 0..rank {
                u.push(u_vecs[k][i]);
            }
        }

        // Flatten Vt: rank × cols
        let mut vt = Vec::with_capacity(rank * cols);
        for k in 0..rank {
            vt.extend_from_slice(&vt_vecs[k]);
        }

        Ok(SvdCompressedWeight {
            rows,
            cols,
            rank,
            u,
            s: s_vals,
            vt,
        })
    }

    /// Randomized SVD for large matrices.
    ///
    /// Algorithm:
    /// 1. Create random projection matrix Ω (cols × (rank + oversampling))
    /// 2. Form Y = A × Ω
    /// 3. Orthogonalize Y to get Q
    /// 4. Form B = Q^T × A
    /// 5. SVD of smaller matrix B
    /// 6. Recover U = Q × U_B
    fn randomized_svd(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
        target_rank: usize,
    ) -> Result<SvdCompressedWeight> {
        let sketch_size = (target_rank + self.oversampling).min(cols);

        // Step 1: Random projection matrix (using simple PRNG for reproducibility)
        let mut omega = vec![0.0f32; cols * sketch_size];
        let mut rng_state = 42u64;
        for val in &mut omega {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
        }

        // Step 2: Y = A × Ω (rows × sketch_size)
        let mut y = vec![0.0f32; rows * sketch_size];
        for i in 0..rows {
            for k in 0..sketch_size {
                let mut sum = 0.0f32;
                for j in 0..cols {
                    sum += data[i * cols + j] * omega[j * sketch_size + k];
                }
                y[i * sketch_size + k] = sum;
            }
        }

        // Power iterations for better accuracy
        for _ in 0..self.power_iterations {
            // Y = A × A^T × Y
            let mut temp = vec![0.0f32; cols * sketch_size];
            // temp = A^T × Y
            for j in 0..cols {
                for k in 0..sketch_size {
                    let mut sum = 0.0f32;
                    for i in 0..rows {
                        sum += data[i * cols + j] * y[i * sketch_size + k];
                    }
                    temp[j * sketch_size + k] = sum;
                }
            }
            // Y = A × temp
            for i in 0..rows {
                for k in 0..sketch_size {
                    let mut sum = 0.0f32;
                    for j in 0..cols {
                        sum += data[i * cols + j] * temp[j * sketch_size + k];
                    }
                    y[i * sketch_size + k] = sum;
                }
            }
        }

        // Step 3: QR decomposition of Y to get orthonormal Q
        let q = self.qr_q(&y, rows, sketch_size)?;

        // Step 4: B = Q^T × A (sketch_size × cols)
        let mut b = vec![0.0f32; sketch_size * cols];
        for k in 0..sketch_size {
            for j in 0..cols {
                let mut sum = 0.0f32;
                for i in 0..rows {
                    sum += q[i * sketch_size + k] * data[i * cols + j];
                }
                b[k * cols + j] = sum;
            }
        }

        // Step 5: SVD of B (smaller matrix)
        let b_svd = self.direct_svd(&b, sketch_size, cols, target_rank)?;

        // Step 6: U = Q × U_B
        let mut u = vec![0.0f32; rows * b_svd.rank];
        for i in 0..rows {
            for r in 0..b_svd.rank {
                let mut sum = 0.0f32;
                for k in 0..sketch_size {
                    sum += q[i * sketch_size + k] * b_svd.u[k * b_svd.rank + r];
                }
                u[i * b_svd.rank + r] = sum;
            }
        }

        Ok(SvdCompressedWeight {
            rows,
            cols,
            rank: b_svd.rank,
            u,
            s: b_svd.s,
            vt: b_svd.vt,
        })
    }

    /// Power iteration to find dominant singular triplet.
    fn power_iteration(
        &self,
        a: &[f32],
        rows: usize,
        cols: usize,
        max_iters: usize,
    ) -> Result<(Vec<f32>, f32, Vec<f32>)> {
        // Initialize v randomly
        let mut v = vec![0.0f32; cols];
        let mut rng_state = 12345u64;
        for val in &mut v {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
        }
        normalize(&mut v);

        let mut u = vec![0.0f32; rows];

        for _ in 0..max_iters {
            // u = A × v
            for i in 0..rows {
                let mut sum = 0.0f32;
                for j in 0..cols {
                    sum += a[i * cols + j] * v[j];
                }
                u[i] = sum;
            }
            normalize(&mut u);

            // v = A^T × u
            for j in 0..cols {
                let mut sum = 0.0f32;
                for i in 0..rows {
                    sum += a[i * cols + j] * u[i];
                }
                v[j] = sum;
            }
            normalize(&mut v);
        }

        // Compute sigma = u^T × A × v
        let mut sigma = 0.0f32;
        for i in 0..rows {
            let mut row_sum = 0.0f32;
            for j in 0..cols {
                row_sum += a[i * cols + j] * v[j];
            }
            sigma += u[i] * row_sum;
        }

        Ok((u, sigma, v))
    }

    /// QR decomposition, returns Q matrix only.
    fn qr_q(&self, a: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let mut q = a.to_vec();

        // Modified Gram-Schmidt
        for j in 0..cols {
            // Normalize column j
            let mut norm = 0.0f32;
            for i in 0..rows {
                norm += q[i * cols + j] * q[i * cols + j];
            }
            norm = norm.sqrt();

            if norm > 1e-10 {
                for i in 0..rows {
                    q[i * cols + j] /= norm;
                }
            }

            // Orthogonalize remaining columns against column j
            for k in (j + 1)..cols {
                let mut dot = 0.0f32;
                for i in 0..rows {
                    dot += q[i * cols + j] * q[i * cols + k];
                }
                for i in 0..rows {
                    q[i * cols + k] -= dot * q[i * cols + j];
                }
            }
        }

        Ok(q)
    }

    /// Analyze a matrix to determine optimal rank.
    ///
    /// Returns the rank needed to capture the target energy fraction.
    pub fn analyze_rank(&self, data: &[f32], rows: usize, cols: usize, target_energy: f32) -> Result<usize> {
        // Do a quick SVD to get singular values
        let max_rank = rows.min(cols).min(100); // Limit analysis to first 100 singular values
        let svd = self.direct_svd(data, rows, cols, max_rank)?;

        // Compute energy (sum of squared singular values)
        let total_energy: f32 = svd.s.iter().map(|s| s * s).sum();
        let mut cumulative = 0.0f32;

        for (i, &s) in svd.s.iter().enumerate() {
            cumulative += s * s;
            if cumulative / total_energy >= target_energy {
                return Ok(i + 1);
            }
        }

        Ok(svd.s.len())
    }
}

/// SVD Decoder for reconstructing weight matrices.
#[derive(Debug, Clone, Default)]
pub struct SvdDecoder;

impl SvdDecoder {
    /// Create a new decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decompress SVD-compressed weight back to full matrix.
    ///
    /// Computes: A = U × diag(S) × V^T
    pub fn decompress(&self, compressed: &SvdCompressedWeight) -> Result<Vec<f32>> {
        let SvdCompressedWeight { rows, cols, rank, u, s, vt } = compressed;

        if u.len() != rows * rank || s.len() != *rank || vt.len() != rank * cols {
            return Err(Error::corrupted("invalid SVD dimensions"));
        }

        // Compute U × diag(S) × V^T
        let mut result = vec![0.0f32; rows * cols];

        for i in 0..*rows {
            for j in 0..*cols {
                let mut sum = 0.0f32;
                for k in 0..*rank {
                    // result[i,j] = sum_k (U[i,k] * S[k] * Vt[k,j])
                    sum += u[i * rank + k] * s[k] * vt[k * cols + j];
                }
                result[i * cols + j] = sum;
            }
        }

        Ok(result)
    }

    /// Decompress with partial rank (progressive decompression).
    ///
    /// Uses only the first `use_rank` singular values.
    pub fn decompress_progressive(
        &self,
        compressed: &SvdCompressedWeight,
        use_rank: usize,
    ) -> Result<Vec<f32>> {
        let SvdCompressedWeight { rows, cols, rank, u, s, vt } = compressed;
        let use_rank = use_rank.min(*rank);

        let mut result = vec![0.0f32; rows * cols];

        for i in 0..*rows {
            for j in 0..*cols {
                let mut sum = 0.0f32;
                for k in 0..use_rank {
                    sum += u[i * rank + k] * s[k] * vt[k * cols + j];
                }
                result[i * cols + j] = sum;
            }
        }

        Ok(result)
    }
}

/// Normalize a vector to unit length.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Calculate MSE between two vectors.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::MAX;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / a.len() as f32
}

/// Calculate cosine similarity between two vectors.
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
    fn test_svd_encoder_basic() {
        let encoder = SvdEncoder::new(0.5);

        // Create a simple rank-2 matrix: A = u1*v1^T + u2*v2^T
        let rows = 8;
        let cols = 8;
        let mut data = vec![0.0f32; rows * cols];

        // First rank-1 component
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = (i + 1) as f32 * (j + 1) as f32;
            }
        }
        // Second rank-1 component
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] += ((i + 1) as f32 * 0.5) * ((j + 1) as f32 * 0.3);
            }
        }

        let compressed = encoder.compress(&data, rows, cols).unwrap();

        assert!(compressed.rank > 0);
        assert!(compressed.rank <= 4); // 50% of 8
        assert_eq!(compressed.rows, rows);
        assert_eq!(compressed.cols, cols);
    }

    #[test]
    fn test_svd_roundtrip() {
        let encoder = SvdEncoder::new(1.0); // Full rank
        let decoder = SvdDecoder::new();

        let rows = 4;
        let cols = 4;
        let data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let compressed = encoder.compress(&data, rows, cols).unwrap();
        let reconstructed = decoder.decompress(&compressed).unwrap();

        assert_eq!(reconstructed.len(), data.len());

        // With full rank, should be very close
        let error = mse(&data, &reconstructed);
        assert!(error < 0.01, "MSE too high: {}", error);
    }

    #[test]
    fn test_svd_low_rank_matrix() {
        let encoder = SvdEncoder::new(0.5);
        let decoder = SvdDecoder::new();

        // Create a rank-1 matrix
        let rows = 16;
        let cols = 16;
        let u: Vec<f32> = (0..rows).map(|i| (i as f32 + 1.0)).collect();
        let v: Vec<f32> = (0..cols).map(|j| (j as f32 + 1.0) * 0.5).collect();

        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = u[i] * v[j];
            }
        }

        let compressed = encoder.compress(&data, rows, cols).unwrap();
        let reconstructed = decoder.decompress(&compressed).unwrap();

        // Rank-1 matrix should compress very well
        let cos_sim = cosine_similarity(&data, &reconstructed);
        assert!(cos_sim > 0.99, "Cosine similarity too low: {}", cos_sim);
    }

    #[test]
    fn test_svd_compression_ratio() {
        let encoder = SvdEncoder::new(0.25);

        let rows = 64;
        let cols = 64;
        let data: Vec<f32> = (0..rows*cols).map(|i| (i as f32).sin()).collect();

        let compressed = encoder.compress(&data, rows, cols).unwrap();

        // At 25% rank, compression ratio should be significant
        let ratio = compressed.compression_ratio();
        assert!(ratio > 1.5, "Compression ratio too low: {}", ratio);
    }

    #[test]
    fn test_svd_progressive_decompress() {
        let encoder = SvdEncoder::new(1.0);
        let decoder = SvdDecoder::new();

        let rows = 8;
        let cols = 8;
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let compressed = encoder.compress(&data, rows, cols).unwrap();

        // Progressive decompress with increasing rank should improve quality
        let mut prev_error = f32::MAX;
        for r in 1..=compressed.rank {
            let partial = decoder.decompress_progressive(&compressed, r).unwrap();
            let error = mse(&data, &partial);
            assert!(error <= prev_error, "Quality should improve with more rank");
            prev_error = error;
        }
    }

    #[test]
    fn test_svd_empty_matrix() {
        let encoder = SvdEncoder::new(0.5);

        let result = encoder.compress(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_svd_size_mismatch() {
        let encoder = SvdEncoder::new(0.5);

        let data = vec![1.0f32; 10];
        let result = encoder.compress(&data, 4, 4); // Should be 16 elements
        assert!(result.is_err());
    }

    #[test]
    fn test_svd_1d_matrix() {
        let encoder = SvdEncoder::new(1.0);
        let decoder = SvdDecoder::new();

        // 1D vector as 1×n matrix
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let compressed = encoder.compress(&data, 1, 5).unwrap();
        let reconstructed = decoder.decompress(&compressed).unwrap();

        assert_eq!(compressed.rank, 1);
        let cos_sim = cosine_similarity(&data, &reconstructed);
        assert!(cos_sim > 0.99);
    }

    #[test]
    fn test_svd_analyze_rank() {
        let encoder = SvdEncoder::new(1.0);

        // Rank-2 matrix
        let rows = 16;
        let cols = 16;
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = (i as f32) * (j as f32) + (i as f32 * 0.5) * (j as f32 * 0.3);
            }
        }

        let optimal_rank = encoder.analyze_rank(&data, rows, cols, 0.99).unwrap();
        assert!(optimal_rank <= 4, "Rank-2 matrix should need few singular values: {}", optimal_rank);
    }

    #[test]
    fn test_svd_decoder_invalid_dimensions() {
        let decoder = SvdDecoder::new();

        let invalid = SvdCompressedWeight {
            rows: 4,
            cols: 4,
            rank: 2,
            u: vec![1.0; 4], // Wrong size, should be 4*2=8
            s: vec![1.0, 0.5],
            vt: vec![1.0; 8],
        };

        let result = decoder.decompress(&invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_svd_vs_identity() {
        let encoder = SvdEncoder::new(1.0);
        let decoder = SvdDecoder::new();

        // Identity matrix has all singular values = 1
        let n = 8;
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }

        let compressed = encoder.compress(&data, n, n).unwrap();
        let reconstructed = decoder.decompress(&compressed).unwrap();

        let error = mse(&data, &reconstructed);
        assert!(error < 0.01, "Identity matrix error: {}", error);
    }

    #[test]
    fn test_svd_storage_calculation() {
        let weight = SvdCompressedWeight {
            rows: 100,
            cols: 200,
            rank: 10,
            u: vec![0.0; 100 * 10],
            s: vec![0.0; 10],
            vt: vec![0.0; 10 * 200],
        };

        assert_eq!(weight.storage_bytes(), (1000 + 10 + 2000) * 4);
        assert_eq!(weight.original_bytes(), 100 * 200 * 4);
        assert!(weight.compression_ratio() > 6.0); // Should compress well
    }
}
