//! # Haagenti Compressed Tensor (HCT) Format
//!
//! High-performance compressed tensor storage for neural network weights,
//! with HoloTensor holographic compression support for progressive loading.
//!
//! ## Overview
//!
//! HCT provides two complementary storage modes:
//!
//! - **Standard HCT**: Block-compressed tensor storage with random access
//! - **HoloTensor**: Holographic compression enabling progressive reconstruction
//!
//! ## Standard HCT Format
//!
//! Block-based compression with LZ4 or Zstd for fast random access:
//!
//! ```ignore
//! use haagenti_hct::{HctWriter, HctReader, CompressionAlgorithm, DType};
//! use std::fs::File;
//!
//! // Write compressed tensor
//! let mut writer = HctWriter::new(
//!     File::create("weights.hct")?,
//!     CompressionAlgorithm::Zstd,
//!     DType::F16,
//!     &[4096, 4096],
//! )?;
//! writer.write_data(&weight_data)?;
//! writer.finish()?;
//!
//! // Read tensor
//! let mut reader = HctReader::open("weights.hct")?;
//! let data = reader.read_all()?;
//! ```
//!
//! ## HoloTensor Format
//!
//! Holographic compression enables progressive reconstruction from partial data:
//!
//! ```ignore
//! use haagenti_hct::{
//!     HoloTensorEncoder, HoloTensorDecoder,
//!     HolographicEncoding, DType,
//! };
//!
//! // Encode with spectral holography (8 fragments)
//! let encoder = HoloTensorEncoder::new(HolographicEncoding::Spectral)
//!     .with_fragments(8);
//! let (header, fragments) = encoder.encode_1d(&weights)?;
//!
//! // Reconstruct from partial fragments (any 4 of 8 for ~90% quality)
//! let mut decoder = HoloTensorDecoder::new(header);
//! decoder.add_fragment(fragments[0].clone())?;
//! decoder.add_fragment(fragments[3].clone())?;
//! decoder.add_fragment(fragments[5].clone())?;
//! decoder.add_fragment(fragments[7].clone())?;
//!
//! let approx_data = decoder.reconstruct()?;
//! ```
//!
//! ## Encoding Schemes
//!
//! | Scheme | Best For | Min Quality | Progressive |
//! |--------|----------|-------------|-------------|
//! | Spectral (DCT) | Dense MLP weights | 60% | Smooth curve |
//! | Random Projection | High-dimensional | 10% | Linear curve |
//! | Low-Rank Distributed | Attention layers | 30% | Sharp knee |
//!
//! ## Feature Flags
//!
//! - `lz4` - LZ4 compression for base blocks
//! - `zstd` - Zstd compression for better ratios
//! - `simd` - SIMD-accelerated primitives
//! - `full` - All features (default)

// Re-export core error types
pub use haagenti_core::{Error, Result};

// ==================== HCT Tensor Format ====================

// Format constants
pub use haagenti::tensor::{
    HCT_MAGIC, HCT_VERSION, HCT_VERSION_V2, DEFAULT_BLOCK_SIZE,
    FLAG_HEADER_CHECKSUM, FLAG_BLOCK_CHECKSUMS, FLAG_QUANTIZATION,
    FLAG_TENSOR_NAME, FLAG_HOLOGRAPHIC,
};

// Core types
pub use haagenti::tensor::{
    CompressionAlgorithm, DType, HctHeader, BlockIndex,
};

// V2 types (with quantization support)
pub use haagenti::tensor::{
    QuantizationScheme, QuantizationMetadata, BlockIndexV2,
};

// Reader/Writer
pub use haagenti::tensor::{
    HctReader, HctWriter, HctReaderV2, HctWriterV2,
};

// Utilities
pub use haagenti::tensor::{
    compress_file, ChecksumError,
    CompressionStats as HctCompressionStats,
};

// ==================== HoloTensor Holographic Compression ====================

// Format constants
pub use haagenti::holotensor::{
    HOLO_MAGIC, HOLO_VERSION,
    HOLO_FLAG_HEADER_CHECKSUM, HOLO_FLAG_FRAGMENT_CHECKSUMS,
    HOLO_FLAG_QUANTIZATION, HOLO_FLAG_QUALITY_CURVE,
    HOLO_FLAG_ESSENTIAL_FIRST, HOLO_FLAG_INTERLEAVED,
};

// Core types
pub use haagenti::holotensor::{
    HolographicEncoding, QualityCurve, HoloFragment, FragmentIndexEntry,
};

// Header
pub use haagenti::holotensor::HoloTensorHeader;

// DCT primitives (for advanced use)
pub use haagenti::holotensor::{dct_1d, idct_1d, dct_2d, idct_2d};

// Seeded RNG (for reproducible random projections)
pub use haagenti::holotensor::SeededRng;

// Spectral (DCT-based) encoder/decoder
pub use haagenti::holotensor::{SpectralEncoder, SpectralDecoder};

// Random Projection (JL-based) encoder/decoder
pub use haagenti::holotensor::{RphEncoder, RphDecoder};

// Low-Rank Distributed (SVD-based) encoder/decoder
pub use haagenti::holotensor::{LrdfEncoder, LrdfDecoder};

// Unified encoder/decoder API
pub use haagenti::holotensor::{HoloTensorEncoder, HoloTensorDecoder};

// File I/O
pub use haagenti::holotensor::{HoloTensorWriter, HoloTensorReader};

// Convenience functions
pub use haagenti::holotensor::{
    write_holotensor, read_holotensor, open_holotensor,
    encode_to_file, decode_from_file, decode_from_file_progressive,
};

/// Prelude module for common imports.
pub mod prelude {
    //! Convenient imports for common HCT operations.
    //!
    //! ```ignore
    //! use haagenti_hct::prelude::*;
    //! ```

    // Error handling
    pub use crate::{Error, Result};

    // Core HCT types
    pub use crate::{
        CompressionAlgorithm, DType,
        HctReader, HctWriter,
        HctReaderV2, HctWriterV2,
    };

    // HoloTensor types
    pub use crate::{
        HolographicEncoding, QualityCurve,
        HoloFragment, HoloTensorHeader,
        HoloTensorEncoder, HoloTensorDecoder,
        HoloTensorWriter, HoloTensorReader,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_values() {
        assert_eq!(DType::F32 as u8, 0);
        assert_eq!(DType::F16 as u8, 1);
        assert_eq!(DType::BF16 as u8, 2);
        assert_eq!(DType::I8 as u8, 3);
        assert_eq!(DType::I4 as u8, 4);
    }

    #[test]
    fn test_compression_algorithm_values() {
        assert_eq!(CompressionAlgorithm::Lz4 as u8, 0);
        assert_eq!(CompressionAlgorithm::Zstd as u8, 1);
    }

    #[test]
    fn test_holographic_encoding_values() {
        assert_eq!(HolographicEncoding::Spectral as u8, 0);
        assert_eq!(HolographicEncoding::RandomProjection as u8, 1);
        assert_eq!(HolographicEncoding::LowRankDistributed as u8, 2);
    }

    #[test]
    fn test_holographic_encoding_names() {
        assert_eq!(HolographicEncoding::Spectral.name(), "Spectral (DCT)");
        assert_eq!(HolographicEncoding::RandomProjection.name(), "Random Projection (JL)");
        assert_eq!(HolographicEncoding::LowRankDistributed.name(), "Low-Rank Distributed (SVD)");
    }

    #[test]
    fn test_default_quality_curves() {
        let spectral = HolographicEncoding::Spectral.default_quality_curve();
        assert_eq!(spectral.min_fragments, 1);
        assert_eq!(spectral.sufficient_fragments, 6);

        let rph = HolographicEncoding::RandomProjection.default_quality_curve();
        assert_eq!(rph.min_fragments, 2);

        let lrdf = HolographicEncoding::LowRankDistributed.default_quality_curve();
        assert_eq!(lrdf.sufficient_fragments, 4);
    }

    #[test]
    fn test_quality_curve_predict() {
        let curve = QualityCurve {
            coefficients: [0.0, 1.0, 0.0, 0.0],
            min_fragments: 1,
            sufficient_fragments: 8,
        };

        // Linear curve: quality = k/N
        let q = curve.predict(4, 8);
        assert!((q - 0.5).abs() < 0.001);

        let q = curve.predict(8, 8);
        assert!((q - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_holo_fragment_creation() {
        let data = vec![1, 2, 3, 4, 5];
        let fragment = HoloFragment::new(3, data.clone());

        assert_eq!(fragment.index, 3);
        assert_eq!(fragment.data, data);
        assert!(fragment.checksum != 0);
    }

    #[test]
    fn test_holo_fragment_checksum_consistency() {
        let data = vec![42u8; 100];
        let f1 = HoloFragment::new(0, data.clone());
        let f2 = HoloFragment::new(0, data);

        assert_eq!(f1.checksum, f2.checksum);
    }

    #[test]
    fn test_seeded_rng_determinism() {
        let mut rng1 = SeededRng::new(12345);
        let mut rng2 = SeededRng::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_dct_roundtrip_1d() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dct_output = vec![0.0f32; 8];
        dct_1d(&signal, &mut dct_output);

        let mut reconstructed = vec![0.0f32; 8];
        idct_1d(&dct_output, &mut reconstructed);

        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dct_roundtrip_2d() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut dct_output = vec![0.0f32; 16];
        dct_2d(&data, &mut dct_output, 4, 4);

        let mut reconstructed = vec![0.0f32; 16];
        idct_2d(&dct_output, &mut reconstructed, 4, 4);

        for (a, b) in data.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-4, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_spectral_encoder_decoder_roundtrip() {
        // Create small test matrix
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();

        let encoder = SpectralEncoder::new(4);
        let fragments = encoder.encode_2d(&data, 8, 8).expect("encode failed");

        assert_eq!(fragments.len(), 4);

        let mut decoder = SpectralDecoder::new(8, 8, 4);
        for frag in &fragments {
            decoder.add_fragment(frag).expect("add fragment failed");
        }

        let reconstructed = decoder.reconstruct();
        assert_eq!(reconstructed.len(), 64);

        // Should reconstruct with high quality from all fragments
        let mse: f32 = data.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / data.len() as f32;

        assert!(mse < 1.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_rph_encoder_decoder_roundtrip() {
        // Create small test vector
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();

        let encoder = RphEncoder::new(4, 42);
        let fragments = encoder.encode(&data).expect("encode failed");

        assert_eq!(fragments.len(), 4);

        let mut decoder = RphDecoder::new(32, 4);
        for frag in &fragments {
            decoder.add_fragment(frag).expect("add fragment failed");
        }

        let reconstructed = decoder.reconstruct();
        assert_eq!(reconstructed.len(), 32);

        // RPH provides approximate reconstruction
        let correlation: f32 = data.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();

        let norm_orig: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_recon: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cos_sim = correlation / (norm_orig * norm_recon + 1e-10);
        assert!(cos_sim > 0.5, "Cosine similarity too low: {}", cos_sim);
    }

    #[test]
    fn test_lrdf_encoder_decoder_roundtrip() {
        // Create small test matrix with low-rank structure
        let data: Vec<f32> = (0..64).map(|i| ((i / 8) + (i % 8)) as f32).collect();

        let encoder = LrdfEncoder::new(4, 42);
        let fragments = encoder.encode_2d(&data, 8, 8).expect("encode failed");

        assert_eq!(fragments.len(), 4);

        let mut decoder = LrdfDecoder::new(8, 8, 4);
        for frag in &fragments {
            decoder.add_fragment(frag).expect("add fragment failed");
        }

        let reconstructed = decoder.reconstruct();
        assert_eq!(reconstructed.len(), 64);

        // Should reconstruct low-rank matrix well
        let mse: f32 = data.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / data.len() as f32;

        assert!(mse < 10.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_unified_encoder_spectral() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

        let encoder = HoloTensorEncoder::new(HolographicEncoding::Spectral)
            .with_fragments(4)
            .with_seed(42);

        let (header, fragments) = encoder.encode_1d(&data).expect("encode failed");
        assert_eq!(fragments.len(), 4);
        assert_eq!(header.total_fragments, 4);
    }

    #[test]
    fn test_unified_encoder_decoder_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

        let encoder = HoloTensorEncoder::new(HolographicEncoding::Spectral)
            .with_fragments(4)
            .with_seed(42);

        let (header, fragments) = encoder.encode_1d(&data).expect("encode failed");

        let mut decoder = HoloTensorDecoder::new(header);
        for frag in fragments {
            decoder.add_fragment(frag).expect("add fragment failed");
        }

        assert!(decoder.can_reconstruct());
        let reconstructed = decoder.reconstruct().expect("reconstruct failed");
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_format_flags() {
        // Verify flag values don't overlap
        let flags = [
            FLAG_HEADER_CHECKSUM,
            FLAG_BLOCK_CHECKSUMS,
            FLAG_QUANTIZATION,
            FLAG_TENSOR_NAME,
            FLAG_HOLOGRAPHIC,
        ];

        for (i, &a) in flags.iter().enumerate() {
            for (j, &b) in flags.iter().enumerate() {
                if i != j {
                    assert_eq!(a & b, 0, "Flags {} and {} overlap", a, b);
                }
            }
        }
    }

    #[test]
    fn test_holo_flags() {
        // Verify holo flag values don't overlap
        let flags = [
            HOLO_FLAG_HEADER_CHECKSUM,
            HOLO_FLAG_FRAGMENT_CHECKSUMS,
            HOLO_FLAG_QUANTIZATION,
            HOLO_FLAG_QUALITY_CURVE,
            HOLO_FLAG_ESSENTIAL_FIRST,
            HOLO_FLAG_INTERLEAVED,
        ];

        for (i, &a) in flags.iter().enumerate() {
            for (j, &b) in flags.iter().enumerate() {
                if i != j {
                    assert_eq!(a & b, 0, "Holo flags {} and {} overlap", a, b);
                }
            }
        }
    }

    #[test]
    fn test_progressive_reconstruction_quality() {
        // Test that partial fragment loading gives partial quality
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();

        let encoder = SpectralEncoder::new(8);
        let fragments = encoder.encode_1d(&data).expect("encode failed");

        // Test with 2 of 8 fragments
        let mut decoder = SpectralDecoder::new(256, 1, 8);
        decoder.add_fragment(&fragments[0]).expect("add fragment 0");
        decoder.add_fragment(&fragments[4]).expect("add fragment 4");

        let partial = decoder.reconstruct();
        assert_eq!(partial.len(), 256);

        // Partial reconstruction should have some similarity
        let correlation: f32 = data.iter()
            .zip(partial.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();

        let norm_orig: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_partial: f32 = partial.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cos_sim = correlation / (norm_orig * norm_partial + 1e-10);
        // With essential coefficients replicated, should have decent similarity
        assert!(cos_sim > 0.3, "Cosine similarity from partial reconstruction too low: {}", cos_sim);
    }
}
