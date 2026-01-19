//! Testing utilities for HoloTensor compression validation.
//!
//! This module provides reusable utilities for testing compression pipelines:
//! - Safetensors file parsing
//! - Quality metrics computation (MSE, PSNR, cosine similarity)
//! - INT4 quantization helpers
//!
//! # Feature Gate
//!
//! This module is only available when the `testing` feature is enabled,
//! or when running tests.
//!
//! ```toml
//! [dev-dependencies]
//! haagenti = { version = "...", features = ["testing"] }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use haagenti::testing::{metrics, safetensors, quantization};
//!
//! // Load model weights
//! let data = std::fs::read("model.safetensors").unwrap();
//! let (offset, tensors) = safetensors::parse_safetensors_header(&data).unwrap();
//!
//! // Convert to f32
//! let info = &tensors["model.embed_tokens.weight"];
//! let weights = safetensors::bytes_to_f32(
//!     &data[offset + info.data_offsets.0..offset + info.data_offsets.1],
//!     &info.dtype
//! );
//!
//! // Quantize and dequantize
//! let quantized = quantization::quantize_int4(&weights);
//! let reconstructed = quantization::dequantize_int4(&quantized, weights.len());
//!
//! // Compute quality metrics
//! let report = metrics::compute_quality(&weights, &reconstructed);
//! println!("{}", report);
//! ```

pub mod metrics;
pub mod quantization;
pub mod safetensors;

// Re-export commonly used items from metrics
pub use metrics::{
    compute_quality, cosine_similarity, max_error, mse, psnr_from_mse, QualityReport,
};

// Re-export commonly used items from quantization
pub use quantization::{
    compressed_size, compression_ratio, dequantize_int4, f16_bytes_to_f32, f32_to_f16_bytes,
    quantize_int4, Q4_BLOCK_SIZE,
};

// Re-export commonly used items from safetensors
pub use safetensors::{
    bytes_to_f32, find_all_model_shards, find_model_in_cache, parse_safetensors_header, TensorInfo,
};
