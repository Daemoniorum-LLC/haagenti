//! Integration tests for HoloTensor compression pipeline.
//!
//! These tests verify the full compression/decompression roundtrip
//! across the entire pipeline, from raw tensors to HCT files and back.

mod format_compatibility;
mod full_pipeline;
mod quality_regression;
