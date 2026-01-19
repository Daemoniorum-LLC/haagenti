//! Integration tests for HoloTensor compression pipeline.
//!
//! These tests verify the full compression/decompression roundtrip
//! across the entire pipeline, from raw tensors to HCT files and back.

mod full_pipeline;
mod format_compatibility;
mod quality_regression;
