//! Streaming decompression for progressive inference loading.
//!
//! This module provides streaming/progressive loading of HCT-compressed tensors,
//! enabling inference to start with partial data and improve quality as more
//! fragments arrive.
//!
//! ## Design
//!
//! The CompressiveSpectral format stores tensors as:
//! - Fragment 0: Essential coefficients (top 20% by energy) + index map
//! - Fragments 1..N: Detail coefficients (remaining 80%)
//!
//! This allows progressive reconstruction:
//! 1. Load fragment 0 only -> ~70-80% quality, can start inference
//! 2. Load fragments 1-4 -> 90%+ quality
//! 3. Load all fragments -> Full quality
//!
//! ## Usage
//!
//! ```ignore
//! use haagenti::streaming::{StreamingTensorLoader, LoadPriority};
//!
//! // Create loader with async IO
//! let mut loader = StreamingTensorLoader::new(file_path)?;
//!
//! // Load essentials first (blocking)
//! loader.load_essentials().await?;
//!
//! // Start inference while loading details in background
//! let tensor = loader.reconstruct()?;  // ~80% quality
//!
//! // Continue loading in background
//! loader.load_next_detail().await?;
//!
//! // Get improved tensor when needed
//! let better_tensor = loader.reconstruct()?;  // ~90% quality
//! ```

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Arc, Mutex};

use crate::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use crate::holotensor::HoloFragment;
use crate::{Error, Result};

/// Priority for loading tensor fragments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadPriority {
    /// Load only essentials (fragment 0) - fastest, ~80% quality
    EssentialsOnly,
    /// Load essentials + first detail fragment - ~85% quality
    QuickStart,
    /// Load essentials + half of details - ~92% quality
    Balanced,
    /// Load all fragments - full quality
    Full,
}

impl LoadPriority {
    /// Returns the fraction of detail fragments to load (0.0 to 1.0).
    #[must_use]
    pub fn detail_fraction(&self) -> f32 {
        match self {
            LoadPriority::EssentialsOnly => 0.0,
            LoadPriority::QuickStart => 0.15,
            LoadPriority::Balanced => 0.5,
            LoadPriority::Full => 1.0,
        }
    }
}

/// Status of a streaming tensor load operation.
#[derive(Debug, Clone)]
pub struct LoadStatus {
    /// Whether essentials have been loaded.
    pub essentials_loaded: bool,
    /// Number of detail fragments loaded.
    pub details_loaded: u16,
    /// Total number of detail fragments.
    pub total_details: u16,
    /// Estimated quality (0.0 to 1.0).
    pub estimated_quality: f32,
    /// Whether loading is complete.
    pub complete: bool,
}

/// A streaming tensor loader for progressive decompression.
///
/// Allows loading tensor fragments incrementally, enabling inference
/// to start with partial data while more fragments load in the background.
pub struct StreamingTensorLoader {
    /// Tensor name.
    name: String,
    /// Tensor dimensions.
    width: usize,
    height: usize,
    /// The decoder accumulating fragments.
    decoder: CompressiveSpectralDecoder,
    /// Fragment data storage (for fragments not yet added).
    pending_fragments: Vec<HoloFragment>,
    /// Whether essentials are loaded.
    essentials_loaded: bool,
    /// Number of detail fragments loaded.
    details_loaded: u16,
    /// Total detail fragments expected.
    total_details: u16,
    /// Cached reconstruction (invalidated on new fragment).
    cached_reconstruction: Option<Vec<f32>>,
    /// Mean value to add back (DCT removes DC).
    mean_value: f32,
}

impl StreamingTensorLoader {
    /// Creates a new streaming tensor loader.
    pub fn new(name: impl Into<String>, width: usize, height: usize) -> Self {
        Self {
            name: name.into(),
            width,
            height,
            decoder: CompressiveSpectralDecoder::new(),
            pending_fragments: Vec::new(),
            essentials_loaded: false,
            details_loaded: 0,
            total_details: 0,
            cached_reconstruction: None,
            mean_value: 0.0,
        }
    }

    /// Sets the mean value to add back after reconstruction.
    pub fn with_mean(mut self, mean: f32) -> Self {
        self.mean_value = mean;
        self
    }

    /// Adds a fragment to the loader.
    ///
    /// Fragment 0 (essentials) is processed immediately.
    /// Detail fragments are queued for progressive loading.
    pub fn add_fragment(&mut self, fragment: HoloFragment) -> Result<()> {
        // Invalidate cache
        self.cached_reconstruction = None;

        if fragment.index == 0 {
            // Essential fragment - load immediately
            self.decoder.add_essentials(&fragment)?;
            self.essentials_loaded = true;

            // Extract total fragment count from decoder
            self.total_details = self.decoder.quality().ceil() as u16; // Approximation
        } else {
            // Detail fragment - add to decoder
            self.decoder.add_detail(&fragment)?;
            self.details_loaded += 1;
        }

        Ok(())
    }

    /// Queues a fragment for later loading.
    pub fn queue_fragment(&mut self, fragment: HoloFragment) {
        self.pending_fragments.push(fragment);
    }

    /// Loads the next pending fragment.
    ///
    /// Returns true if a fragment was loaded, false if queue is empty.
    pub fn load_next(&mut self) -> Result<bool> {
        if let Some(fragment) = self.pending_fragments.pop() {
            self.add_fragment(fragment)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Loads all pending fragments.
    pub fn load_all_pending(&mut self) -> Result<usize> {
        let count = self.pending_fragments.len();
        while !self.pending_fragments.is_empty() {
            self.load_next()?;
        }
        Ok(count)
    }

    /// Returns whether the tensor can be reconstructed.
    #[must_use]
    pub fn can_reconstruct(&self) -> bool {
        self.essentials_loaded && self.decoder.can_reconstruct()
    }

    /// Returns the current loading status.
    #[must_use]
    pub fn status(&self) -> LoadStatus {
        LoadStatus {
            essentials_loaded: self.essentials_loaded,
            details_loaded: self.details_loaded,
            total_details: self.total_details,
            estimated_quality: self.decoder.quality(),
            complete: self.pending_fragments.is_empty() && self.essentials_loaded,
        }
    }

    /// Reconstructs the tensor with currently loaded fragments.
    ///
    /// Returns an error if essentials haven't been loaded yet.
    pub fn reconstruct(&mut self) -> Result<Vec<f32>> {
        if !self.can_reconstruct() {
            return Err(Error::corrupted("essentials not loaded, cannot reconstruct"));
        }

        // Use cached reconstruction if available
        if let Some(ref cached) = self.cached_reconstruction {
            return Ok(cached.clone());
        }

        // Reconstruct from decoder
        let mut values = self.decoder.reconstruct()?;

        // Add back mean value
        if self.mean_value.abs() > 1e-10 {
            for v in &mut values {
                *v += self.mean_value;
            }
        }

        // Cache for reuse
        self.cached_reconstruction = Some(values.clone());

        Ok(values)
    }

    /// Returns tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns tensor dimensions.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// A collection of streaming tensor loaders for a model.
pub struct StreamingModelLoader {
    /// Tensor loaders by name.
    loaders: HashMap<String, StreamingTensorLoader>,
    /// Global loading priority.
    priority: LoadPriority,
    /// Whether to load in parallel.
    parallel: bool,
}

impl StreamingModelLoader {
    /// Creates a new streaming model loader.
    pub fn new(priority: LoadPriority) -> Self {
        Self {
            loaders: HashMap::new(),
            priority,
            parallel: true,
        }
    }

    /// Sets whether to load tensors in parallel.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Adds a tensor loader.
    pub fn add_tensor(&mut self, loader: StreamingTensorLoader) {
        self.loaders.insert(loader.name.clone(), loader);
    }

    /// Gets a tensor loader by name.
    pub fn get(&self, name: &str) -> Option<&StreamingTensorLoader> {
        self.loaders.get(name)
    }

    /// Gets a mutable tensor loader by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut StreamingTensorLoader> {
        self.loaders.get_mut(name)
    }

    /// Returns the number of tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.loaders.len()
    }

    /// Returns how many tensors can be reconstructed.
    #[must_use]
    pub fn ready_count(&self) -> usize {
        self.loaders.values().filter(|l| l.can_reconstruct()).count()
    }

    /// Returns overall loading progress (0.0 to 1.0).
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.loaders.is_empty() {
            return 1.0;
        }

        let total_quality: f32 = self.loaders.values()
            .map(|l| l.decoder.quality())
            .sum();

        total_quality / self.loaders.len() as f32
    }

    /// Loads next fragment for all tensors.
    ///
    /// Returns the number of fragments loaded.
    pub fn load_next_round(&mut self) -> Result<usize> {
        let mut loaded = 0;
        for loader in self.loaders.values_mut() {
            if loader.load_next()? {
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    /// Returns iterator over tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.loaders.keys()
    }

    /// Returns the loading priority.
    #[must_use]
    pub fn priority(&self) -> LoadPriority {
        self.priority
    }

    /// Returns whether parallel loading is enabled.
    #[must_use]
    pub fn is_parallel(&self) -> bool {
        self.parallel
    }
}

/// Callback for streaming load progress updates.
pub type ProgressCallback = Box<dyn Fn(&str, f32) + Send + Sync>;

/// Configuration for progressive tensor loading.
#[derive(Debug, Clone)]
pub struct ProgressiveLoadConfig {
    /// Initial priority for quick startup.
    pub initial_priority: LoadPriority,
    /// Target priority to reach eventually.
    pub target_priority: LoadPriority,
    /// Batch size for loading fragments.
    pub batch_size: usize,
    /// Whether to prioritize attention layers.
    pub prioritize_attention: bool,
    /// Whether to prioritize embedding layers.
    pub prioritize_embeddings: bool,
}

impl Default for ProgressiveLoadConfig {
    fn default() -> Self {
        Self {
            initial_priority: LoadPriority::EssentialsOnly,
            target_priority: LoadPriority::Full,
            batch_size: 10,
            prioritize_attention: true,
            prioritize_embeddings: true,
        }
    }
}

impl ProgressiveLoadConfig {
    /// Config for fastest startup (essentials only initially).
    #[must_use]
    pub fn fast_start() -> Self {
        Self {
            initial_priority: LoadPriority::EssentialsOnly,
            target_priority: LoadPriority::Balanced,
            batch_size: 20,
            ..Default::default()
        }
    }

    /// Config for balanced startup.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            initial_priority: LoadPriority::QuickStart,
            target_priority: LoadPriority::Full,
            batch_size: 10,
            ..Default::default()
        }
    }

    /// Config for high quality (load more before starting).
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            initial_priority: LoadPriority::Balanced,
            target_priority: LoadPriority::Full,
            batch_size: 5,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_priority_fraction() {
        assert!((LoadPriority::EssentialsOnly.detail_fraction() - 0.0).abs() < 0.01);
        assert!((LoadPriority::QuickStart.detail_fraction() - 0.15).abs() < 0.01);
        assert!((LoadPriority::Balanced.detail_fraction() - 0.5).abs() < 0.01);
        assert!((LoadPriority::Full.detail_fraction() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_streaming_tensor_loader_creation() {
        let loader = StreamingTensorLoader::new("test.weight", 128, 128);
        assert_eq!(loader.name(), "test.weight");
        assert_eq!(loader.dimensions(), (128, 128));
        assert!(!loader.can_reconstruct());
    }

    #[test]
    fn test_streaming_tensor_loader_with_mean() {
        let loader = StreamingTensorLoader::new("test", 64, 64)
            .with_mean(0.5);
        assert!((loader.mean_value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_load_status() {
        let loader = StreamingTensorLoader::new("test", 64, 64);
        let status = loader.status();

        assert!(!status.essentials_loaded);
        assert_eq!(status.details_loaded, 0);
        assert!(!status.complete);
    }

    #[test]
    fn test_streaming_model_loader() {
        let mut model = StreamingModelLoader::new(LoadPriority::Balanced);

        model.add_tensor(StreamingTensorLoader::new("layer.0.weight", 64, 64));
        model.add_tensor(StreamingTensorLoader::new("layer.1.weight", 64, 64));

        assert_eq!(model.tensor_count(), 2);
        assert_eq!(model.ready_count(), 0);
    }

    #[test]
    fn test_progressive_load_config() {
        let config = ProgressiveLoadConfig::default();
        assert_eq!(config.initial_priority, LoadPriority::EssentialsOnly);
        assert_eq!(config.target_priority, LoadPriority::Full);

        let fast = ProgressiveLoadConfig::fast_start();
        assert_eq!(fast.batch_size, 20);

        let balanced = ProgressiveLoadConfig::balanced();
        assert_eq!(balanced.initial_priority, LoadPriority::QuickStart);
    }

    #[test]
    fn test_streaming_loader_integration() {
        // Create encoder and encode some data
        let encoder = CompressiveSpectralEncoder::new(4, 0.70);
        let data: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

        // Encode
        let fragments = encoder.encode_2d(&data, 64, 64).unwrap();

        // Create streaming loader
        let mut loader = StreamingTensorLoader::new("test", 64, 64)
            .with_mean(mean);

        // Load essentials first
        for frag in &fragments {
            if frag.index == 0 {
                loader.add_fragment(frag.clone()).unwrap();
                break;
            }
        }

        assert!(loader.can_reconstruct());
        let status = loader.status();
        assert!(status.essentials_loaded);

        // Queue remaining fragments
        for frag in &fragments {
            if frag.index > 0 {
                loader.queue_fragment(frag.clone());
            }
        }

        // Load all and verify quality improves
        let initial_quality = loader.decoder.quality();
        loader.load_all_pending().unwrap();
        let final_quality = loader.decoder.quality();

        assert!(final_quality >= initial_quality);
    }
}
