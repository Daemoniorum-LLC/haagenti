//! Fragment library for cross-model storage
//!
//! The library stores deduplicated fragments and provides:
//! - Content-addressable storage
//! - Similarity-based deduplication
//! - Reference counting for garbage collection
//! - Model manifest management

use crate::{
    Fragment, FragmentError, FragmentId, ModelManifest, Result, SignatureConfig, SimilarityIndex,
    SimilarityMatch, SimilarityThreshold,
};
use dashmap::DashMap;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info};

/// Library configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryConfig {
    /// Root directory for library storage
    pub root_path: PathBuf,
    /// Signature configuration for similarity detection
    pub signature_config: SignatureConfig,
    /// Similarity thresholds
    pub similarity_thresholds: SimilarityThreshold,
    /// Maximum fragment size (bytes)
    pub max_fragment_size: usize,
    /// Enable compression
    pub compression_enabled: bool,
    /// Compression level (1-22 for zstd)
    pub compression_level: i32,
}

impl Default for LibraryConfig {
    fn default() -> Self {
        Self {
            root_path: PathBuf::from("./fragment_library"),
            signature_config: SignatureConfig::default(),
            similarity_thresholds: SimilarityThreshold::default(),
            max_fragment_size: 16 * 1024 * 1024, // 16MB
            compression_enabled: true,
            compression_level: 3,
        }
    }
}

/// Library statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryStats {
    /// Total fragments stored
    pub total_fragments: usize,
    /// Total models registered
    pub total_models: usize,
    /// Total storage used (bytes)
    pub storage_used: u64,
    /// Storage saved by deduplication (bytes)
    pub dedup_savings: u64,
    /// Average fragment size
    pub avg_fragment_size: usize,
    /// Fragments shared across multiple models
    pub shared_fragments: usize,
    /// Average references per shared fragment
    pub avg_refs_per_shared: f32,
}

/// Fragment library for cross-model storage
pub struct FragmentLibrary {
    /// Configuration
    config: LibraryConfig,
    /// Similarity index for deduplication
    similarity_index: SimilarityIndex,
    /// In-memory fragment cache
    fragment_cache: DashMap<FragmentId, Arc<Fragment>>,
    /// Model manifests
    manifests: DashMap<String, Arc<ModelManifest>>,
    /// Reference counts
    ref_counts: DashMap<FragmentId, u32>,
    /// Total storage used
    storage_used: std::sync::atomic::AtomicU64,
}

impl FragmentLibrary {
    /// Create or open a fragment library
    pub async fn open(config: LibraryConfig) -> Result<Self> {
        // Create directories
        fs::create_dir_all(&config.root_path).await?;
        fs::create_dir_all(config.root_path.join("fragments")).await?;
        fs::create_dir_all(config.root_path.join("manifests")).await?;

        let similarity_index = SimilarityIndex::new(
            config.signature_config.clone(),
            config.similarity_thresholds.clone(),
        );

        let library = Self {
            config,
            similarity_index,
            fragment_cache: DashMap::new(),
            manifests: DashMap::new(),
            ref_counts: DashMap::new(),
            storage_used: std::sync::atomic::AtomicU64::new(0),
        };

        // Load existing index
        library.load_index().await?;

        Ok(library)
    }

    /// Load library index from disk
    async fn load_index(&self) -> Result<()> {
        let index_path = self.config.root_path.join("index.bin");

        if !index_path.exists() {
            info!("Creating new fragment library");
            return Ok(());
        }

        let data = fs::read(&index_path).await?;
        let index: LibraryIndex = bincode::deserialize(&data)?;

        info!(
            "Loaded library index: {} fragments, {} models",
            index.fragments.len(),
            index.manifests.len()
        );

        // Restore ref counts
        for (id, count) in index.ref_counts {
            self.ref_counts.insert(id, count);
        }

        // Load manifests
        for manifest_id in &index.manifests {
            if let Ok(manifest) = self.load_manifest_from_disk(manifest_id).await {
                self.manifests
                    .insert(manifest_id.clone(), Arc::new(manifest));
            }
        }

        // Rebuild similarity index from fragment metadata
        for frag_meta in &index.fragments {
            // Create minimal fragment for indexing
            let fragment = Fragment {
                id: frag_meta.id,
                metadata: frag_meta.metadata.clone(),
                data: Vec::new(), // Don't load data yet
                signature: frag_meta.signature,
            };
            self.similarity_index.index(&fragment);
        }

        self.storage_used
            .store(index.storage_used, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Save library index to disk
    pub async fn save_index(&self) -> Result<()> {
        let mut fragments = Vec::new();

        for entry in self.ref_counts.iter() {
            if let Some(frag) = self.fragment_cache.get(entry.key()) {
                fragments.push(FragmentMeta {
                    id: frag.id,
                    metadata: frag.metadata.clone(),
                    signature: frag.signature,
                });
            }
        }

        let manifests: Vec<String> = self.manifests.iter().map(|e| e.key().clone()).collect();
        let ref_counts: IndexMap<FragmentId, u32> = self
            .ref_counts
            .iter()
            .map(|e| (*e.key(), *e.value()))
            .collect();

        let index = LibraryIndex {
            version: crate::LIBRARY_FORMAT_VERSION,
            fragments,
            manifests,
            ref_counts,
            storage_used: self.storage_used.load(std::sync::atomic::Ordering::Relaxed),
        };

        let data = bincode::serialize(&index)?;
        let index_path = self.config.root_path.join("index.bin");

        // Atomic write
        let tmp_path = index_path.with_extension("tmp");
        fs::write(&tmp_path, &data).await?;
        fs::rename(&tmp_path, &index_path).await?;

        info!("Saved library index: {} fragments", index.fragments.len());

        Ok(())
    }

    /// Store a fragment (with deduplication)
    pub async fn store_fragment(&self, fragment: Fragment) -> Result<StoreResult> {
        // Check for duplicates
        if let Some(existing) = self.similarity_index.find_duplicate(&fragment.data) {
            debug!(
                "Found duplicate fragment {} (similarity: {:.4})",
                existing.fragment_id, existing.similarity
            );

            // Increment ref count
            self.ref_counts
                .entry(existing.fragment_id)
                .and_modify(|c| *c += 1)
                .or_insert(1);

            return Ok(StoreResult::Deduplicated {
                fragment_id: existing.fragment_id,
                similarity: existing.similarity,
                saved_bytes: fragment.data.len(),
            });
        }

        // Store new fragment
        let fragment_id = fragment.id;
        let size = fragment.data.len();

        // Write to disk
        self.write_fragment_to_disk(&fragment).await?;

        // Update index
        self.similarity_index.index(&fragment);
        self.ref_counts.insert(fragment_id, 1);
        self.fragment_cache.insert(fragment_id, Arc::new(fragment));

        self.storage_used
            .fetch_add(size as u64, std::sync::atomic::Ordering::Relaxed);

        Ok(StoreResult::Stored { fragment_id, size })
    }

    /// Write fragment to disk
    async fn write_fragment_to_disk(&self, fragment: &Fragment) -> Result<()> {
        let path = self.fragment_path(&fragment.id);

        // Ensure parent directory exists (sharded by first 2 hex chars)
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Serialize fragment
        let data = bincode::serialize(fragment)?;

        // Write atomically
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, &data).await?;
        fs::rename(&tmp_path, &path).await?;

        Ok(())
    }

    /// Read fragment from disk
    async fn read_fragment_from_disk(&self, id: &FragmentId) -> Result<Fragment> {
        let path = self.fragment_path(id);

        if !path.exists() {
            return Err(FragmentError::NotFound(id.to_hex()));
        }

        let data = fs::read(&path).await?;
        let fragment: Fragment = bincode::deserialize(&data)?;

        Ok(fragment)
    }

    /// Get fragment path
    fn fragment_path(&self, id: &FragmentId) -> PathBuf {
        let hex = id.to_hex();
        self.config
            .root_path
            .join("fragments")
            .join(&hex[..2])
            .join(format!("{}.bin", hex))
    }

    /// Load a fragment (from cache or disk)
    pub async fn load_fragment(&self, id: &FragmentId) -> Result<Arc<Fragment>> {
        // Check cache first
        if let Some(fragment) = self.fragment_cache.get(id) {
            return Ok(fragment.clone());
        }

        // Load from disk
        let fragment = self.read_fragment_from_disk(id).await?;
        let fragment = Arc::new(fragment);

        // Cache it
        self.fragment_cache.insert(*id, fragment.clone());

        Ok(fragment)
    }

    /// Register a model manifest
    pub async fn register_model(&self, manifest: ModelManifest) -> Result<()> {
        let model_id = manifest.model_id.clone();

        // Save manifest to disk
        self.save_manifest_to_disk(&manifest).await?;

        // Update ref counts for all fragments
        for frag_id in manifest.fragment_ids() {
            self.ref_counts
                .entry(frag_id)
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }

        self.manifests.insert(model_id.clone(), Arc::new(manifest));

        info!("Registered model: {}", model_id);

        Ok(())
    }

    /// Save manifest to disk
    async fn save_manifest_to_disk(&self, manifest: &ModelManifest) -> Result<()> {
        let path = self
            .config
            .root_path
            .join("manifests")
            .join(format!("{}.bin", manifest.model_id));

        let data = manifest.to_bytes()?;

        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, &data).await?;
        fs::rename(&tmp_path, &path).await?;

        Ok(())
    }

    /// Load manifest from disk
    async fn load_manifest_from_disk(&self, model_id: &str) -> Result<ModelManifest> {
        let path = self
            .config
            .root_path
            .join("manifests")
            .join(format!("{}.bin", model_id));

        let data = fs::read(&path).await?;
        ModelManifest::from_bytes(&data)
    }

    /// Get a model manifest
    pub fn get_manifest(&self, model_id: &str) -> Option<Arc<ModelManifest>> {
        self.manifests.get(model_id).map(|e| e.value().clone())
    }

    /// List all models
    pub fn list_models(&self) -> Vec<String> {
        self.manifests.iter().map(|e| e.key().clone()).collect()
    }

    /// Find similar fragments across the library
    pub fn find_similar(&self, data: &[u8], max_results: usize) -> Vec<SimilarityMatch> {
        self.similarity_index.find_similar(data, max_results)
    }

    /// Get library statistics
    pub fn stats(&self) -> LibraryStats {
        let total_fragments = self.ref_counts.len();
        let total_models = self.manifests.len();
        let storage_used = self.storage_used.load(std::sync::atomic::Ordering::Relaxed);

        let mut shared_count = 0;
        let mut total_shared_refs = 0u64;

        for entry in self.ref_counts.iter() {
            if *entry.value() > 1 {
                shared_count += 1;
                total_shared_refs += *entry.value() as u64;
            }
        }

        let avg_refs_per_shared = if shared_count > 0 {
            total_shared_refs as f32 / shared_count as f32
        } else {
            0.0
        };

        // Calculate dedup savings from manifests
        let dedup_savings: u64 = self.manifests.iter().map(|e| e.value().dedup_savings).sum();

        let avg_fragment_size = if total_fragments > 0 {
            storage_used as usize / total_fragments
        } else {
            0
        };

        LibraryStats {
            total_fragments,
            total_models,
            storage_used,
            dedup_savings,
            avg_fragment_size,
            shared_fragments: shared_count,
            avg_refs_per_shared,
        }
    }

    /// Garbage collect unreferenced fragments
    pub async fn gc(&self) -> Result<GcResult> {
        let mut removed = 0;
        let mut bytes_freed = 0u64;

        // Find fragments with zero refs
        let zero_ref_ids: Vec<FragmentId> = self
            .ref_counts
            .iter()
            .filter(|e| *e.value() == 0)
            .map(|e| *e.key())
            .collect();

        for id in zero_ref_ids {
            // Get size before removing
            if let Some((_, frag)) = self.fragment_cache.remove(&id) {
                bytes_freed += frag.data.len() as u64;
            }

            // Remove from disk
            let path = self.fragment_path(&id);
            if path.exists() {
                if let Ok(meta) = fs::metadata(&path).await {
                    bytes_freed += meta.len();
                }
                let _ = fs::remove_file(&path).await;
            }

            // Remove from index
            self.similarity_index.remove(&id);
            self.ref_counts.remove(&id);

            removed += 1;
        }

        if removed > 0 {
            self.storage_used
                .fetch_sub(bytes_freed, std::sync::atomic::Ordering::Relaxed);
            info!(
                "GC removed {} fragments, freed {} bytes",
                removed, bytes_freed
            );
        }

        Ok(GcResult {
            fragments_removed: removed,
            bytes_freed,
        })
    }
}

/// Result of storing a fragment
#[derive(Debug)]
pub enum StoreResult {
    /// Fragment was stored as new
    Stored {
        fragment_id: FragmentId,
        size: usize,
    },
    /// Fragment was deduplicated against existing
    Deduplicated {
        fragment_id: FragmentId,
        similarity: f32,
        saved_bytes: usize,
    },
}

/// Result of garbage collection
#[derive(Debug)]
pub struct GcResult {
    /// Number of fragments removed
    pub fragments_removed: usize,
    /// Bytes freed
    pub bytes_freed: u64,
}

/// Serializable library index
#[derive(Debug, Serialize, Deserialize)]
struct LibraryIndex {
    version: u32,
    fragments: Vec<FragmentMeta>,
    manifests: Vec<String>,
    ref_counts: IndexMap<FragmentId, u32>,
    storage_used: u64,
}

/// Minimal fragment metadata for index
#[derive(Debug, Serialize, Deserialize)]
struct FragmentMeta {
    id: FragmentId,
    metadata: crate::FragmentMetadata,
    signature: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FragmentType;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_store_and_load() {
        let dir = tempdir().unwrap();
        let config = LibraryConfig {
            root_path: dir.path().to_path_buf(),
            ..Default::default()
        };

        let library = FragmentLibrary::open(config).await.unwrap();

        let fragment = Fragment::new(
            vec![42u8; 1024],
            FragmentType::Generic,
            smallvec::smallvec![32, 32],
            "fp16",
            "lz4",
            0.5,
        );

        let id = fragment.id;
        let result = library.store_fragment(fragment).await.unwrap();

        assert!(matches!(result, StoreResult::Stored { .. }));

        let loaded = library.load_fragment(&id).await.unwrap();
        assert_eq!(loaded.data.len(), 1024);
    }

    #[tokio::test]
    async fn test_deduplication() {
        let dir = tempdir().unwrap();
        let config = LibraryConfig {
            root_path: dir.path().to_path_buf(),
            ..Default::default()
        };

        let library = FragmentLibrary::open(config).await.unwrap();

        let data = vec![42u8; 1024];

        let fragment1 = Fragment::new(
            data.clone(),
            FragmentType::Generic,
            smallvec::smallvec![32, 32],
            "fp16",
            "lz4",
            0.5,
        );

        let fragment2 = Fragment::new(
            data,
            FragmentType::Generic,
            smallvec::smallvec![32, 32],
            "fp16",
            "lz4",
            0.5,
        );

        library.store_fragment(fragment1).await.unwrap();
        let result = library.store_fragment(fragment2).await.unwrap();

        assert!(matches!(result, StoreResult::Deduplicated { .. }));
    }
}
