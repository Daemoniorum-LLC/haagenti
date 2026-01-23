//! Memory-efficient shard reader using memory-mapped I/O.
//!
//! Provides zero-copy access to tensor data in safetensors files without
//! loading the entire file into memory.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;

use crate::{Error, Result};

/// Data type in safetensors format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// BFloat16.
    BF16,
    /// 8-bit signed integer.
    I8,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// Boolean.
    Bool,
    /// Unknown type.
    Unknown,
}

impl DType {
    /// Returns the size of one element in bytes.
    #[must_use]
    pub fn element_size(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::Bool => 1,
            DType::I64 => 8,
            DType::Unknown => 1,
        }
    }

    /// Parses dtype from string.
    pub fn from_str(s: &str) -> Self {
        match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I8" => DType::I8,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "BOOL" => DType::Bool,
            _ => DType::Unknown,
        }
    }
}

/// Metadata for a single tensor in the shard.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Tensor name.
    pub name: String,
    /// Data type.
    pub dtype: DType,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Offset in data section (relative to data start).
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
}

impl TensorEntry {
    /// Returns the number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns true if this is a 1D tensor (bias, layernorm).
    #[must_use]
    pub fn is_1d(&self) -> bool {
        self.shape.len() == 1
    }

    /// Returns true if this is a 2D tensor (weight matrix).
    #[must_use]
    pub fn is_2d(&self) -> bool {
        self.shape.len() == 2
    }
}

/// Raw tensor info from safetensors JSON.
#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: Option<String>,
    shape: Option<Vec<usize>>,
    data_offsets: Option<(usize, usize)>,
}

/// Memory-mapped shard reader.
///
/// Provides zero-copy access to tensor data using memory-mapped I/O.
/// The OS handles paging data in/out as needed.
pub struct ShardReader {
    /// Path to the shard file.
    path: PathBuf,
    /// Memory-mapped file.
    mmap: Mmap,
    /// Tensor metadata sorted by name.
    tensors: Vec<TensorEntry>,
    /// Name to index lookup.
    name_index: HashMap<String, usize>,
    /// Offset where data section starts.
    data_offset: usize,
}

impl ShardReader {
    /// Opens a safetensors shard with memory-mapping.
    ///
    /// This does NOT load the file into memory - it creates a virtual mapping
    /// that the OS will page in on demand.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Open file for memory mapping
        let file = File::open(path)
            .map_err(|e| Error::io(format!("failed to open shard {}: {}", path.display(), e)))?;

        // Create memory map
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| Error::io(format!("failed to mmap shard {}: {}", path.display(), e)))?
        };

        // Parse header
        if mmap.len() < 8 {
            return Err(Error::corrupted("shard too small for header"));
        }

        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;

        if mmap.len() < header_end {
            return Err(Error::corrupted("shard truncated before header end"));
        }

        // Parse JSON header
        let header_json = std::str::from_utf8(&mmap[8..header_end])
            .map_err(|e| Error::corrupted(format!("invalid UTF-8 in header: {}", e)))?;

        let raw_tensors: HashMap<String, RawTensorInfo> = serde_json::from_str(header_json)
            .map_err(|e| Error::corrupted(format!("invalid JSON header: {}", e)))?;

        // Convert to TensorEntry, skipping __metadata__ and any entries without required fields
        let mut tensors: Vec<TensorEntry> = raw_tensors
            .into_iter()
            .filter(|(name, _)| name != "__metadata__")
            .filter_map(|(name, info)| {
                // Skip entries without required fields (like __metadata__)
                let dtype_str = info.dtype?;
                let shape = info.shape?;
                let (start, end) = info.data_offsets?;

                let dtype = DType::from_str(&dtype_str);
                Some(TensorEntry {
                    name,
                    dtype,
                    shape,
                    offset: start,
                    size: end - start,
                })
            })
            .collect();

        // Sort by name for deterministic ordering
        tensors.sort_by(|a, b| a.name.cmp(&b.name));

        // Build name index
        let name_index: HashMap<String, usize> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            tensors,
            name_index,
            data_offset: header_end,
        })
    }

    /// Returns the path to this shard.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the number of tensors in this shard.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Returns iterator over tensor entries.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors.iter()
    }

    /// Gets a tensor entry by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.name_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Returns true if the shard contains a tensor with this name.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.name_index.contains_key(name)
    }

    /// Gets raw tensor data as a byte slice (zero-copy).
    ///
    /// This returns a view into the memory-mapped file without copying.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .get(name)
            .ok_or_else(|| Error::corrupted(format!("tensor '{}' not found in shard", name)))?;

        let start = self.data_offset + entry.offset;
        let end = start + entry.size;

        if end > self.mmap.len() {
            return Err(Error::corrupted(format!(
                "tensor '{}' extends past end of file",
                name
            )));
        }

        Ok(&self.mmap[start..end])
    }

    /// Converts tensor bytes to f32 values.
    ///
    /// Handles F32, F16, and BF16 input types.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let entry = self
            .get(name)
            .ok_or_else(|| Error::corrupted(format!("tensor '{}' not found", name)))?;
        let bytes = self.tensor_bytes(name)?;

        match entry.dtype {
            DType::F32 => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(floats)
            }
            DType::F16 => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes(c.try_into().unwrap());
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            DType::BF16 => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes(c.try_into().unwrap());
                        half::bf16::from_bits(bits).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            _ => Err(Error::corrupted(format!(
                "unsupported dtype {:?} for tensor '{}'",
                entry.dtype, name
            ))),
        }
    }

    /// Returns total size of all tensor data in bytes.
    #[must_use]
    pub fn total_data_size(&self) -> usize {
        self.tensors.iter().map(|t| t.size).sum()
    }

    /// Returns total number of elements across all tensors.
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.tensors.iter().map(|t| t.num_elements()).sum()
    }
}

/// Discovers all shard files for a model.
///
/// Searches for patterns like:
/// - `model.safetensors` (single shard)
/// - `model-00001-of-00100.safetensors` (multi-shard)
pub fn discover_shards(model_path: &Path) -> Result<Vec<PathBuf>> {
    // If it's a file, return just that
    if model_path.is_file() && model_path.extension().is_some_and(|e| e == "safetensors") {
        return Ok(vec![model_path.to_path_buf()]);
    }

    // If it's a directory, find all safetensors files
    if model_path.is_dir() {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(model_path)
            .map_err(|e| Error::io(format!("failed to read model directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|e| e == "safetensors"))
            .collect();

        if shards.is_empty() {
            return Err(Error::io("no safetensors files found in directory"));
        }

        // Sort for deterministic order
        shards.sort();
        return Ok(shards);
    }

    // Try HuggingFace cache
    if let Some(cache_shards) = find_in_hf_cache(model_path)? {
        return Ok(cache_shards);
    }

    Err(Error::io(format!(
        "model not found: {}",
        model_path.display()
    )))
}

/// Searches HuggingFace cache for model shards.
fn find_in_hf_cache(model_id: &Path) -> Result<Option<Vec<PathBuf>>> {
    let model_str = model_id.to_string_lossy();

    // Check if it looks like a HF model ID (org/model)
    if !model_str.contains('/') {
        return Ok(None);
    }

    // Standard HF cache location
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let cache_base = PathBuf::from(home).join(".cache/huggingface/hub");

    // Convert model ID to cache path format
    let cache_name = format!("models--{}", model_str.replace('/', "--"));
    let model_cache = cache_base.join(&cache_name);

    if !model_cache.exists() {
        return Ok(None);
    }

    // Find snapshots directory
    let snapshots_dir = model_cache.join("snapshots");
    if !snapshots_dir.exists() {
        return Ok(None);
    }

    // Get the latest snapshot (by modification time or first one)
    let snapshot = std::fs::read_dir(&snapshots_dir)
        .map_err(|e| Error::io(format!("failed to read snapshots: {}", e)))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path());

    let snapshot_path = match snapshot {
        Some(p) => p,
        None => return Ok(None),
    };

    // Find all safetensors files in snapshot
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&snapshot_path)
        .map_err(|e| Error::io(format!("failed to read snapshot: {}", e)))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();

    if shards.is_empty() {
        return Ok(None);
    }

    shards.sort();
    Ok(Some(shards))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(DType::F32.element_size(), 4);
        assert_eq!(DType::F16.element_size(), 2);
        assert_eq!(DType::BF16.element_size(), 2);
        assert_eq!(DType::I8.element_size(), 1);
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(DType::from_str("F32"), DType::F32);
        assert_eq!(DType::from_str("F16"), DType::F16);
        assert_eq!(DType::from_str("BF16"), DType::BF16);
        assert_eq!(DType::from_str("UNKNOWN"), DType::Unknown);
    }

    #[test]
    fn test_tensor_entry() {
        let entry = TensorEntry {
            name: "layer.0.weight".to_string(),
            dtype: DType::F32,
            shape: vec![64, 128],
            offset: 0,
            size: 64 * 128 * 4,
        };

        assert_eq!(entry.num_elements(), 64 * 128);
        assert!(entry.is_2d());
        assert!(!entry.is_1d());
    }

    // Integration tests require actual safetensors files
    #[test]
    #[ignore]
    fn test_shard_reader_open() {
        // Requires a real safetensors file
        let path = "/tmp/test.safetensors";
        if std::path::Path::new(path).exists() {
            let reader = ShardReader::open(path).unwrap();
            assert!(reader.tensor_count() > 0);
        }
    }
}
