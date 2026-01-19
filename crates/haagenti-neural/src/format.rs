//! NCT (Neural Compressed Tensor) file format

use crate::{EncodedTensor, LayerCodebook, NeuralError, Result, NCT_MAGIC};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// NCT (Neural Compressed Tensor) file header.
///
/// Defines the binary layout of an NCT file, which stores neural network
/// tensors in a compressed format using vector quantization.
///
/// # Binary Layout
///
/// ```text
/// Offset  Size   Field
/// ──────  ────   ─────
/// 0       4      magic ("NCT\0")
/// 4       2      version (u16)
/// 6       4      num_tensors (u32)
/// 10      8      codebook_offset (u64)
/// 18      8      tensor_offset (u64)
/// 26      8      metadata_offset (u64)
/// 34      8      file_size (u64)
/// ──────────────────────────────────
/// Total: 42 bytes (SIZE constant is 40)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NctHeader {
    /// Magic bytes
    pub magic: [u8; 4],
    /// Format version
    pub version: u16,
    /// Number of tensors
    pub num_tensors: u32,
    /// Codebook section offset
    pub codebook_offset: u64,
    /// Tensor data section offset
    pub tensor_offset: u64,
    /// Metadata section offset
    pub metadata_offset: u64,
    /// Total file size
    pub file_size: u64,
}

impl NctHeader {
    /// Create a new header
    pub fn new(num_tensors: u32) -> Self {
        Self {
            magic: NCT_MAGIC,
            version: 1,
            num_tensors,
            codebook_offset: 0,
            tensor_offset: 0,
            metadata_offset: 0,
            file_size: 0,
        }
    }

    /// Header size in bytes
    pub const SIZE: usize = 40;

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::SIZE);
        bytes.extend_from_slice(&self.magic);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.num_tensors.to_le_bytes());
        bytes.extend_from_slice(&self.codebook_offset.to_le_bytes());
        bytes.extend_from_slice(&self.tensor_offset.to_le_bytes());
        bytes.extend_from_slice(&self.metadata_offset.to_le_bytes());
        bytes.extend_from_slice(&self.file_size.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(NeuralError::InvalidFormat("Header too short".into()));
        }

        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != NCT_MAGIC {
            return Err(NeuralError::InvalidFormat("Invalid magic bytes".into()));
        }

        Ok(Self {
            magic,
            version: u16::from_le_bytes([bytes[4], bytes[5]]),
            num_tensors: u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]),
            codebook_offset: u64::from_le_bytes(bytes[10..18].try_into().unwrap()),
            tensor_offset: u64::from_le_bytes(bytes[18..26].try_into().unwrap()),
            metadata_offset: u64::from_le_bytes(bytes[26..34].try_into().unwrap()),
            file_size: u64::from_le_bytes(bytes[34..42].try_into().unwrap()),
        })
    }
}

/// NCT file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NctMetadata {
    /// Model identifier
    pub model_id: String,
    /// Original model size in bytes
    pub original_size: u64,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Mean quality (PSNR)
    pub mean_quality: f32,
    /// Creation timestamp
    pub created_at: u64,
    /// Encoder version
    pub encoder_version: String,
    /// Additional metadata
    pub extra: std::collections::HashMap<String, String>,
}

impl NctMetadata {
    /// Create new metadata
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            original_size: 0,
            compressed_size: 0,
            compression_ratio: 0.0,
            mean_quality: 0.0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            encoder_version: env!("CARGO_PKG_VERSION").to_string(),
            extra: std::collections::HashMap::new(),
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| NeuralError::InvalidFormat(e.to_string()))
    }
}

/// NCT file container
#[derive(Debug)]
pub struct NctFile {
    /// File header
    pub header: NctHeader,
    /// Layer codebooks
    pub codebooks: LayerCodebook,
    /// Encoded tensors
    pub tensors: Vec<EncodedTensor>,
    /// File metadata
    pub metadata: NctMetadata,
}

impl NctFile {
    /// Create a new NCT file
    pub fn new(
        codebooks: LayerCodebook,
        tensors: Vec<EncodedTensor>,
        metadata: NctMetadata,
    ) -> Self {
        Self {
            header: NctHeader::new(tensors.len() as u32),
            codebooks,
            tensors,
            metadata,
        }
    }

    /// Write to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Serialize sections
        let codebook_bytes = self.codebooks.to_bytes();
        let tensor_bytes = bincode::serialize(&self.tensors)
            .map_err(|e| NeuralError::InvalidFormat(e.to_string()))?;
        let metadata_bytes = self.metadata.to_bytes();

        // Calculate offsets
        let header_size = NctHeader::SIZE as u64;
        let codebook_offset = header_size;
        let tensor_offset = codebook_offset + codebook_bytes.len() as u64;
        let metadata_offset = tensor_offset + tensor_bytes.len() as u64;
        let file_size = metadata_offset + metadata_bytes.len() as u64;

        // Write header with correct offsets
        let header = NctHeader {
            magic: NCT_MAGIC,
            version: 1,
            num_tensors: self.tensors.len() as u32,
            codebook_offset,
            tensor_offset,
            metadata_offset,
            file_size,
        };

        writer.write_all(&header.to_bytes())?;
        writer.write_all(&codebook_bytes)?;
        writer.write_all(&tensor_bytes)?;
        writer.write_all(&metadata_bytes)?;

        Ok(())
    }

    /// Read from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        // Read header
        let mut header_bytes = vec![0u8; NctHeader::SIZE];
        reader.read_exact(&mut header_bytes)?;
        let header = NctHeader::from_bytes(&header_bytes)?;

        // Read codebooks
        let codebook_size = (header.tensor_offset - header.codebook_offset) as usize;
        let mut codebook_bytes = vec![0u8; codebook_size];
        reader.read_exact(&mut codebook_bytes)?;
        let codebooks = LayerCodebook::from_bytes(&codebook_bytes)?;

        // Read tensors
        let tensor_size = (header.metadata_offset - header.tensor_offset) as usize;
        let mut tensor_bytes = vec![0u8; tensor_size];
        reader.read_exact(&mut tensor_bytes)?;
        let tensors: Vec<EncodedTensor> = bincode::deserialize(&tensor_bytes)
            .map_err(|e| NeuralError::InvalidFormat(e.to_string()))?;

        // Read metadata
        let metadata_size = (header.file_size - header.metadata_offset) as usize;
        let mut metadata_bytes = vec![0u8; metadata_size];
        reader.read_exact(&mut metadata_bytes)?;
        let metadata = NctMetadata::from_bytes(&metadata_bytes)?;

        Ok(Self {
            header,
            codebooks,
            tensors,
            metadata,
        })
    }

    /// Write to file path
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)
    }

    /// Read from file path
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        Self::read_from(&mut file)
    }

    /// Get tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&EncodedTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Total compressed size
    pub fn compressed_size(&self) -> usize {
        self.tensors.iter().map(|t| t.stats.compressed_size).sum()
    }

    /// Total original size
    pub fn original_size(&self) -> usize {
        self.tensors.iter().map(|t| t.stats.original_size).sum()
    }

    /// Overall compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original = self.original_size();
        let compressed = self.compressed_size();
        if compressed > 0 {
            original as f32 / compressed as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_serialization() {
        let header = NctHeader::new(10);
        let bytes = header.to_bytes();
        let restored = NctHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.magic, restored.magic);
        assert_eq!(header.version, restored.version);
        assert_eq!(header.num_tensors, restored.num_tensors);
    }

    #[test]
    fn test_metadata() {
        let mut metadata = NctMetadata::new("test-model");
        metadata.original_size = 1000;
        metadata.compressed_size = 100;
        metadata.compression_ratio = 10.0;

        let bytes = metadata.to_bytes();
        let restored = NctMetadata::from_bytes(&bytes).unwrap();

        assert_eq!(metadata.model_id, restored.model_id);
        assert_eq!(metadata.compression_ratio, restored.compression_ratio);
    }

    #[test]
    fn test_nct_file_roundtrip() {
        use crate::{EncoderConfig, LayerType, NeuralEncoder};

        let codebooks = LayerCodebook::with_defaults("test");
        let encoder = NeuralEncoder::new(EncoderConfig::default(), codebooks.clone());

        // Create a test tensor
        let data: Vec<f32> = (0..640).map(|i| (i as f32 / 640.0) - 0.5).collect();
        let encoded = encoder
            .encode_tensor("test.weight", &data, &[10, 64], LayerType::AttentionQK)
            .unwrap();

        let metadata = NctMetadata::new("test-model");
        let nct = NctFile::new(codebooks, vec![encoded], metadata);

        // Write to buffer
        let mut buffer = Vec::new();
        nct.write_to(&mut buffer).unwrap();

        // Read back
        let mut cursor = std::io::Cursor::new(buffer);
        let restored = NctFile::read_from(&mut cursor).unwrap();

        assert_eq!(restored.tensors.len(), 1);
        assert_eq!(restored.get_tensor("test.weight").unwrap().name, "test.weight");
    }
}
