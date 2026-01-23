// PyO3 deprecation warnings - these require PyO3 version updates to fix properly
#![allow(deprecated)]
// Allow manual div_ceil in numeric code
#![allow(clippy::manual_div_ceil)]
// Unused import from numpy feature flags
#![allow(unused_imports)]
// PyO3 error handling macro false positives
#![allow(clippy::useless_conversion)]
// PyO3 cfg condition for gil-refs feature
#![allow(unexpected_cfgs)]

//! Python bindings for Haagenti tensor compression library.
//!
//! Provides:
//! - HCT format reading/writing for tensor storage
//! - HoloTensor progressive encoding/decoding
//! - LZ4/Zstd compression backends
//!
//! # Example (Python)
//! ```python
//! from haagenti import HctReader, CompressionAlgorithm, DType
//!
//! # Read an HCT file
//! reader = HctReader("model.hct")
//! header = reader.header()
//! print(f"Shape: {header.shape}, DType: {header.dtype}")
//!
//! # Decompress all data
//! data = reader.decompress_all()
//! ```

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};

// Re-exports from haagenti
use haagenti::{
    CompressionAlgorithm as RustCompressionAlgorithm, Compressor, DType as RustDType, Decompressor,
    HctHeader as RustHctHeader, HctReaderV2, HctWriterV2,
    HolographicEncoding as RustHolographicEncoding, QuantizationScheme as RustQuantizationScheme,
};

// Type aliases for V2 readers/writers
type RustHctReaderV2<R> = HctReaderV2<R>;
type RustHctWriterV2<W> = HctWriterV2<W>;

use haagenti_lz4::{Lz4Compressor, Lz4Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

// ============================================================================
// Enums
// ============================================================================

/// Compression algorithm for HCT files.
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompressionAlgorithm {
    Lz4,
    Zstd,
}

impl From<RustCompressionAlgorithm> for CompressionAlgorithm {
    fn from(algo: RustCompressionAlgorithm) -> Self {
        match algo {
            RustCompressionAlgorithm::Lz4 => CompressionAlgorithm::Lz4,
            RustCompressionAlgorithm::Zstd => CompressionAlgorithm::Zstd,
        }
    }
}

impl From<CompressionAlgorithm> for RustCompressionAlgorithm {
    fn from(algo: CompressionAlgorithm) -> Self {
        match algo {
            CompressionAlgorithm::Lz4 => RustCompressionAlgorithm::Lz4,
            CompressionAlgorithm::Zstd => RustCompressionAlgorithm::Zstd,
        }
    }
}

#[pymethods]
impl CompressionAlgorithm {
    fn __repr__(&self) -> String {
        format!("CompressionAlgorithm.{:?}", self)
    }
}

/// Data type for tensor elements.
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
}

impl From<RustDType> for DType {
    fn from(dtype: RustDType) -> Self {
        match dtype {
            RustDType::F32 => DType::F32,
            RustDType::F16 => DType::F16,
            RustDType::BF16 => DType::BF16,
            RustDType::I8 => DType::I8,
            RustDType::I4 => DType::I4,
        }
    }
}

impl From<DType> for RustDType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F32 => RustDType::F32,
            DType::F16 => RustDType::F16,
            DType::BF16 => RustDType::BF16,
            DType::I8 => RustDType::I8,
            DType::I4 => RustDType::I4,
        }
    }
}

#[pymethods]
impl DType {
    /// Bits per element for this dtype.
    fn bits(&self) -> u32 {
        match self {
            DType::F32 => 32,
            DType::F16 | DType::BF16 => 16,
            DType::I8 => 8,
            DType::I4 => 4,
        }
    }

    /// Bytes per element (rounded up for sub-byte types).
    fn bytes(&self) -> u32 {
        (self.bits() + 7) / 8
    }

    fn __repr__(&self) -> String {
        format!("DType.{:?}", self)
    }
}

/// Quantization scheme for compressed tensors.
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantizationScheme {
    None,
    GptqInt4,
    AwqInt4,
    SymmetricInt8,
    AsymmetricInt8,
}

impl From<RustQuantizationScheme> for QuantizationScheme {
    fn from(scheme: RustQuantizationScheme) -> Self {
        match scheme {
            RustQuantizationScheme::None => QuantizationScheme::None,
            RustQuantizationScheme::GptqInt4 => QuantizationScheme::GptqInt4,
            RustQuantizationScheme::AwqInt4 => QuantizationScheme::AwqInt4,
            RustQuantizationScheme::SymmetricInt8 => QuantizationScheme::SymmetricInt8,
            RustQuantizationScheme::AsymmetricInt8 => QuantizationScheme::AsymmetricInt8,
        }
    }
}

#[pymethods]
impl QuantizationScheme {
    fn __repr__(&self) -> String {
        format!("QuantizationScheme.{:?}", self)
    }
}

/// Holographic encoding type.
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HolographicEncoding {
    /// DCT-based spectral encoding (best for smooth weights)
    Spectral,
    /// Random projection hash (Johnson-Lindenstrauss)
    RandomProjection,
    /// Low-rank distributed factorization (SVD-based)
    LowRankDistributed,
}

impl From<RustHolographicEncoding> for HolographicEncoding {
    fn from(enc: RustHolographicEncoding) -> Self {
        match enc {
            RustHolographicEncoding::Spectral => HolographicEncoding::Spectral,
            RustHolographicEncoding::RandomProjection => HolographicEncoding::RandomProjection,
            RustHolographicEncoding::LowRankDistributed => HolographicEncoding::LowRankDistributed,
        }
    }
}

impl From<HolographicEncoding> for RustHolographicEncoding {
    fn from(enc: HolographicEncoding) -> Self {
        match enc {
            HolographicEncoding::Spectral => RustHolographicEncoding::Spectral,
            HolographicEncoding::RandomProjection => RustHolographicEncoding::RandomProjection,
            HolographicEncoding::LowRankDistributed => RustHolographicEncoding::LowRankDistributed,
        }
    }
}

#[pymethods]
impl HolographicEncoding {
    fn __repr__(&self) -> String {
        format!("HolographicEncoding.{:?}", self)
    }
}

// ============================================================================
// HCT Header
// ============================================================================

/// Header information for an HCT file.
#[pyclass]
#[derive(Clone)]
pub struct HctHeader {
    #[pyo3(get)]
    pub algorithm: CompressionAlgorithm,
    #[pyo3(get)]
    pub dtype: DType,
    #[pyo3(get)]
    pub shape: Vec<u64>,
    #[pyo3(get)]
    pub original_size: u64,
    #[pyo3(get)]
    pub compressed_size: u64,
    #[pyo3(get)]
    pub block_size: u32,
    #[pyo3(get)]
    pub num_blocks: u32,
}

impl From<&RustHctHeader> for HctHeader {
    fn from(header: &RustHctHeader) -> Self {
        HctHeader {
            algorithm: header.algorithm.into(),
            dtype: header.dtype.into(),
            shape: header.shape.clone(),
            original_size: header.original_size,
            compressed_size: header.compressed_size,
            block_size: header.block_size,
            num_blocks: header.num_blocks,
        }
    }
}

#[pymethods]
impl HctHeader {
    /// Total number of elements in the tensor.
    fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Compression ratio (original / compressed).
    fn compression_ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            0.0
        } else {
            self.original_size as f64 / self.compressed_size as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HctHeader(dtype={:?}, shape={:?}, ratio={:.2}x)",
            self.dtype,
            self.shape,
            self.compression_ratio()
        )
    }
}

// ============================================================================
// HCT Reader
// ============================================================================

/// Reader for HCT (Haagenti Compressed Tensor) files.
///
/// Supports both V1 and V2 formats with optional checksum validation.
#[pyclass]
pub struct HctReader {
    reader: HctReaderV2<BufReader<File>>,
    path: String,
}

#[pymethods]
impl HctReader {
    /// Open an HCT file for reading.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open {}: {}", path, e)))?;
        let buf_reader = BufReader::new(file);
        let reader = RustHctReaderV2::new(buf_reader)
            .map_err(|e| PyIOError::new_err(format!("Failed to read HCT header: {}", e)))?;
        Ok(HctReader {
            reader,
            path: path.to_string(),
        })
    }

    /// Get the file header.
    fn header(&self) -> HctHeader {
        HctHeader::from(self.reader.header())
    }

    /// Number of compressed blocks.
    fn num_blocks(&self) -> usize {
        self.reader.num_blocks()
    }

    /// Decompress all blocks and return as numpy array (float32).
    fn decompress_all<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // Clone header to avoid borrowing issues
        let algorithm = self.reader.header().algorithm;
        let dtype = self.reader.header().dtype;

        // Create appropriate decompressor
        let data = match algorithm {
            RustCompressionAlgorithm::Lz4 => {
                let decompressor = Lz4Decompressor::new();
                self.reader
                    .decompress_all_validated(&decompressor)
                    .map_err(|e| PyIOError::new_err(format!("Decompression failed: {}", e)))?
            }
            RustCompressionAlgorithm::Zstd => {
                let decompressor = ZstdDecompressor::new();
                self.reader
                    .decompress_all_validated(&decompressor)
                    .map_err(|e| PyIOError::new_err(format!("Decompression failed: {}", e)))?
            }
        };

        // Convert bytes to f32 based on dtype
        let floats = bytes_to_f32(&data, dtype)?;
        Ok(floats.into_pyarray_bound(py))
    }

    /// Decompress a single block by index.
    fn decompress_block<'py>(
        &mut self,
        py: Python<'py>,
        block_idx: usize,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let algorithm = self.reader.header().algorithm;

        let data = match algorithm {
            RustCompressionAlgorithm::Lz4 => {
                let decompressor = Lz4Decompressor::new();
                self.reader
                    .decompress_block_validated(block_idx, &decompressor)
                    .map_err(|e| PyIOError::new_err(format!("Block decompression failed: {}", e)))?
            }
            RustCompressionAlgorithm::Zstd => {
                let decompressor = ZstdDecompressor::new();
                self.reader
                    .decompress_block_validated(block_idx, &decompressor)
                    .map_err(|e| PyIOError::new_err(format!("Block decompression failed: {}", e)))?
            }
        };

        Ok(data.into_pyarray_bound(py))
    }

    /// Validate all block checksums (V2 only).
    fn validate_checksums(&mut self) -> PyResult<()> {
        self.reader
            .validate_checksums()
            .map_err(|e| PyValueError::new_err(format!("Checksum validation failed: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!("HctReader('{}', blocks={})", self.path, self.num_blocks())
    }
}

// ============================================================================
// HCT Writer
// ============================================================================

/// Writer for HCT (Haagenti Compressed Tensor) files.
#[pyclass]
pub struct HctWriter {
    writer: Option<HctWriterV2<BufWriter<File>>>,
    path: String,
    algorithm: CompressionAlgorithm,
}

#[pymethods]
impl HctWriter {
    /// Create a new HCT file for writing.
    #[new]
    #[pyo3(signature = (path, algorithm, dtype, shape, block_size=None))]
    fn new(
        path: &str,
        algorithm: CompressionAlgorithm,
        dtype: DType,
        shape: Vec<u64>,
        block_size: Option<u32>,
    ) -> PyResult<Self> {
        let file = File::create(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to create {}: {}", path, e)))?;
        let buf_writer = BufWriter::new(file);

        let mut writer = RustHctWriterV2::new(buf_writer, algorithm.into(), dtype.into(), shape);

        if let Some(bs) = block_size {
            writer = writer.with_block_size(bs);
        }

        Ok(HctWriter {
            writer: Some(writer),
            path: path.to_string(),
            algorithm,
        })
    }

    /// Compress and write data from a numpy array.
    fn compress_data(&mut self, data: PyReadonlyArray1<f32>) -> PyResult<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Writer already finalized"))?;

        let slice = data.as_slice()?;
        let bytes: Vec<u8> = slice.iter().flat_map(|f| f.to_le_bytes()).collect();

        match self.algorithm {
            CompressionAlgorithm::Lz4 => {
                let compressor = Lz4Compressor::new();
                writer
                    .compress_data(&bytes, &compressor)
                    .map_err(|e| PyIOError::new_err(format!("Compression failed: {}", e)))?;
            }
            CompressionAlgorithm::Zstd => {
                let compressor = ZstdCompressor::new(); // Level 3 compression
                writer
                    .compress_data(&bytes, &compressor)
                    .map_err(|e| PyIOError::new_err(format!("Compression failed: {}", e)))?;
            }
        }

        Ok(())
    }

    /// Finalize the file and flush to disk.
    fn finish(&mut self) -> PyResult<()> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| PyValueError::new_err("Writer already finalized"))?;

        writer
            .finish()
            .map_err(|e| PyIOError::new_err(format!("Failed to finalize: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!("HctWriter('{}')", self.path)
    }
}

// ============================================================================
// HoloTensor Encoder (Phase 4 - Progressive Loading)
// ============================================================================

/// Encoder for HoloTensor (progressive tensor loading).
///
/// Supports three encoding schemes:
/// - Spectral: DCT-based, best for smooth weights (attention, embeddings)
/// - RandomProjection: Johnson-Lindenstrauss, good for dense layers
/// - LowRankDistributed: SVD-based, best for low-rank matrices
///
/// Example:
///     encoder = HoloTensorEncoder(HolographicEncoding.Spectral, n_fragments=8)
///     header_bytes, fragment_list = encoder.encode_2d(weights, 4096, 4096)
#[pyclass]
pub struct HoloTensorEncoder {
    encoder: haagenti::HoloTensorEncoder,
    encoding: HolographicEncoding,
    n_fragments: u16,
}

#[pymethods]
impl HoloTensorEncoder {
    /// Create a new HoloTensor encoder.
    ///
    /// Args:
    ///     encoding: Holographic encoding scheme
    ///     n_fragments: Number of fragments to create (default 8)
    ///     seed: Random seed for deterministic encoding
    ///     essential_ratio: Ratio of essential data in first fragment (0.01-0.5)
    ///     max_rank: Maximum rank for LRDF encoding
    #[new]
    #[pyo3(signature = (encoding, n_fragments=None, seed=None, essential_ratio=None, max_rank=None))]
    fn new(
        encoding: HolographicEncoding,
        n_fragments: Option<u16>,
        seed: Option<u64>,
        essential_ratio: Option<f32>,
        max_rank: Option<usize>,
    ) -> Self {
        let n_frags = n_fragments.unwrap_or(8);
        let mut encoder = haagenti::HoloTensorEncoder::new(encoding.into()).with_fragments(n_frags);

        if let Some(s) = seed {
            encoder = encoder.with_seed(s);
        }
        if let Some(r) = essential_ratio {
            encoder = encoder.with_essential_ratio(r);
        }
        if let Some(r) = max_rank {
            encoder = encoder.with_max_rank(r);
        }

        HoloTensorEncoder {
            encoder,
            encoding,
            n_fragments: n_frags,
        }
    }

    /// Encode a 2D tensor (matrix) into holographic fragments.
    ///
    /// Args:
    ///     data: Flattened tensor data (float32)
    ///     rows: Number of rows
    ///     cols: Number of columns
    ///
    /// Returns:
    ///     Tuple of (header_bytes, list of fragment_bytes)
    fn encode_2d(
        &self,
        data: PyReadonlyArray1<f32>,
        rows: usize,
        cols: usize,
    ) -> PyResult<(HoloTensorHeaderPy, Vec<HoloFragmentPy>)> {
        let slice = data.as_slice()?;

        if slice.len() != rows * cols {
            return Err(PyValueError::new_err(format!(
                "Data length {} doesn't match {}x{}={}",
                slice.len(),
                rows,
                cols,
                rows * cols
            )));
        }

        let (header, fragments) = self
            .encoder
            .encode_2d(slice, rows, cols)
            .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?;

        // Convert to Python types
        let header_py = HoloTensorHeaderPy::from(&header);
        let fragments_py: Vec<HoloFragmentPy> =
            fragments.into_iter().map(HoloFragmentPy::from).collect();

        Ok((header_py, fragments_py))
    }

    /// Encode a 1D tensor (vector) into holographic fragments.
    fn encode_1d(
        &self,
        data: PyReadonlyArray1<f32>,
    ) -> PyResult<(HoloTensorHeaderPy, Vec<HoloFragmentPy>)> {
        let slice = data.as_slice()?;

        let (header, fragments) = self
            .encoder
            .encode_1d(slice)
            .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?;

        let header_py = HoloTensorHeaderPy::from(&header);
        let fragments_py: Vec<HoloFragmentPy> =
            fragments.into_iter().map(HoloFragmentPy::from).collect();

        Ok((header_py, fragments_py))
    }

    /// Get the encoding scheme.
    #[getter]
    fn encoding(&self) -> HolographicEncoding {
        self.encoding
    }

    /// Get the number of fragments.
    #[getter]
    fn n_fragments(&self) -> u16 {
        self.n_fragments
    }

    fn __repr__(&self) -> String {
        format!(
            "HoloTensorEncoder(encoding={:?}, n_fragments={})",
            self.encoding, self.n_fragments
        )
    }
}

// ============================================================================
// HoloTensor Decoder (Phase 4 - Progressive Loading)
// ============================================================================

/// Decoder for HoloTensor with progressive reconstruction.
///
/// Allows loading fragments incrementally and reconstructing
/// the tensor at any quality level. Quality improves as more
/// fragments are added.
///
/// Example:
/// ```python
/// decoder = HoloTensorDecoder(header)
/// decoder.add_fragment(fragments[0])  # ~30% quality
/// decoder.add_fragment(fragments[1])  # ~50% quality
/// weights = decoder.reconstruct()
/// ```
#[pyclass]
pub struct HoloTensorDecoder {
    decoder: haagenti::HoloTensorDecoder,
    header: HoloTensorHeaderPy,
}

#[pymethods]
impl HoloTensorDecoder {
    /// Create a decoder from a header.
    #[new]
    fn new(header: HoloTensorHeaderPy) -> PyResult<Self> {
        let rust_header = header.to_rust_header()?;
        Ok(HoloTensorDecoder {
            decoder: haagenti::HoloTensorDecoder::new(rust_header),
            header,
        })
    }

    /// Add a fragment to the reconstruction.
    ///
    /// Returns the new quality level (0.0-1.0).
    fn add_fragment(&mut self, fragment: &HoloFragmentPy) -> PyResult<f32> {
        let rust_fragment = fragment.to_rust_fragment();
        self.decoder
            .add_fragment(rust_fragment)
            .map_err(|e| PyValueError::new_err(format!("Failed to add fragment: {}", e)))
    }

    /// Current reconstruction quality (0.0-1.0).
    ///
    /// Quality represents how close the reconstruction is to the original.
    /// 1.0 means perfect reconstruction (all fragments loaded).
    #[getter]
    fn quality(&self) -> f32 {
        self.decoder.quality()
    }

    /// Number of fragments loaded so far.
    #[getter]
    fn fragments_loaded(&self) -> u16 {
        self.decoder.fragments_loaded()
    }

    /// Total number of fragments.
    #[getter]
    fn total_fragments(&self) -> u16 {
        self.header.total_fragments
    }

    /// Check if minimum fragments for reconstruction are loaded.
    fn can_reconstruct(&self) -> bool {
        self.decoder.can_reconstruct()
    }

    /// Reconstruct the tensor from loaded fragments.
    ///
    /// Returns a numpy array with the reconstructed weights.
    /// Quality depends on how many fragments have been loaded.
    fn reconstruct<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let data = self
            .decoder
            .reconstruct()
            .map_err(|e| PyValueError::new_err(format!("Reconstruction failed: {}", e)))?;
        Ok(data.into_pyarray_bound(py))
    }

    /// Get the header.
    #[getter]
    fn header(&self) -> HoloTensorHeaderPy {
        self.header.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "HoloTensorDecoder(quality={:.1}%, fragments={}/{})",
            self.quality() * 100.0,
            self.fragments_loaded(),
            self.total_fragments()
        )
    }
}

// ============================================================================
// HoloTensor Header (Python wrapper)
// ============================================================================

/// Header for a HoloTensor file.
#[pyclass]
#[derive(Clone)]
pub struct HoloTensorHeaderPy {
    #[pyo3(get)]
    pub encoding: HolographicEncoding,
    #[pyo3(get)]
    pub total_fragments: u16,
    #[pyo3(get)]
    pub min_fragments: u16,
    #[pyo3(get)]
    pub shape: Vec<u64>,
    #[pyo3(get)]
    pub original_size: u64,
    #[pyo3(get)]
    pub seed: u64,
}

impl From<&haagenti::HoloTensorHeader> for HoloTensorHeaderPy {
    fn from(h: &haagenti::HoloTensorHeader) -> Self {
        HoloTensorHeaderPy {
            encoding: h.encoding.into(),
            total_fragments: h.total_fragments,
            min_fragments: h.min_fragments,
            shape: h.shape.clone(),
            original_size: h.original_size,
            seed: h.seed,
        }
    }
}

impl HoloTensorHeaderPy {
    fn to_rust_header(&self) -> PyResult<haagenti::HoloTensorHeader> {
        Ok(haagenti::HoloTensorHeader {
            encoding: self.encoding.into(),
            compression: RustCompressionAlgorithm::Lz4,
            flags: 0,
            total_fragments: self.total_fragments,
            min_fragments: self.min_fragments,
            original_size: self.original_size,
            seed: self.seed,
            dtype: RustDType::F32,
            shape: self.shape.clone(),
            quality_curve: haagenti::QualityCurve::default(),
            quantization: None,
        })
    }
}

#[pymethods]
impl HoloTensorHeaderPy {
    fn __repr__(&self) -> String {
        format!(
            "HoloTensorHeader(encoding={:?}, fragments={}, shape={:?})",
            self.encoding, self.total_fragments, self.shape
        )
    }
}

// ============================================================================
// HoloTensor Fragment (Python wrapper)
// ============================================================================

/// A fragment of a HoloTensor.
///
/// Each fragment contains information about the whole tensor.
/// Any subset of fragments can reconstruct an approximation.
#[pyclass]
#[derive(Clone)]
pub struct HoloFragmentPy {
    #[pyo3(get)]
    pub index: u16,
    #[pyo3(get)]
    pub flags: u16,
    #[pyo3(get)]
    pub checksum: u64,
    data: Vec<u8>,
}

impl From<haagenti::HoloFragment> for HoloFragmentPy {
    fn from(f: haagenti::HoloFragment) -> Self {
        HoloFragmentPy {
            index: f.index,
            flags: f.flags,
            checksum: f.checksum,
            data: f.data,
        }
    }
}

impl HoloFragmentPy {
    fn to_rust_fragment(&self) -> haagenti::HoloFragment {
        haagenti::HoloFragment {
            index: self.index,
            flags: self.flags,
            checksum: self.checksum,
            data: self.data.clone(),
        }
    }
}

#[pymethods]
impl HoloFragmentPy {
    /// Get fragment data as bytes.
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.data.clone().into_pyarray_bound(py)
    }

    /// Size of fragment data in bytes.
    #[getter]
    fn size(&self) -> usize {
        self.data.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "HoloFragment(index={}, size={})",
            self.index,
            self.data.len()
        )
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert safetensors file to HCT format.
#[pyfunction]
#[pyo3(signature = (input_path, output_path, algorithm=CompressionAlgorithm::Lz4))]
fn convert_safetensors_to_hct(
    input_path: &str,
    output_path: &str,
    algorithm: CompressionAlgorithm,
) -> PyResult<(u64, u64, f64)> {
    // Read safetensors file
    let data = std::fs::read(input_path)
        .map_err(|e| PyIOError::new_err(format!("Failed to read {}: {}", input_path, e)))?;

    // Parse safetensors header (JSON + tensors)
    // For now, we just compress the raw bytes - real implementation would parse properly
    let original_size = data.len() as u64;

    // Create HCT writer
    let file = File::create(output_path)
        .map_err(|e| PyIOError::new_err(format!("Failed to create {}: {}", output_path, e)))?;
    let buf_writer = BufWriter::new(file);

    let mut writer = RustHctWriterV2::new(
        buf_writer,
        algorithm.into(),
        RustDType::F32,          // Default to F32 for safetensors
        vec![data.len() as u64], // Treat as 1D for raw conversion
    );

    match algorithm {
        CompressionAlgorithm::Lz4 => {
            let compressor = Lz4Compressor::new();
            writer
                .compress_data(&data, &compressor)
                .map_err(|e| PyIOError::new_err(format!("Compression failed: {}", e)))?;
        }
        CompressionAlgorithm::Zstd => {
            let compressor = ZstdCompressor::new();
            writer
                .compress_data(&data, &compressor)
                .map_err(|e| PyIOError::new_err(format!("Compression failed: {}", e)))?;
        }
    }

    writer
        .finish()
        .map_err(|e| PyIOError::new_err(format!("Failed to finalize: {}", e)))?;

    // Get compressed size
    let compressed_size = std::fs::metadata(output_path)
        .map_err(|e| PyIOError::new_err(format!("Failed to stat {}: {}", output_path, e)))?
        .len();

    let ratio = original_size as f64 / compressed_size as f64;

    Ok((original_size, compressed_size, ratio))
}

/// Get version information.
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Top-Level Compression Functions (C.5)
// ============================================================================

/// Compress data using the specified algorithm.
///
/// Args:
///     data: Bytes to compress
///     algorithm: Compression algorithm ("zstd" or "lz4")
///     level: Compression level ("fast", "default", or "best")
///     dictionary: Optional ZstdDict for dictionary compression
///
/// Returns:
///     Compressed bytes
#[pyfunction]
#[pyo3(signature = (data, algorithm="zstd", level="default", dictionary=None))]
fn compress(
    data: &[u8],
    algorithm: &str,
    level: &str,
    dictionary: Option<&ZstdDict>,
) -> PyResult<Vec<u8>> {
    let _ = dictionary; // Dictionary support is future work

    let compression_level = match level {
        "fast" => haagenti_core::CompressionLevel::Fast,
        "default" => haagenti_core::CompressionLevel::Default,
        "best" => haagenti_core::CompressionLevel::Best,
        _ => return Err(PyValueError::new_err(format!("Invalid level: {}", level))),
    };

    match algorithm.to_lowercase().as_str() {
        "zstd" => {
            let compressor = ZstdCompressor::with_level(compression_level);
            compressor
                .compress(data)
                .map_err(|e| PyValueError::new_err(format!("Zstd compression failed: {}", e)))
        }
        "lz4" => {
            let compressor = Lz4Compressor::new();
            compressor
                .compress(data)
                .map_err(|e| PyValueError::new_err(format!("LZ4 compression failed: {}", e)))
        }
        _ => Err(PyValueError::new_err(format!(
            "Invalid algorithm: {}. Use 'zstd' or 'lz4'",
            algorithm
        ))),
    }
}

/// Decompress data using the specified algorithm.
///
/// Args:
///     data: Compressed bytes
///     algorithm: Compression algorithm ("zstd" or "lz4")
///
/// Returns:
///     Decompressed bytes
#[pyfunction]
#[pyo3(signature = (data, algorithm="zstd"))]
fn decompress(data: &[u8], algorithm: &str) -> PyResult<Vec<u8>> {
    match algorithm.to_lowercase().as_str() {
        "zstd" => {
            let decompressor = ZstdDecompressor::new();
            decompressor.decompress(data).map_err(|e| {
                DecompressionError::new_err(format!("Zstd decompression failed: {}", e))
            })
        }
        "lz4" => {
            let decompressor = Lz4Decompressor::new();
            decompressor.decompress(data).map_err(|e| {
                DecompressionError::new_err(format!("LZ4 decompression failed: {}", e))
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "Invalid algorithm: {}. Use 'zstd' or 'lz4'",
            algorithm
        ))),
    }
}

// ============================================================================
// Zstd Dictionary Support (C.5)
// ============================================================================

/// A trained Zstd dictionary for improved compression.
#[pyclass]
#[derive(Clone)]
pub struct ZstdDict {
    id: u32,
    data: Vec<u8>,
}

#[pymethods]
impl ZstdDict {
    /// Train a dictionary from sample data.
    ///
    /// Args:
    ///     samples: List of bytes samples to train on
    ///     max_size: Maximum dictionary size in bytes
    ///
    /// Returns:
    ///     Trained ZstdDict
    #[staticmethod]
    #[pyo3(signature = (samples, max_size=8192))]
    fn train(samples: Vec<Vec<u8>>, max_size: usize) -> PyResult<Self> {
        use haagenti_zstd::ZstdDictionary;

        if samples.len() < 5 {
            return Err(PyValueError::new_err(
                "Need at least 5 samples for dictionary training",
            ));
        }

        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = ZstdDictionary::train(&sample_refs, max_size)
            .map_err(|e| PyValueError::new_err(format!("Dictionary training failed: {}", e)))?;

        Ok(ZstdDict {
            id: dict.id(),
            data: dict.serialize(),
        })
    }

    /// Dictionary ID.
    #[getter]
    fn id(&self) -> u32 {
        self.id
    }

    /// Get dictionary as bytes.
    fn as_bytes(&self) -> Vec<u8> {
        self.data.clone()
    }

    fn __repr__(&self) -> String {
        format!("ZstdDict(id={}, size={})", self.id, self.data.len())
    }
}

// ============================================================================
// Streaming Encoder/Decoder (C.5)
// ============================================================================

/// Streaming encoder for incremental compression.
#[pyclass]
pub struct StreamingEncoder {
    algorithm: String,
    buffer: Vec<u8>,
}

#[pymethods]
impl StreamingEncoder {
    /// Create a new streaming encoder.
    #[new]
    fn new(algorithm: &str) -> PyResult<Self> {
        match algorithm.to_lowercase().as_str() {
            "zstd" | "lz4" => Ok(StreamingEncoder {
                algorithm: algorithm.to_lowercase(),
                buffer: Vec::new(),
            }),
            _ => Err(PyValueError::new_err(format!(
                "Invalid algorithm: {}",
                algorithm
            ))),
        }
    }

    /// Write data to the encoder.
    fn write(&mut self, data: &[u8]) -> PyResult<()> {
        self.buffer.extend_from_slice(data);
        Ok(())
    }

    /// Finish encoding and return compressed data.
    fn finish(&mut self) -> PyResult<Vec<u8>> {
        let result = match self.algorithm.as_str() {
            "zstd" => {
                let compressor = ZstdCompressor::new();
                compressor
                    .compress(&self.buffer)
                    .map_err(|e| PyValueError::new_err(format!("Compression failed: {}", e)))?
            }
            "lz4" => {
                let compressor = Lz4Compressor::new();
                compressor
                    .compress(&self.buffer)
                    .map_err(|e| PyValueError::new_err(format!("Compression failed: {}", e)))?
            }
            _ => return Err(PyValueError::new_err("Invalid algorithm")),
        };
        self.buffer.clear();
        Ok(result)
    }

    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> bool {
        false
    }
}

/// Streaming decoder for incremental decompression.
#[pyclass]
pub struct StreamingDecoder {
    algorithm: String,
    buffer: Vec<u8>,
}

#[pymethods]
impl StreamingDecoder {
    /// Create a new streaming decoder.
    #[new]
    fn new(algorithm: &str) -> PyResult<Self> {
        match algorithm.to_lowercase().as_str() {
            "zstd" | "lz4" => Ok(StreamingDecoder {
                algorithm: algorithm.to_lowercase(),
                buffer: Vec::new(),
            }),
            _ => Err(PyValueError::new_err(format!(
                "Invalid algorithm: {}",
                algorithm
            ))),
        }
    }

    /// Write compressed data to the decoder.
    ///
    /// Returns any decompressed data available (may be empty).
    fn write(&mut self, data: &[u8]) -> PyResult<Option<Vec<u8>>> {
        self.buffer.extend_from_slice(data);
        Ok(None) // Streaming decompression returns data on finish
    }

    /// Finish decoding and return remaining data.
    fn finish(&mut self) -> PyResult<Vec<u8>> {
        let result = match self.algorithm.as_str() {
            "zstd" => {
                let decompressor = ZstdDecompressor::new();
                decompressor.decompress(&self.buffer).map_err(|e| {
                    DecompressionError::new_err(format!("Decompression failed: {}", e))
                })?
            }
            "lz4" => {
                let decompressor = Lz4Decompressor::new();
                decompressor.decompress(&self.buffer).map_err(|e| {
                    DecompressionError::new_err(format!("Decompression failed: {}", e))
                })?
            }
            _ => return Err(PyValueError::new_err("Invalid algorithm")),
        };
        self.buffer.clear();
        Ok(result)
    }

    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> bool {
        false
    }
}

// ============================================================================
// Custom Exception Types (C.5)
// ============================================================================

pyo3::create_exception!(haagenti, DecompressionError, pyo3::exceptions::PyException);

// ============================================================================
// Helper Functions
// ============================================================================

fn bytes_to_f32(data: &[u8], dtype: RustDType) -> PyResult<Vec<f32>> {
    match dtype {
        RustDType::F32 => {
            if data.len() % 4 != 0 {
                return Err(PyValueError::new_err("Invalid F32 data length"));
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        RustDType::F16 => {
            if data.len() % 2 != 0 {
                return Err(PyValueError::new_err("Invalid F16 data length"));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect())
        }
        RustDType::BF16 => {
            if data.len() % 2 != 0 {
                return Err(PyValueError::new_err("Invalid BF16 data length"));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect())
        }
        RustDType::I8 => Ok(data.iter().map(|&b| b as i8 as f32).collect()),
        RustDType::I4 => {
            // Unpack 4-bit values
            Ok(data
                .iter()
                .flat_map(|&b| {
                    let lo = (b & 0x0F) as i8;
                    let hi = ((b >> 4) & 0x0F) as i8;
                    // Sign-extend 4-bit to 8-bit
                    let lo = if lo & 0x08 != 0 {
                        lo | 0xF0u8 as i8
                    } else {
                        lo
                    };
                    let hi = if hi & 0x08 != 0 {
                        hi | 0xF0u8 as i8
                    } else {
                        hi
                    };
                    vec![lo as f32, hi as f32]
                })
                .collect())
        }
    }
}

// ============================================================================
// Python Module
// ============================================================================

/// Haagenti Python bindings for tensor compression.
#[pymodule]
fn _haagenti_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Enums
    m.add_class::<CompressionAlgorithm>()?;
    m.add_class::<DType>()?;
    m.add_class::<QuantizationScheme>()?;
    m.add_class::<HolographicEncoding>()?;

    // HCT classes
    m.add_class::<HctHeader>()?;
    m.add_class::<HctReader>()?;
    m.add_class::<HctWriter>()?;

    // HoloTensor classes
    m.add_class::<HoloTensorEncoder>()?;
    m.add_class::<HoloTensorDecoder>()?;
    m.add_class::<HoloTensorHeaderPy>()?;
    m.add_class::<HoloFragmentPy>()?;

    // C.5: Compression classes
    m.add_class::<ZstdDict>()?;
    m.add_class::<StreamingEncoder>()?;
    m.add_class::<StreamingDecoder>()?;

    // Functions
    m.add_function(wrap_pyfunction!(convert_safetensors_to_hct, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;

    // C.5: Custom exceptions
    m.add(
        "DecompressionError",
        m.py().get_type_bound::<DecompressionError>(),
    )?;

    // Create streaming submodule
    let streaming = PyModule::new_bound(m.py(), "streaming")?;
    streaming.add_class::<StreamingEncoder>()?;
    streaming.add_class::<StreamingDecoder>()?;
    m.add_submodule(&streaming)?;

    Ok(())
}
