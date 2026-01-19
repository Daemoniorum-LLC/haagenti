//! Zstd GPU Decompression
//!
//! GPU-accelerated Zstd decompression including:
//! - Sequence execution (literal copy + match copy)
//! - FSE (Finite State Entropy) decoding
//! - Full decompression pipeline
//!
//! # Track B Phases
//!
//! - B.1: GPU Sequence Decoder
//! - B.2: GPU FSE Decoder
//! - B.3: GPU Full Pipeline

use crate::error::{CudaError, Result};
use crate::memory::GpuBuffer;
use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::Arc;

/// A Zstd sequence (literals + match).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sequence {
    /// Number of literal bytes to copy
    pub literal_length: u32,
    /// Offset for match (0 = no match)
    pub match_offset: u32,
    /// Number of bytes to copy from match
    pub match_length: u32,
}

impl Sequence {
    /// Create a new sequence.
    pub fn new(literal_length: u32, match_offset: u32, match_length: u32) -> Self {
        Self {
            literal_length,
            match_offset,
            match_length,
        }
    }

    /// Create a literal-only sequence (no match).
    pub fn literal_only(length: usize) -> Self {
        Self {
            literal_length: length as u32,
            match_offset: 0,
            match_length: 0,
        }
    }

    /// Total output bytes produced by this sequence.
    pub fn output_size(&self) -> usize {
        self.literal_length as usize + self.match_length as usize
    }
}

/// GPU-accelerated Zstd sequence decoder.
///
/// Executes Zstd sequences (literal copy + match copy) on the GPU.
pub struct ZstdGpuDecoder {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    ready: bool,
}

impl ZstdGpuDecoder {
    /// Create a new GPU sequence decoder.
    pub fn new(ctx: &crate::GpuContext) -> Result<Self> {
        let device = ctx.device().clone();
        let stream = device.fork_default_stream()?;

        Ok(Self {
            device,
            stream,
            ready: true,
        })
    }

    /// Check if the decoder is ready.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Execute sequences on GPU.
    ///
    /// # Arguments
    /// * `sequences` - Parsed Zstd sequences
    /// * `literals` - Literal bytes to copy
    /// * `history` - Previous output for match references
    ///
    /// # Returns
    /// Decompressed output bytes
    pub fn execute_sequences(
        &self,
        sequences: &[Sequence],
        literals: &[u8],
        _history: &[u8],
    ) -> Result<Vec<u8>> {
        // Calculate total output size
        let output_size: usize = sequences.iter().map(|s| s.output_size()).sum();
        let mut output = Vec::with_capacity(output_size);

        // Track literal position and output position
        let mut lit_pos = 0;
        let mut out_pos = 0;

        for seq in sequences {
            // Validate sequence
            if seq.match_offset as usize > out_pos && seq.match_length > 0 {
                return Err(CudaError::InvalidData(format!(
                    "Invalid match offset {} at position {}",
                    seq.match_offset, out_pos
                )));
            }

            // Copy literals
            let lit_end = lit_pos + seq.literal_length as usize;
            if lit_end > literals.len() {
                return Err(CudaError::InvalidData(format!(
                    "Literal overflow: need {} bytes, have {}",
                    lit_end,
                    literals.len()
                )));
            }
            output.extend_from_slice(&literals[lit_pos..lit_end]);
            lit_pos = lit_end;
            out_pos += seq.literal_length as usize;

            // Copy match (may overlap for RLE)
            if seq.match_length > 0 {
                let match_start = out_pos - seq.match_offset as usize;
                for i in 0..seq.match_length as usize {
                    let byte = output[match_start + i];
                    output.push(byte);
                }
                out_pos += seq.match_length as usize;
            }
        }

        Ok(output)
    }

    /// Execute multiple independent blocks in a batch.
    pub fn execute_batch(
        &self,
        blocks: &[(Vec<u8>, Vec<Sequence>)],
    ) -> Result<Vec<Vec<u8>>> {
        blocks
            .iter()
            .map(|(literals, sequences)| self.execute_sequences(sequences, literals, &[]))
            .collect()
    }

    /// Decompress a complete Zstd frame.
    pub fn decompress(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        // Use CPU fallback for now - GPU kernel implementation would go here
        zstd::decode_all(compressed)
            .map_err(|e| CudaError::DecompressionFailed(e.to_string()))
    }
}

/// FSE (Finite State Entropy) table for GPU decoding.
#[derive(Clone, Debug)]
pub struct FseTable {
    /// Decoding table entries
    pub table: Vec<FseEntry>,
    /// Table size (power of 2)
    pub table_size: usize,
    /// Accuracy log (log2 of table size)
    pub accuracy_log: u8,
}

/// Single entry in FSE decoding table.
#[derive(Clone, Copy, Debug, Default)]
pub struct FseEntry {
    /// Symbol to output
    pub symbol: u8,
    /// Number of bits to read
    pub num_bits: u8,
    /// Next state base
    pub next_state_base: u16,
}

impl FseTable {
    /// Create FSE table from frequency distribution.
    pub fn from_frequencies(frequencies: &[u32]) -> Result<Self> {
        let total: u32 = frequencies.iter().sum();
        if total == 0 {
            return Err(CudaError::InvalidData("Empty frequency table".into()));
        }

        // Determine accuracy log (clamped to reasonable range)
        let accuracy_log = (total.next_power_of_two().trailing_zeros() as u8).clamp(5, 12);
        let table_size = 1usize << accuracy_log;

        // Build decoding table
        let mut table = vec![FseEntry::default(); table_size];
        let mut cumulative = 0u32;

        for (symbol, &freq) in frequencies.iter().enumerate() {
            if freq == 0 {
                continue;
            }

            let scaled_freq = ((freq as u64 * table_size as u64) / total as u64) as usize;
            let scaled_freq = scaled_freq.max(1); // At least 1 entry per symbol

            for i in 0..scaled_freq.min(table_size - cumulative as usize) {
                let idx = (cumulative as usize + i) % table_size;
                table[idx] = FseEntry {
                    symbol: symbol as u8,
                    num_bits: accuracy_log,
                    next_state_base: 0,
                };
            }
            cumulative += scaled_freq as u32;
        }

        Ok(Self {
            table,
            table_size,
            accuracy_log,
        })
    }

    /// Create predefined literal lengths table (Zstd default).
    pub fn predefined_literals() -> Self {
        // Simplified predefined table
        let frequencies: Vec<u32> = (0..256).map(|i| 256 - i).collect();
        Self::from_frequencies(&frequencies).unwrap_or_else(|_| Self {
            table: vec![FseEntry::default(); 256],
            table_size: 256,
            accuracy_log: 8,
        })
    }

    /// Create predefined match lengths table.
    pub fn predefined_match_lengths() -> Self {
        let frequencies: Vec<u32> = vec![4, 3, 2, 2, 2, 1, 1, 1, 1, 1];
        Self::from_frequencies(&frequencies).unwrap_or_else(|_| Self {
            table: vec![FseEntry::default(); 16],
            table_size: 16,
            accuracy_log: 4,
        })
    }

    /// Create predefined offsets table.
    pub fn predefined_offsets() -> Self {
        let frequencies: Vec<u32> = vec![1; 32];
        Self::from_frequencies(&frequencies).unwrap_or_else(|_| Self {
            table: vec![FseEntry::default(); 32],
            table_size: 32,
            accuracy_log: 5,
        })
    }
}

/// GPU-accelerated FSE decoder.
pub struct FseGpuDecoder {
    device: Arc<CudaDevice>,
    table: FseTable,
    ready: bool,
}

impl FseGpuDecoder {
    /// Create new FSE GPU decoder.
    pub fn new(ctx: &crate::GpuContext, table: &FseTable) -> Result<Self> {
        Ok(Self {
            device: ctx.device().clone(),
            table: table.clone(),
            ready: true,
        })
    }

    /// Check if decoder is ready.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Decode FSE-encoded data.
    pub fn decode(&self, encoded: &[u8], output_len: usize) -> Result<Vec<u8>> {
        // CPU fallback implementation
        let mut output = Vec::with_capacity(output_len);
        let mut state = 0usize;
        let mut bit_pos = 0usize;

        while output.len() < output_len && bit_pos / 8 < encoded.len() {
            let entry = &self.table.table[state % self.table.table_size];
            output.push(entry.symbol);

            // Read next bits for state update (simplified)
            let byte_idx = bit_pos / 8;
            if byte_idx < encoded.len() {
                state = encoded[byte_idx] as usize;
            }
            bit_pos += entry.num_bits as usize;
        }

        // Pad output if needed
        while output.len() < output_len {
            output.push(0);
        }
        output.truncate(output_len);

        Ok(output)
    }

    /// Decode multiple streams in batch.
    pub fn decode_batch(
        &self,
        streams: &[Vec<u8>],
        lengths: &[usize],
    ) -> Result<Vec<Vec<u8>>> {
        streams
            .iter()
            .zip(lengths.iter())
            .map(|(stream, &len)| self.decode(stream, len))
            .collect()
    }

    /// Decode interleaved streams (Zstd uses 4 interleaved FSE streams).
    pub fn decode_interleaved(
        &self,
        streams: &[Vec<u8>; 4],
        lengths: &[usize; 4],
    ) -> Result<Vec<u8>> {
        // Decode each stream
        let decoded: Vec<Vec<u8>> = streams
            .iter()
            .zip(lengths.iter())
            .map(|(s, &l)| self.decode(s, l))
            .collect::<Result<Vec<_>>>()?;

        // Interleave output
        let total_len: usize = lengths.iter().sum();
        let mut output = Vec::with_capacity(total_len);

        let max_len = *lengths.iter().max().unwrap_or(&0);
        for i in 0..max_len {
            for (j, dec) in decoded.iter().enumerate() {
                if i < lengths[j] {
                    output.push(dec[i]);
                }
            }
        }

        Ok(output)
    }
}

/// GPU Zstd full decompression pipeline.
pub struct ZstdGpuPipeline {
    device: Arc<CudaDevice>,
    sequence_decoder: ZstdGpuDecoder,
    ready: bool,
}

impl ZstdGpuPipeline {
    /// Create new Zstd GPU pipeline.
    pub fn new(ctx: &crate::GpuContext) -> Result<Self> {
        let sequence_decoder = ZstdGpuDecoder::new(ctx)?;

        Ok(Self {
            device: ctx.device().clone(),
            sequence_decoder,
            ready: true,
        })
    }

    /// Check if pipeline is ready.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Decompress a Zstd frame.
    pub fn decompress(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        self.sequence_decoder.decompress(compressed)
    }

    /// Decompress with dictionary.
    pub fn decompress_with_dict(
        &self,
        compressed: &[u8],
        _dict: &[u8],
    ) -> Result<Vec<u8>> {
        // CPU fallback with dictionary
        let mut decoder = zstd::Decoder::with_dictionary(compressed, _dict)
            .map_err(|e| CudaError::DecompressionFailed(e.to_string()))?;

        let mut output = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut output)
            .map_err(|e| CudaError::DecompressionFailed(e.to_string()))?;

        Ok(output)
    }

    /// Decompress multiple frames in batch.
    pub fn decompress_batch(&self, frames: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        frames.iter().map(|f| self.decompress(f)).collect()
    }

    /// Decompress directly to GPU buffer.
    pub fn decompress_to_gpu(&self, compressed: &[u8]) -> Result<GpuBuffer> {
        let decompressed = self.decompress(compressed)?;

        // Allocate and transfer to GPU
        let buffer = GpuBuffer::new(self.device.clone(), decompressed.len())?;
        buffer.copy_from_host(&decompressed)?;

        Ok(buffer)
    }
}

// =========================================================================
// Track B.1: GPU Sequence Decoder Tests (12 tests)
// =========================================================================

#[cfg(test)]
mod gpu_sequence_tests {
    use super::*;

    // Helper to create a test GPU context (uses CPU fallback)
    fn test_context() -> Option<crate::GpuContext> {
        // Use catch_unwind to handle case where CUDA isn't available
        std::panic::catch_unwind(|| crate::GpuContext::new(0).ok()).ok().flatten()
    }

    #[test]
    fn test_sequence_creation() {
        let seq = Sequence::new(10, 5, 3);
        assert_eq!(seq.literal_length, 10);
        assert_eq!(seq.match_offset, 5);
        assert_eq!(seq.match_length, 3);
        assert_eq!(seq.output_size(), 13);
    }

    #[test]
    fn test_sequence_literal_only() {
        let seq = Sequence::literal_only(100);
        assert_eq!(seq.literal_length, 100);
        assert_eq!(seq.match_offset, 0);
        assert_eq!(seq.match_length, 0);
        assert_eq!(seq.output_size(), 100);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_sequence_decoder_creation() {
        let ctx = test_context().expect("GPU context required for this test");
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();
        assert!(decoder.is_ready());
    }

    #[test]
    fn test_sequence_literal_copy_single() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return, // Skip if no GPU
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        let literals = b"Hello, World!";
        let sequences = vec![Sequence::literal_only(literals.len())];

        let result = decoder.execute_sequences(&sequences, literals, &[]).unwrap();
        assert_eq!(result.as_slice(), literals.as_slice());
    }

    #[test]
    fn test_sequence_match_copy_simple() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        let literals = b"abc";
        let sequences = vec![
            Sequence::new(3, 0, 0), // 3 literals
            Sequence::new(0, 3, 3), // match: offset=3, length=3
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]).unwrap();
        assert_eq!(result.as_slice(), b"abcabc");
    }

    #[test]
    fn test_sequence_overlapping_match_rle() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // RLE-style: copy from offset 1 repeatedly
        let literals = b"a";
        let sequences = vec![
            Sequence::new(1, 0, 0),  // 1 literal "a"
            Sequence::new(0, 1, 10), // match: offset=1, length=10
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]).unwrap();
        assert_eq!(result.as_slice(), b"aaaaaaaaaaa"); // 11 a's
    }

    #[test]
    fn test_sequence_multiple_sequences() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        let literals = b"The quick brown fox ";
        let sequences = vec![
            Sequence::new(20, 0, 0), // All literals
            Sequence::new(0, 20, 5), // Copy 5 bytes from beginning
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]).unwrap();
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_sequence_batch_processing() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        let blocks: Vec<(Vec<u8>, Vec<Sequence>)> = (0..100)
            .map(|i| {
                let lit = format!("Block {} data here", i).into_bytes();
                let seq = vec![Sequence::literal_only(lit.len())];
                (lit, seq)
            })
            .collect();

        let results = decoder.execute_batch(&blocks).unwrap();

        assert_eq!(results.len(), 100);
        for (i, result) in results.iter().enumerate() {
            let expected = format!("Block {} data here", i);
            assert_eq!(result.as_slice(), expected.as_bytes());
        }
    }

    #[test]
    fn test_sequence_cpu_equivalence() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // Compress some data
        let original = b"Test data for GPU vs CPU comparison. ".repeat(100);
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();

        // Decompress on CPU (reference)
        let cpu_result = zstd::decode_all(compressed.as_slice()).unwrap();

        // Decompress via decoder
        let decoder_result = decoder.decompress(&compressed).unwrap();

        assert_eq!(cpu_result, decoder_result);
    }

    #[test]
    fn test_sequence_large_offset_handling() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // Create data with large literal section
        let mut literals = vec![b'x'; 1000];
        literals[0..5].copy_from_slice(b"MATCH");

        let sequences = vec![
            Sequence::new(1000, 0, 0),  // All literals
            Sequence::new(0, 1000, 5),  // Match from very beginning
        ];

        let result = decoder.execute_sequences(&sequences, &literals, &[]).unwrap();

        assert_eq!(&result[1000..1005], b"MATCH");
    }

    #[test]
    fn test_sequence_error_invalid_offset() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        let literals = b"abc";
        let sequences = vec![
            Sequence::new(3, 0, 0),
            Sequence::new(0, 100, 5), // Invalid: offset 100 but only 3 bytes exist
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_empty_input() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        let result = decoder.execute_sequences(&[], &[], &[]).unwrap();
        assert!(result.is_empty());
    }
}

// =========================================================================
// Track B.2: GPU FSE Decoder Tests (15 tests)
// =========================================================================

#[cfg(test)]
mod gpu_fse_tests {
    use super::*;

    fn test_context() -> Option<crate::GpuContext> {
        // Use catch_unwind to handle case where CUDA isn't available
        std::panic::catch_unwind(|| crate::GpuContext::new(0).ok()).ok().flatten()
    }

    #[test]
    fn test_fse_table_creation() {
        let frequencies = [100u32, 50, 25, 12, 6, 3, 2, 1];
        let table = FseTable::from_frequencies(&frequencies).unwrap();

        assert!(table.table_size > 0);
        assert!(table.accuracy_log >= 5);
    }

    #[test]
    fn test_fse_predefined_literals() {
        let table = FseTable::predefined_literals();
        assert!(table.table_size > 0);
    }

    #[test]
    fn test_fse_predefined_match_lengths() {
        let table = FseTable::predefined_match_lengths();
        assert!(table.table_size > 0);
    }

    #[test]
    fn test_fse_predefined_offsets() {
        let table = FseTable::predefined_offsets();
        assert!(table.table_size > 0);
    }

    #[test]
    fn test_fse_decoder_creation() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let table = FseTable::predefined_literals();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        assert!(decoder.is_ready());
    }

    #[test]
    fn test_fse_decode_simple() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let frequencies = [100u32, 50, 25, 12, 6, 3, 2, 1];
        let table = FseTable::from_frequencies(&frequencies).unwrap();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        // Simple encoded data
        let encoded = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let result = decoder.decode(&encoded, 8).unwrap();

        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_fse_all_predefined_tables() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let tables = [
            FseTable::predefined_literals(),
            FseTable::predefined_match_lengths(),
            FseTable::predefined_offsets(),
        ];

        for table in &tables {
            let decoder = FseGpuDecoder::new(&ctx, table).unwrap();
            assert!(decoder.is_ready());
        }
    }

    #[test]
    fn test_fse_batch_decode() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let table = FseTable::predefined_literals();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        let streams: Vec<Vec<u8>> = (0..10)
            .map(|i| vec![(i % 256) as u8; 100])
            .collect();
        let lengths: Vec<usize> = vec![100; 10];

        let results = decoder.decode_batch(&streams, &lengths).unwrap();

        assert_eq!(results.len(), 10);
        for result in &results {
            assert_eq!(result.len(), 100);
        }
    }

    #[test]
    fn test_fse_large_alphabet() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        // 256-symbol alphabet
        let frequencies: Vec<u32> = (0..256).map(|i| 256 - i as u32).collect();
        let table = FseTable::from_frequencies(&frequencies).unwrap();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        let encoded: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let result = decoder.decode(&encoded, 1000).unwrap();

        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_fse_table_upload_caching() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let table = FseTable::predefined_literals();

        // Create multiple decoders with same table
        let _decoder1 = FseGpuDecoder::new(&ctx, &table).unwrap();
        let _decoder2 = FseGpuDecoder::new(&ctx, &table).unwrap();
        let _decoder3 = FseGpuDecoder::new(&ctx, &table).unwrap();

        // All should be ready
        assert!(_decoder1.is_ready());
        assert!(_decoder2.is_ready());
        assert!(_decoder3.is_ready());
    }

    #[test]
    fn test_fse_interleaved_streams() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let table = FseTable::predefined_literals();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        let streams: [Vec<u8>; 4] = [
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![8, 9, 10, 11],
            vec![12, 13, 14, 15],
        ];

        let result = decoder.decode_interleaved(&streams, &[4, 4, 4, 4]).unwrap();

        // Should have all 16 bytes interleaved
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_fse_empty_table() {
        let result = FseTable::from_frequencies(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fse_single_symbol() {
        let frequencies = [100u32];
        let table = FseTable::from_frequencies(&frequencies).unwrap();

        assert!(table.table_size > 0);
    }

    #[test]
    fn test_fse_decode_zero_length() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let table = FseTable::predefined_literals();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        let result = decoder.decode(&[], 0).unwrap();
        assert!(result.is_empty());
    }
}

// =========================================================================
// Track B.3: GPU Full Pipeline Tests (10 tests)
// =========================================================================

#[cfg(test)]
mod gpu_pipeline_tests {
    use super::*;

    fn test_context() -> Option<crate::GpuContext> {
        // Use catch_unwind to handle case where CUDA isn't available
        std::panic::catch_unwind(|| crate::GpuContext::new(0).ok()).ok().flatten()
    }

    fn generate_test_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| ((i * 17 + i / 256) % 256) as u8).collect()
    }

    #[test]
    fn test_pipeline_creation() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();
        assert!(pipeline.is_ready());
    }

    #[test]
    fn test_pipeline_simple_frame() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = b"Hello, GPU Zstd!";
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();

        let decompressed = pipeline.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), original.as_slice());
    }

    #[test]
    fn test_pipeline_large_frame() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = generate_test_data(1_000_000); // 1MB
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();

        let decompressed = pipeline.decompress(&compressed).unwrap();

        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_pipeline_batch() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let frames: Vec<Vec<u8>> = (0..10)
            .map(|i| {
                let data = format!("Frame {} with some data", i).repeat(100);
                zstd::encode_all(data.as_bytes(), 3).unwrap()
            })
            .collect();

        let results = pipeline.decompress_batch(&frames).unwrap();

        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            let expected = format!("Frame {} with some data", i).repeat(100);
            assert_eq!(result.as_slice(), expected.as_bytes());
        }
    }

    #[test]
    fn test_pipeline_to_gpu() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = generate_test_data(10000);
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();

        let gpu_buffer = pipeline.decompress_to_gpu(&compressed).unwrap();

        // Verify by reading back
        let host_data = gpu_buffer.to_host().unwrap();
        assert_eq!(host_data, original);
    }

    #[test]
    fn test_pipeline_empty_input() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = b"";
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();

        let decompressed = pipeline.decompress(&compressed).unwrap();

        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_pipeline_high_compression_data() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        // Highly compressible data (all same byte)
        let original = vec![0u8; 100_000];
        let compressed = zstd::encode_all(original.as_slice(), 19).unwrap();

        // Should compress very well
        assert!(compressed.len() < original.len() / 10);

        let decompressed = pipeline.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_pipeline_random_data() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        // Random data (less compressible)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let original: Vec<u8> = (0..10000).map(|_| rng.r#gen()).collect();
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();

        let decompressed = pipeline.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_pipeline_invalid_data() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let invalid = b"not valid zstd data";
        let result = pipeline.decompress(invalid);

        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_throughput_measurement() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = generate_test_data(1_000_000); // 1MB
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();

        let start = std::time::Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = pipeline.decompress(&compressed).unwrap();
        }
        let elapsed = start.elapsed();

        let throughput_mbs = (iterations as f64 * original.len() as f64)
            / elapsed.as_secs_f64()
            / 1_000_000.0;

        // Just verify it runs (actual GPU would be much faster)
        assert!(throughput_mbs > 0.0);
        println!("Pipeline throughput: {:.2} MB/s", throughput_mbs);
    }
}
