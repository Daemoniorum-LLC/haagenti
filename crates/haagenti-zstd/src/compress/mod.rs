//! Zstd compression pipeline.
//!
//! This module implements the compression side of Zstandard (RFC 8878).
//!
//! ## Pipeline Overview
//!
//! ```text
//! Input Data
//!     │
//!     ▼
//! ┌─────────────────────────────────────┐
//! │  Compressibility Analysis           │
//! │  - Entropy estimation               │
//! │  - Pattern detection                │
//! │  - Strategy selection               │
//! └─────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────┐
//! │  Match Finding (LZ77)               │
//! │  - Hash chain construction          │
//! │  - Best match selection             │
//! │  - Coverage analysis                │
//! └─────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────┐
//! │  Block Encoding                     │
//! │  - Literals (Huffman/Raw/RLE)       │
//! │  - Sequences (FSE/RLE/Raw)          │
//! │  - Block type selection             │
//! └─────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────┐
//! │  Frame Assembly                     │
//! │  - Magic number                     │
//! │  - Frame header                     │
//! │  - Blocks                           │
//! │  - XXHash64 checksum                │
//! └─────────────────────────────────────┘
//!     │
//!     ▼
//! Compressed Output
//! ```
//!
//! ## Novel Approach: Compressibility Fingerprinting
//!
//! Unlike traditional compressors that blindly attempt compression and
//! fall back on failure, this implementation uses **compressibility
//! fingerprinting** to predict the optimal encoding strategy before
//! attempting compression.
//!
//! ### Benefits
//!
//! 1. **CPU Efficiency**: Skip compression attempts on incompressible data
//! 2. **Better Strategy Selection**: Choose RLE vs FSE vs Raw based on analysis
//! 3. **Reduced Branching**: Predict block type without trial encoding
//!
//! ### Fingerprint Components
//!
//! - **Entropy Estimate**: Shannon entropy from byte frequencies (0.0-8.0)
//! - **Pattern Type**: Uniform, Periodic, TextLike, HighEntropy, Random
//! - **Match Uniformity**: Whether matches have consistent offset/length
//!
//! ## RLE-First Sequence Encoding
//!
//! Instead of jumping straight to complex FSE sequence encoding, we
//! leverage **RLE sequence mode** for uniform match patterns:
//!
//! ```text
//! if all_sequences_have_same(ll_code, ml_code, of_code):
//!     use RLE mode (3 bytes overhead vs FSE table overhead)
//! else:
//!     check reachability and try FSE with predefined tables
//! ```
//!
//! This is simpler, faster, and often just as effective for patterns like:
//! - "abcdabcdabcd..." (uniform 4-byte matches)
//! - Repeated structures in binary data
//!
//! ## Components
//!
//! - `analysis`: Compressibility fingerprinting and strategy selection
//! - `match_finder`: LZ77 match finding using hash chains
//! - `sequences`: RLE-first sequence encoding with FSE fallback
//! - `block`: Block-level encoding (literals + sequences)
//!
//! ## Strategy Selection
//!
//! | Pattern Type | Entropy | Match Coverage | Strategy |
//! |--------------|---------|----------------|----------|
//! | Uniform | 0.0 | N/A | RLE Block |
//! | Periodic (≤4) | Low | High | RLE Sequences |
//! | LowEntropy | Low | High | RLE Sequences |
//! | TextLike | Medium | Medium | Predefined FSE |
//! | HighEntropy | High | Low | Raw Block |
//! | Random | 7.5+ | None | Raw Block |

mod analysis;
mod arena;
pub mod block;
mod match_finder;
mod repeat_offset_finder;
mod sequences;
pub mod speculative;

pub use analysis::{
    // Phase 3: Ultra-fast entropy fingerprinting
    fast_entropy_estimate,
    fast_predict_block_type,
    fast_should_compress,
    CompressibilityFingerprint,
    CompressionStrategy,
    FastBlockType,
    PatternType,
};
pub use arena::{Arena, ArenaVec, DEFAULT_ARENA_SIZE};
pub use block::{encode_block, BlockEncoder};
pub use match_finder::{CacheAligned, LazyMatchFinder, Match, MatchFinder};
pub use repeat_offset_finder::RepeatOffsetMatchFinder;
pub use sequences::{
    analyze_for_rle, encode_sequences_fse, encode_sequences_fse_with_encoded, encode_sequences_rle,
    encode_sequences_with_custom_tables, EncodedSequence, RleSuitability,
};
pub use speculative::{SpeculativeCompressor, SpeculativeStrategy};

use crate::frame::{xxhash64, ZSTD_MAGIC};
use crate::{CustomFseTables, CustomHuffmanTable};
use haagenti_core::{CompressionLevel, Result};

/// Match finder variant based on compression level.
#[derive(Debug)]
enum MatchFinderVariant {
    /// Fast greedy match finder for speed.
    Greedy(MatchFinder),
    /// Lazy match finder for better compression ratio.
    Lazy(LazyMatchFinder),
    /// Repeat offset-aware match finder for best compression.
    RepeatAware(RepeatOffsetMatchFinder),
}

/// Compression context holding state during compression.
pub struct CompressContext {
    /// Compression level (affects match finding depth).
    #[allow(dead_code)] // Used for future FSE encoding depth tuning
    level: CompressionLevel,
    /// Match finder (greedy or lazy based on level).
    match_finder: MatchFinderVariant,
    /// Optional dictionary ID for dictionary compression.
    dictionary_id: Option<u32>,
    /// Arena for per-frame temporary allocations.
    /// Reduces allocation overhead by reusing memory between frames.
    arena: Arena,
    /// Optional custom FSE tables for sequence encoding.
    custom_tables: Option<CustomFseTables>,
    /// Optional custom Huffman table for literal encoding.
    custom_huffman: Option<CustomHuffmanTable>,
}

// Manual Debug impl since Arena uses Cell which has limited Debug output
impl core::fmt::Debug for CompressContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CompressContext")
            .field("level", &self.level)
            .field("match_finder", &self.match_finder)
            .field("dictionary_id", &self.dictionary_id)
            .field("arena_usage", &self.arena.usage())
            .field("arena_capacity", &self.arena.capacity())
            .field("has_custom_tables", &self.custom_tables.is_some())
            .field("has_custom_huffman", &self.custom_huffman.is_some())
            .finish()
    }
}

impl CompressContext {
    /// Create a new compression context.
    pub fn new(level: CompressionLevel) -> Self {
        // Balanced search depths:
        // - Fast mode prioritizes throughput with acceptable ratio
        // - Higher levels trade throughput for compression ratio
        // Note: Our hash chain + 4-byte prefix check is efficient, so we can
        // afford moderate depth without excessive slowdown.
        let search_depth = match level {
            CompressionLevel::None => 1,
            CompressionLevel::Fast => 8,     // Balanced for speed
            CompressionLevel::Default => 24, // Good balance
            CompressionLevel::Best => 48,    // Favor ratio
            CompressionLevel::Ultra => 128,  // Maximum quality
            CompressionLevel::Custom(n) => n.min(255) as usize,
        };

        // Match finder selection:
        // - None: Greedy for maximum speed
        // - Fast/Default: Lazy for good ratio/speed balance
        // - Best/Ultra: RepeatAware for best compression (exploits repeat offsets)
        let match_finder = match level {
            CompressionLevel::None => MatchFinderVariant::Greedy(MatchFinder::new(search_depth)),
            CompressionLevel::Best | CompressionLevel::Ultra => {
                MatchFinderVariant::RepeatAware(RepeatOffsetMatchFinder::new(search_depth))
            }
            _ => MatchFinderVariant::Lazy(LazyMatchFinder::new(search_depth)),
        };

        Self {
            level,
            match_finder,
            dictionary_id: None,
            arena: Arena::with_default_size(),
            custom_tables: None,
            custom_huffman: None,
        }
    }

    /// Create a compression context with custom FSE tables.
    ///
    /// Custom tables allow overriding the predefined FSE tables used for
    /// sequence encoding. This can improve compression ratio when the data
    /// has symbol distributions that differ from the predefined tables.
    pub fn with_custom_tables(level: CompressionLevel, custom_tables: CustomFseTables) -> Self {
        let mut ctx = Self::new(level);
        ctx.custom_tables = Some(custom_tables);
        ctx
    }

    /// Create a compression context with all custom options.
    ///
    /// This is the most flexible constructor, allowing both custom FSE tables
    /// and custom Huffman tables to be specified.
    pub fn with_options(
        level: CompressionLevel,
        custom_tables: Option<CustomFseTables>,
        custom_huffman: Option<CustomHuffmanTable>,
    ) -> Self {
        let mut ctx = Self::new(level);
        ctx.custom_tables = custom_tables;
        ctx.custom_huffman = custom_huffman;
        ctx
    }

    /// Create a new compression context with a custom arena size.
    ///
    /// Larger arena sizes can improve performance for large inputs by reducing
    /// the number of heap allocations during compression.
    pub fn with_arena_size(level: CompressionLevel, arena_size: usize) -> Self {
        let mut ctx = Self::new(level);
        ctx.arena = Arena::new(arena_size);
        ctx
    }

    /// Get the custom FSE tables, if any.
    pub fn custom_tables(&self) -> Option<&CustomFseTables> {
        self.custom_tables.as_ref()
    }

    /// Get the custom Huffman table, if any.
    pub fn custom_huffman(&self) -> Option<&CustomHuffmanTable> {
        self.custom_huffman.as_ref()
    }

    /// Get the arena's peak usage (useful for tuning arena size).
    pub fn arena_peak_usage(&self) -> usize {
        self.arena.peak_usage()
    }

    /// Set dictionary ID for dictionary compression.
    pub fn set_dictionary_id(&mut self, dict_id: u32) {
        self.dictionary_id = Some(dict_id);
    }

    /// Find matches using the appropriate match finder variant.
    ///
    /// For lazy matching, uses adaptive threshold scaling based on input size
    /// to optimize throughput for larger inputs while maintaining good ratio
    /// for smaller inputs.
    fn find_matches(&mut self, input: &[u8]) -> Vec<Match> {
        match &mut self.match_finder {
            MatchFinderVariant::Greedy(mf) => mf.find_matches(input),
            // Use find_matches_auto for adaptive lazy threshold
            MatchFinderVariant::Lazy(mf) => mf.find_matches_auto(input),
            // Repeat offset-aware matching for best compression
            MatchFinderVariant::RepeatAware(mf) => mf.find_matches(input),
        }
    }

    /// Compress data into a Zstd frame.
    ///
    /// By default, does NOT include a checksum (matching reference zstd level 1 behavior).
    /// Use `compress_with_checksum` for checksum-protected frames.
    pub fn compress(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        self.compress_internal(input, false)
    }

    /// Compress data into a Zstd frame with XXH64 checksum.
    ///
    /// The checksum is the lower 32 bits of XXH64 of the original uncompressed content.
    #[allow(dead_code)]
    pub fn compress_with_checksum(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        self.compress_internal(input, true)
    }

    /// Compress using speculative multi-path compression.
    ///
    /// This runs multiple compression strategies in parallel (when the `parallel`
    /// feature is enabled) and returns the smallest result. This provides optimal
    /// compression without needing to know the data characteristics in advance.
    ///
    /// **When to use:**
    /// - When compression ratio matters more than latency
    /// - For mixed content where the optimal strategy varies
    /// - When CPU cores are available for parallel execution
    ///
    /// **Performance characteristics:**
    /// - Uses all available CPU cores via work-stealing
    /// - Minimal overhead for small inputs (< 4KB uses single-threaded)
    /// - Near-linear scaling for large inputs
    #[allow(dead_code)]
    pub fn compress_speculative(&self, input: &[u8]) -> Result<Vec<u8>> {
        let compressor = speculative::SpeculativeCompressor::new();
        compressor.compress(input)
    }

    /// Compress using fast speculative compression.
    ///
    /// Uses fewer strategies for lower latency while still trying multiple approaches.
    #[allow(dead_code)]
    pub fn compress_speculative_fast(&self, input: &[u8]) -> Result<Vec<u8>> {
        let compressor = speculative::SpeculativeCompressor::fast();
        compressor.compress(input)
    }

    /// Compress using best speculative compression.
    ///
    /// Uses aggressive strategies for maximum compression ratio.
    #[allow(dead_code)]
    pub fn compress_speculative_best(&self, input: &[u8]) -> Result<Vec<u8>> {
        let compressor = speculative::SpeculativeCompressor::best();
        compressor.compress(input)
    }

    /// Internal compression with optional checksum.
    fn compress_internal(&mut self, input: &[u8], include_checksum: bool) -> Result<Vec<u8>> {
        // Reset arena for this frame (O(1) operation)
        // This allows reuse of temporary allocations from previous frames
        self.arena.reset();

        let mut output = Vec::with_capacity(input.len() + 32);

        // Write magic number
        output.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());

        // Write frame header
        self.write_frame_header(input.len(), include_checksum, &mut output)?;

        // Encode blocks
        self.encode_blocks(input, &mut output)?;

        // Write checksum if requested
        if include_checksum {
            let checksum = (xxhash64(input, 0) & 0xFFFFFFFF) as u32;
            output.extend_from_slice(&checksum.to_le_bytes());
        }

        Ok(output)
    }

    /// Get access to the arena for temporary allocations.
    ///
    /// This can be used by callers who want to use arena-allocated
    /// buffers for input data preparation.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Write frame header.
    ///
    /// Matches reference zstd's level 1 behavior:
    /// - Uses window descriptor mode (not single_segment) for better compatibility
    /// - No FCS for small frames
    fn write_frame_header(
        &self,
        content_size: usize,
        include_checksum: bool,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        // Frame descriptor layout:
        // FHD: FCS_Field_Size[7:6], Single_Segment[5], Unused[4], Reserved[3], Checksum[2], Dict_ID[1:0]
        let checksum_flag = if include_checksum { 0x04 } else { 0x00 };

        // Match reference zstd behavior: use window descriptor mode (Single_Segment=0)
        // This provides better cross-library compatibility
        //
        // Window_Size = 2^(10 + Exponent) * (1 + Mantissa/8)
        // For simplicity, use Mantissa=0, so Window_Size = 2^(10 + Exponent)
        // Window_Descriptor = Exponent << 3 (with Mantissa=0)

        // Calculate window exponent
        // Use minimum 19 (512KB) to match reference zstd behavior at level 1
        // This provides better cross-decoder compatibility
        let window_log = if content_size == 0 {
            19u8 // 512KB default
        } else {
            let log = (content_size as f64).log2().ceil() as u8;
            log.clamp(19, 30) // minimum 512KB window
        };

        // Use FCS for larger content (>128KB), skip for typical block sizes
        // Reference zstd doesn't use FCS for 64KB inputs, uses window descriptor only
        let (fcs_size, descriptor) = if content_size > 131071 {
            // 4-byte FCS (FCS_Field_Size=10, Single_Segment=0)
            (4, 0x80u8 | checksum_flag)
        } else {
            // No FCS (FCS_Field_Size=00, Single_Segment=0)
            // Decoder will use window size as max content size
            (0, checksum_flag)
        };

        output.push(descriptor);

        // Window descriptor (always present when Single_Segment=0)
        let window_descriptor = (window_log - 10) << 3;
        output.push(window_descriptor);

        // Write FCS if present
        if fcs_size == 4 {
            output.extend_from_slice(&(content_size as u32).to_le_bytes());
        }

        Ok(())
    }

    /// Encode input data into blocks.
    fn encode_blocks(&mut self, input: &[u8], output: &mut Vec<u8>) -> Result<()> {
        const MAX_BLOCK_SIZE: usize = 128 * 1024 - 1; // 128KB - 1

        if input.is_empty() {
            // Empty block
            let header = 1u32; // last=1, type=Raw, size=0
            output.push((header & 0xFF) as u8);
            output.push(((header >> 8) & 0xFF) as u8);
            output.push(((header >> 16) & 0xFF) as u8);
            return Ok(());
        }

        let mut pos = 0;
        while pos < input.len() {
            let remaining = input.len() - pos;
            let block_size = remaining.min(MAX_BLOCK_SIZE);
            let is_last = pos + block_size >= input.len();

            let block_data = &input[pos..pos + block_size];
            self.encode_single_block(block_data, is_last, output)?;

            pos += block_size;
        }

        Ok(())
    }

    /// Encode a single block.
    fn encode_single_block(
        &mut self,
        input: &[u8],
        is_last: bool,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        // Try to find the best encoding strategy
        let (block_type, encoded) = self.select_block_encoding(input)?;

        // Write block header
        let mut header = if is_last { 1u32 } else { 0u32 };
        header |= (block_type as u32) << 1;

        let size = match block_type {
            0 => input.len(),   // Raw: decompressed size
            1 => input.len(),   // RLE: decompressed size
            2 => encoded.len(), // Compressed: compressed size
            _ => unreachable!(),
        };

        header |= (size as u32) << 3;

        output.push((header & 0xFF) as u8);
        output.push(((header >> 8) & 0xFF) as u8);
        output.push(((header >> 16) & 0xFF) as u8);

        // Write block data
        output.extend_from_slice(&encoded);

        Ok(())
    }

    /// Select the best block encoding strategy using compressibility analysis.
    ///
    /// Uses a tiered approach for optimal throughput:
    /// - Small inputs (<4KB): Full fingerprint analysis for best decisions
    /// - Large inputs (>=4KB): Fast sampling to detect special cases, otherwise
    ///   proceed directly to match finding
    fn select_block_encoding(&mut self, input: &[u8]) -> Result<(u8, Vec<u8>)> {
        // For large inputs, use fast sampling to avoid O(n) histogram overhead
        if input.len() >= 4096 {
            return self.select_block_encoding_fast(input);
        }

        // Small inputs: use full fingerprint analysis
        let fingerprint = analysis::CompressibilityFingerprint::analyze(input);

        // Fast-path for uniform data (RLE block)
        if fingerprint.strategy == CompressionStrategy::RleBlock {
            return Ok((1, vec![input[0]]));
        }

        // Note: We don't skip match finding for "Random" patterns because
        // byte-level entropy can be misleading. Periodic patterns like 0,1,2,...255
        // have high entropy (8.0) but are highly compressible via offset matches.
        // Always try match finding and only fall back to raw if it doesn't help.

        // Find matches
        let matches = self.find_matches(input);

        // Refine strategy with match data
        let mut fingerprint = fingerprint;
        fingerprint.refine_with_matches(input, &matches);

        // Choose encoding based on refined strategy
        match fingerprint.strategy {
            CompressionStrategy::RleBlock => Ok((1, vec![input[0]])),
            CompressionStrategy::RawBlock => {
                let compressed = self.encode_literals_only(input)?;
                if compressed.len() < input.len() {
                    Ok((2, compressed))
                } else {
                    Ok((0, input.to_vec()))
                }
            }
            CompressionStrategy::RleSequences | CompressionStrategy::PredefinedFse => {
                let compressed = self.encode_with_rle_sequences(input, &matches)?;
                if compressed.len() < input.len() {
                    Ok((2, compressed))
                } else {
                    Ok((0, input.to_vec()))
                }
            }
        }
    }

    /// Fast block encoding selection for large inputs.
    ///
    /// Uses sampling-based analysis (~100 cycles) instead of full histogram (~10K cycles).
    /// This dramatically improves throughput for 64KB+ inputs.
    fn select_block_encoding_fast(&mut self, input: &[u8]) -> Result<(u8, Vec<u8>)> {
        // Fast check for RLE (uniform data)
        let first = input[0];
        let is_uniform = input.len() >= 64 && {
            // Check first 64 bytes
            let first_uniform = input[..64].iter().all(|&b| b == first);
            if first_uniform {
                // Verify with sampling for rest
                let step = input.len() / 32;
                (1..32).all(|i| input.get(i * step).copied() == Some(first))
            } else {
                false
            }
        };

        if is_uniform {
            return Ok((1, vec![first]));
        }

        // Note: We don't skip match finding based on entropy because byte-level
        // entropy can be misleading. Periodic patterns like 0,1,2,...255 have
        // high entropy (8.0) but are highly compressible via offset matches.
        // Always try match finding and only fall back to raw if it doesn't help.

        // Proceed with match finding - most data will take this path
        let matches = self.find_matches(input);

        // Encode with sequences
        let compressed = self.encode_with_rle_sequences(input, &matches)?;
        if compressed.len() < input.len() {
            Ok((2, compressed))
        } else {
            Ok((0, input.to_vec()))
        }
    }

    /// Encode a compressed block using sequence encoding.
    ///
    /// Uses RLE mode for uniform sequences (same codes for all LL, OF, ML).
    /// Falls back to FSE encoding for non-uniform sequences.
    /// When custom FSE tables are configured, uses them for sequence encoding.
    /// When custom Huffman tables are configured, uses them for literal encoding.
    fn encode_with_rle_sequences(&mut self, input: &[u8], matches: &[Match]) -> Result<Vec<u8>> {
        // Pre-allocate output: compressed size is usually smaller than input
        let mut output = Vec::with_capacity(input.len());

        // Convert matches to sequences
        let (literals, seqs) = block::matches_to_sequences(input, matches);

        if seqs.is_empty() {
            // No sequences - use literals-only mode
            self.encode_literals_with_options(input, &mut output)?;
            block::encode_sequences(&[], &mut output)?;
        } else {
            // Analyze sequences (also pre-encodes into codes+extra bits)
            let suitability = sequences::analyze_for_rle(&seqs);

            // Encode literals using custom Huffman if available
            self.encode_literals_with_options(&literals, &mut output)?;

            // Use custom tables if available, otherwise use predefined
            if let Some(custom_tables) = &self.custom_tables {
                sequences::encode_sequences_with_custom_tables(
                    &suitability.encoded,
                    custom_tables,
                    &mut output,
                )?;
            } else {
                // Always use FSE/Predefined mode for cross-decoder compatibility.
                // RLE mode has subtle implementation differences that cause issues.
                sequences::encode_sequences_fse_with_encoded(&suitability.encoded, &mut output)?;
            }
        }

        Ok(output)
    }

    /// Encode literals using custom Huffman table if available, otherwise use default logic.
    fn encode_literals_with_options(&self, literals: &[u8], output: &mut Vec<u8>) -> Result<()> {
        if let Some(custom_huffman) = &self.custom_huffman {
            block::encode_literals_with_encoder(literals, custom_huffman.encoder(), output)
        } else {
            block::encode_literals(literals, output)
        }
    }

    /// Encode a compressed block with only literals (no sequences).
    fn encode_literals_only(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(input.len());

        // Encode all input as literals
        self.encode_literals_with_options(input, &mut output)?;

        // Empty sequences section
        block::encode_sequences(&[], &mut output)?;

        Ok(output)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_context_creation() {
        let ctx = CompressContext::new(CompressionLevel::Default);
        assert_eq!(ctx.level, CompressionLevel::Default);
    }

    #[test]
    fn test_compress_empty() {
        let mut ctx = CompressContext::new(CompressionLevel::Fast);
        let result = ctx.compress(&[]).unwrap();

        // Should have magic (4) + header (2: descriptor + 1-byte FCS) + empty block (3)
        // Note: compress() does NOT include checksum by default
        assert!(
            result.len() >= 4 + 2 + 3,
            "expected at least 9 bytes, got {}",
            result.len()
        );

        // Verify magic
        assert_eq!(&result[0..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    }

    #[test]
    fn test_compress_small() {
        let mut ctx = CompressContext::new(CompressionLevel::Fast);
        let input = b"Hello, World!";
        let result = ctx.compress(input).unwrap();

        // Should be valid frame
        assert_eq!(&result[0..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    }

    #[test]
    fn test_compress_rle_detection() {
        let mut ctx = CompressContext::new(CompressionLevel::Fast);
        let input = vec![b'A'; 100];
        let result = ctx.compress(&input).unwrap();

        // RLE should be much smaller than raw
        assert!(result.len() < input.len());
    }

    #[test]
    fn test_compression_levels() {
        for level in [
            CompressionLevel::Fast,
            CompressionLevel::Default,
            CompressionLevel::Best,
        ] {
            let mut ctx = CompressContext::new(level);
            let input = b"Test compression with different levels";
            let result = ctx.compress(input);
            assert!(result.is_ok());
        }
    }

    // =========================================================================
    // Track A.4: Compression Ratio Optimization Tests
    // =========================================================================

    #[test]
    fn test_text_compression_ratio_baseline() {
        // Test compression on text-like data
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(1000);

        let mut ctx = CompressContext::new(CompressionLevel::Default);
        let compressed = ctx.compress(&data).unwrap();

        let ratio = data.len() as f64 / compressed.len() as f64;

        // Text should achieve at least 2.5x compression (baseline)
        assert!(
            ratio >= 2.5,
            "Text compression ratio {:.2}x below baseline 2.5x",
            ratio
        );
    }

    #[test]
    fn test_repetitive_data_compression_ratio() {
        // Repetitive data should compress very well
        let data = b"abcdabcdabcdabcd".repeat(5000);

        let mut ctx = CompressContext::new(CompressionLevel::Default);
        let compressed = ctx.compress(&data).unwrap();

        let ratio = data.len() as f64 / compressed.len() as f64;

        // Repetitive data should achieve at least 50x compression
        assert!(
            ratio >= 50.0,
            "Repetitive data ratio {:.2}x below expected 50x",
            ratio
        );
    }

    #[test]
    fn test_compression_levels_ratio_ordering() {
        // Better compression levels should produce smaller or equal output
        let data = b"This is test data that will be compressed at different levels. ".repeat(500);

        let mut fast_ctx = CompressContext::new(CompressionLevel::Fast);
        let mut default_ctx = CompressContext::new(CompressionLevel::Default);
        let mut best_ctx = CompressContext::new(CompressionLevel::Best);

        let fast = fast_ctx.compress(&data).unwrap();
        let default = default_ctx.compress(&data).unwrap();
        let best = best_ctx.compress(&data).unwrap();

        // Best should be <= Default <= Fast (in terms of output size)
        assert!(
            best.len() <= default.len(),
            "Best ({}) should be <= Default ({})",
            best.len(),
            default.len()
        );
        assert!(
            default.len() <= fast.len() + 50, // Allow small tolerance for heuristics
            "Default ({}) should be close to or better than Fast ({})",
            default.len(),
            fast.len()
        );
    }

    #[test]
    fn test_adaptive_search_depth_performance() {
        // Larger data shouldn't take disproportionately longer
        let small_data = vec![b'a'; 1000];
        let large_data = vec![b'a'; 100_000];

        // Use fast level for timing test
        let mut ctx_small = CompressContext::new(CompressionLevel::Fast);
        let mut ctx_large = CompressContext::new(CompressionLevel::Fast);

        let start = std::time::Instant::now();
        let _ = ctx_small.compress(&small_data).unwrap();
        let small_time = start.elapsed();

        let start = std::time::Instant::now();
        let _ = ctx_large.compress(&large_data).unwrap();
        let large_time = start.elapsed();

        // Large should take at most 200x longer (100x more data + overhead)
        let time_ratio = large_time.as_nanos() as f64 / small_time.as_nanos().max(1) as f64;
        assert!(
            time_ratio < 200.0,
            "Time ratio {:.1}x exceeds expected linear scaling",
            time_ratio
        );
    }

    #[test]
    fn test_lazy_match_finder_produces_matches() {
        // Lazy match finder should find good matches
        let data = b"abcdefabcdefabcdef".repeat(100);

        let mut lazy_mf = LazyMatchFinder::new(24);
        let matches = lazy_mf.find_matches(&data);

        // Should find matches in repetitive data
        assert!(
            !matches.is_empty(),
            "Lazy match finder should find matches in repetitive data"
        );

        // Matches should have valid offsets and lengths
        for m in &matches {
            assert!(m.offset > 0, "Match offset should be positive");
            assert!(m.length >= 3, "Match length should be at least 3");
        }
    }

    #[test]
    fn test_long_distance_pattern_detection() {
        // Create data with long-distance repeats
        let pattern = b"This is a distinctive pattern.";
        let mut data = Vec::new();
        data.extend_from_slice(pattern);
        data.extend_from_slice(&vec![b'x'; 10_000]); // 10KB gap
        data.extend_from_slice(pattern); // Repeat

        let mut ctx = CompressContext::new(CompressionLevel::Default);
        let compressed = ctx.compress(&data).unwrap();

        // Should compress well due to finding the long-distance match
        // Compressed size should be noticeably smaller than input
        assert!(
            compressed.len() < data.len() - 20,
            "Should find long-distance pattern match"
        );
    }

    #[test]
    fn test_entropy_based_strategy_selection() {
        // Low entropy data should trigger efficient encoding
        let low_entropy = vec![0u8; 100_000];

        let mut ctx = CompressContext::new(CompressionLevel::Fast);
        let compressed = ctx.compress(&low_entropy).unwrap();

        // Low entropy should achieve extreme compression (RLE)
        let ratio = low_entropy.len() as f64 / compressed.len() as f64;
        assert!(
            ratio > 1000.0,
            "Low entropy ratio {:.0}x should be >1000x",
            ratio
        );
    }

    #[test]
    fn test_mixed_content_compression() {
        // Mixed content: some compressible, some not
        let mut data = Vec::new();
        data.extend_from_slice(b"Compressible repeated text. ".repeat(50).as_slice());
        data.extend_from_slice(&(0u8..=255u8).cycle().take(1000).collect::<Vec<u8>>());
        data.extend_from_slice(b"More compressible text here. ".repeat(50).as_slice());

        let mut ctx = CompressContext::new(CompressionLevel::Default);
        let compressed = ctx.compress(&data).unwrap();

        // Should still achieve some compression
        assert!(
            compressed.len() < data.len(),
            "Mixed content should still compress"
        );
    }

    #[test]
    fn test_speculative_compression_available() {
        // Verify speculative compression is available
        let data = b"Test data for speculative compression. ".repeat(100);

        let ctx = CompressContext::new(CompressionLevel::Default);
        let result = ctx.compress_speculative(&data);

        assert!(result.is_ok(), "Speculative compression should work");
        assert!(
            result.unwrap().len() < data.len(),
            "Should produce compression"
        );
    }

    #[test]
    fn test_compression_with_checksum() {
        let data = b"Data to compress with checksum verification.".repeat(100);

        let mut ctx = CompressContext::new(CompressionLevel::Default);
        let with_checksum = ctx.compress_with_checksum(&data).unwrap();
        let without_checksum = ctx.compress(&data).unwrap();

        // With checksum should be 4 bytes larger (XXH64 lower 32 bits)
        assert_eq!(
            with_checksum.len(),
            without_checksum.len() + 4,
            "Checksum adds 4 bytes"
        );
    }

    #[test]
    fn test_arena_reuse_efficiency() {
        let mut ctx = CompressContext::new(CompressionLevel::Default);

        // Compress multiple times with different data to force arena usage
        for i in 0..5 {
            let data = format!("Test arena reuse iteration {}. ", i).repeat(1000);
            let result = ctx.compress(data.as_bytes()).unwrap();
            // Verify compression succeeded
            assert!(!result.is_empty());
        }

        // Arena capacity should be available for inspection
        let capacity = ctx.arena().capacity();
        assert!(capacity > 0, "Arena should have capacity");
    }

    #[test]
    fn test_block_size_handling() {
        // Test compression with data larger than max block size (128KB - 1)
        let large_data = vec![b'A'; 150_000]; // >128KB

        let mut ctx = CompressContext::new(CompressionLevel::Fast);
        let compressed = ctx.compress(&large_data).unwrap();

        // Should produce valid compressed output
        assert!(!compressed.is_empty());
        // Should be smaller than input for RLE-able data
        assert!(
            compressed.len() < large_data.len(),
            "Large RLE data should compress well"
        );
    }

    #[test]
    fn test_match_finder_variant_selection() {
        // Different levels should use different match finder variants
        let data = b"test pattern for matching variants".repeat(100);

        // None uses Greedy
        let mut none_ctx = CompressContext::new(CompressionLevel::None);
        let none_result = none_ctx.compress(&data);
        assert!(none_result.is_ok());

        // Best uses RepeatAware
        let mut best_ctx = CompressContext::new(CompressionLevel::Best);
        let best_result = best_ctx.compress(&data);
        assert!(best_result.is_ok());

        // Best should typically produce smaller output for repetitive data
        assert!(
            best_result.unwrap().len() <= none_result.unwrap().len(),
            "Best should be at least as good as None"
        );
    }

    #[test]
    fn test_repeat_offset_match_finder() {
        // RepeatAware should find repeat offset patterns
        let data = b"abcdefghijklmnopabcdefghijklmnopabcdefghijklmnop".repeat(50);

        let mut mf = RepeatOffsetMatchFinder::new(48);
        let matches = mf.find_matches(&data);

        // Should find matches using repeat offsets
        assert!(
            !matches.is_empty(),
            "RepeatAware should find matches in repetitive data"
        );
    }

    #[test]
    fn test_compression_fingerprint_accuracy() {
        // Test fingerprint analysis accuracy
        let uniform_data = vec![42u8; 1000];
        let fp_uniform = analysis::CompressibilityFingerprint::analyze(&uniform_data);
        assert_eq!(
            fp_uniform.pattern,
            PatternType::Uniform,
            "Should detect uniform pattern"
        );

        let text_data = b"The quick brown fox. ".repeat(100);
        let fp_text = analysis::CompressibilityFingerprint::analyze(&text_data);
        assert!(
            matches!(
                fp_text.pattern,
                PatternType::TextLike | PatternType::Periodic { .. } | PatternType::LowEntropy
            ),
            "Should detect text-like or periodic pattern, got {:?}",
            fp_text.pattern
        );
    }

    #[test]
    fn test_fast_entropy_estimate() {
        // Test fast entropy estimation
        let low_entropy = vec![0u8; 10000];
        let est_low = fast_entropy_estimate(&low_entropy);
        assert!(est_low < 1.0, "Low entropy data should have low estimate");

        let high_entropy: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let est_high = fast_entropy_estimate(&high_entropy);
        assert!(
            est_high > 5.0,
            "High entropy data should have high estimate: {}",
            est_high
        );
    }

    #[test]
    fn test_fast_predict_block_type() {
        // Test fast block type prediction
        let zeros = vec![0u8; 1000];
        assert_eq!(
            fast_predict_block_type(&zeros),
            FastBlockType::Rle,
            "Uniform data should predict RLE"
        );

        let text = b"Hello world! ".repeat(100);
        let predicted = fast_predict_block_type(&text);
        assert!(
            predicted == FastBlockType::Compress || predicted == FastBlockType::Raw,
            "Text should predict Compress or Raw"
        );
    }
}
