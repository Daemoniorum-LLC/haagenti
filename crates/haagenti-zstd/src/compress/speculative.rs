//! Speculative Multi-Path Compression.
//!
//! This module implements a novel compression approach that runs multiple
//! strategies in parallel and picks the strategy producing the smallest output.
//!
//! ## Why Speculative Compression?
//!
//! Traditional compressors use a single fixed strategy, which may not be optimal
//! for all data types. For example:
//!
//! - Greedy matching is fast but may miss better matches
//! - Lazy matching finds better matches but is slower
//! - High-entropy data compresses poorly with LZ77 but may benefit from Huffman
//! - Repetitive data benefits from aggressive match finding
//!
//! By running multiple strategies in parallel and picking the best result, we
//! achieve optimal compression without knowing the data characteristics in advance.
//!
//! ## Performance Characteristics
//!
//! - Uses all available CPU cores via rayon work-stealing
//! - Minimal overhead for small inputs (single-threaded fast path)
//! - Near-linear scaling for large inputs (> 16KB)
//! - Automatically adapts to data: O(1) additional latency, O(cores) throughput gain
//!
//! ## Strategies Tested
//!
//! 1. **Greedy Fast** (depth=4): Fastest, baseline for throughput
//! 2. **Greedy Deep** (depth=16): Better matches, moderate speed
//! 3. **Lazy Default** (depth=16): Look-ahead matching, best ratio
//! 4. **Lazy Best** (depth=64): Aggressive look-ahead, maximum compression
//! 5. **Literals Only**: Skip match finding, Huffman-only compression
//!
//! ## Usage
//!
//! ```rust,ignore
//! use haagenti_zstd::compress::speculative::SpeculativeCompressor;
//!
//! let compressor = SpeculativeCompressor::new();
//! let compressed = compressor.compress(&data)?;
//! ```

use super::block::{encode_literals, encode_sequences, matches_to_sequences};
use super::sequences::encode_sequences_fse;
use super::{
    analysis::{CompressibilityFingerprint, CompressionStrategy, PatternType},
    LazyMatchFinder, Match, MatchFinder,
};
use crate::frame::{xxhash64, ZSTD_MAGIC};
use haagenti_core::Result;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Compression strategy for speculative execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeStrategy {
    /// Greedy matching with shallow search (depth=4).
    GreedyFast,
    /// Greedy matching with deeper search (depth=16).
    GreedyDeep,
    /// Lazy matching with default depth (depth=16).
    LazyDefault,
    /// Lazy matching with aggressive search (depth=64).
    LazyBest,
    /// Literals-only encoding (skip match finding).
    LiteralsOnly,
    /// RLE block (single byte).
    Rle,
}

impl SpeculativeStrategy {
    /// All strategies to try for general data.
    pub fn all() -> &'static [SpeculativeStrategy] {
        &[
            SpeculativeStrategy::GreedyFast,
            SpeculativeStrategy::GreedyDeep,
            SpeculativeStrategy::LazyDefault,
            SpeculativeStrategy::LazyBest,
            SpeculativeStrategy::LiteralsOnly,
        ]
    }

    /// Fast strategies for latency-sensitive workloads.
    pub fn fast() -> &'static [SpeculativeStrategy] {
        &[
            SpeculativeStrategy::GreedyFast,
            SpeculativeStrategy::GreedyDeep,
            SpeculativeStrategy::LiteralsOnly,
        ]
    }

    /// Best compression strategies (slower but smaller).
    pub fn best() -> &'static [SpeculativeStrategy] {
        &[
            SpeculativeStrategy::LazyDefault,
            SpeculativeStrategy::LazyBest,
        ]
    }
}

/// Result of a single compression attempt.
#[allow(dead_code)]
struct CompressionResult {
    /// The compressed data.
    data: Vec<u8>,
    /// The strategy used.
    strategy: SpeculativeStrategy,
}

/// Speculative multi-path compressor.
///
/// Runs multiple compression strategies in parallel and picks the best result.
pub struct SpeculativeCompressor {
    /// Strategies to try.
    strategies: Vec<SpeculativeStrategy>,
    /// Minimum input size for parallel execution.
    /// Smaller inputs use single-threaded fast path.
    parallel_threshold: usize,
    /// Include checksum in output.
    include_checksum: bool,
}

impl Default for SpeculativeCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeculativeCompressor {
    /// Create a new speculative compressor with default settings.
    pub fn new() -> Self {
        Self {
            strategies: SpeculativeStrategy::all().to_vec(),
            parallel_threshold: 4096, // 4KB minimum for parallel
            include_checksum: false,
        }
    }

    /// Create a compressor optimized for throughput (fewer strategies).
    pub fn fast() -> Self {
        Self {
            strategies: SpeculativeStrategy::fast().to_vec(),
            parallel_threshold: 4096,
            include_checksum: false,
        }
    }

    /// Create a compressor optimized for compression ratio.
    pub fn best() -> Self {
        Self {
            strategies: SpeculativeStrategy::best().to_vec(),
            parallel_threshold: 2048,
            include_checksum: false,
        }
    }

    /// Set the strategies to try.
    pub fn with_strategies(mut self, strategies: &[SpeculativeStrategy]) -> Self {
        self.strategies = strategies.to_vec();
        self
    }

    /// Set the parallel execution threshold.
    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Enable checksum in output.
    pub fn with_checksum(mut self, enabled: bool) -> Self {
        self.include_checksum = enabled;
        self
    }

    /// Compress data using speculative multi-path compression.
    ///
    /// Tries multiple strategies and returns the smallest result.
    pub fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        // Fast path: use analysis to skip unnecessary work
        let fingerprint = CompressibilityFingerprint::analyze(input);

        // If data is uniform, use RLE directly
        if fingerprint.strategy == CompressionStrategy::RleBlock {
            return self.compress_rle(input);
        }

        // If data is random, use raw block directly (skip all strategies)
        if fingerprint.pattern == PatternType::Random {
            return self.compress_raw(input);
        }

        // Determine if we should use parallel execution
        #[cfg(feature = "parallel")]
        {
            if input.len() >= self.parallel_threshold && self.strategies.len() > 1 {
                return self.compress_parallel(input, &fingerprint);
            }
        }

        // Sequential execution for small inputs or when parallel feature is disabled
        self.compress_sequential(input, &fingerprint)
    }

    /// Compress a single block with a specific strategy.
    fn compress_with_strategy(
        &self,
        input: &[u8],
        strategy: SpeculativeStrategy,
    ) -> Result<Vec<u8>> {
        let (block_type, encoded) = match strategy {
            SpeculativeStrategy::Rle => {
                if input.iter().all(|&b| b == input[0]) {
                    (1u8, vec![input[0]])
                } else {
                    // Fall back to raw if not actually RLE
                    (0u8, input.to_vec())
                }
            }
            SpeculativeStrategy::LiteralsOnly => {
                // Encode as literals only (Huffman or raw)
                let mut output = Vec::new();
                encode_literals(input, &mut output)?;
                encode_sequences(&[], &mut output)?;

                if output.len() < input.len() {
                    (2u8, output)
                } else {
                    (0u8, input.to_vec())
                }
            }
            SpeculativeStrategy::GreedyFast => {
                let mut finder = MatchFinder::new(4);
                let matches = finder.find_matches(input);
                self.encode_with_matches(input, &matches)?
            }
            SpeculativeStrategy::GreedyDeep => {
                let mut finder = MatchFinder::new(16);
                let matches = finder.find_matches(input);
                self.encode_with_matches(input, &matches)?
            }
            SpeculativeStrategy::LazyDefault => {
                let mut finder = LazyMatchFinder::new(16);
                let matches = finder.find_matches(input);
                self.encode_with_matches(input, &matches)?
            }
            SpeculativeStrategy::LazyBest => {
                let mut finder = LazyMatchFinder::new(64);
                let matches = finder.find_matches(input);
                self.encode_with_matches(input, &matches)?
            }
        };

        // Build frame around the block
        self.build_frame(input, block_type, &encoded)
    }

    /// Encode input with matches.
    fn encode_with_matches(&self, input: &[u8], matches: &[Match]) -> Result<(u8, Vec<u8>)> {
        if matches.is_empty() {
            // No matches found - use literals only
            let mut output = Vec::new();
            encode_literals(input, &mut output)?;
            encode_sequences(&[], &mut output)?;

            if output.len() < input.len() {
                Ok((2, output))
            } else {
                Ok((0, input.to_vec()))
            }
        } else {
            // Encode with matches
            let (literals, sequences) = matches_to_sequences(input, matches);
            let mut output = Vec::new();
            encode_literals(&literals, &mut output)?;
            encode_sequences_fse(&sequences, &mut output)?;

            if output.len() < input.len() {
                Ok((2, output))
            } else {
                Ok((0, input.to_vec()))
            }
        }
    }

    /// Build a complete Zstd frame.
    fn build_frame(&self, input: &[u8], block_type: u8, encoded: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(encoded.len() + 32);

        // Magic number
        output.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());

        // Frame header
        self.write_frame_header(input.len(), &mut output)?;

        // Block header (last block = true)
        let size = match block_type {
            0 => input.len(),   // Raw
            1 => input.len(),   // RLE
            2 => encoded.len(), // Compressed
            _ => unreachable!(),
        };
        let header = 1u32 | ((block_type as u32) << 1) | ((size as u32) << 3);
        output.push((header & 0xFF) as u8);
        output.push(((header >> 8) & 0xFF) as u8);
        output.push(((header >> 16) & 0xFF) as u8);

        // Block data
        output.extend_from_slice(encoded);

        // Optional checksum
        if self.include_checksum {
            let checksum = (xxhash64(input, 0) & 0xFFFFFFFF) as u32;
            output.extend_from_slice(&checksum.to_le_bytes());
        }

        Ok(output)
    }

    /// Write frame header.
    fn write_frame_header(&self, content_size: usize, output: &mut Vec<u8>) -> Result<()> {
        let checksum_flag = if self.include_checksum { 0x04 } else { 0x00 };

        let window_log = if content_size == 0 {
            10u8
        } else {
            let log = (content_size as f64).log2().ceil() as u8;
            log.clamp(10, 30)
        };

        let (fcs_size, descriptor) = if content_size > 65535 {
            (4, 0x80u8 | checksum_flag)
        } else {
            (0, checksum_flag)
        };

        output.push(descriptor);

        let window_descriptor = (window_log - 10) << 3;
        output.push(window_descriptor);

        if fcs_size == 4 {
            output.extend_from_slice(&(content_size as u32).to_le_bytes());
        }

        Ok(())
    }

    /// Compress as RLE block.
    fn compress_rle(&self, input: &[u8]) -> Result<Vec<u8>> {
        self.build_frame(input, 1, &[input[0]])
    }

    /// Compress as raw block.
    fn compress_raw(&self, input: &[u8]) -> Result<Vec<u8>> {
        self.build_frame(input, 0, input)
    }

    /// Sequential compression (single-threaded).
    fn compress_sequential(
        &self,
        input: &[u8],
        _fingerprint: &CompressibilityFingerprint,
    ) -> Result<Vec<u8>> {
        let mut best: Option<Vec<u8>> = None;

        for &strategy in &self.strategies {
            let result = self.compress_with_strategy(input, strategy)?;

            if best
                .as_ref()
                .map(|b| result.len() < b.len())
                .unwrap_or(true)
            {
                best = Some(result);
            }
        }

        Ok(best.unwrap_or_else(|| {
            // Fallback to raw
            self.compress_raw(input).unwrap_or_else(|_| input.to_vec())
        }))
    }

    /// Parallel compression using rayon.
    #[cfg(feature = "parallel")]
    fn compress_parallel(
        &self,
        input: &[u8],
        _fingerprint: &CompressibilityFingerprint,
    ) -> Result<Vec<u8>> {
        // Run all strategies in parallel
        let results: Vec<_> = self
            .strategies
            .par_iter()
            .filter_map(|&strategy| {
                self.compress_with_strategy(input, strategy)
                    .ok()
                    .map(|data| CompressionResult { data, strategy })
            })
            .collect();

        // Find the smallest result
        let best = results
            .into_iter()
            .min_by_key(|r| r.data.len())
            .map(|r| r.data);

        Ok(best.unwrap_or_else(|| self.compress_raw(input).unwrap_or_else(|_| input.to_vec())))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_compressor_creation() {
        let compressor = SpeculativeCompressor::new();
        assert_eq!(compressor.strategies.len(), 5);
        assert_eq!(compressor.parallel_threshold, 4096);
    }

    #[test]
    fn test_speculative_fast_mode() {
        let compressor = SpeculativeCompressor::fast();
        assert_eq!(compressor.strategies.len(), 3);
    }

    #[test]
    fn test_speculative_best_mode() {
        let compressor = SpeculativeCompressor::best();
        assert_eq!(compressor.strategies.len(), 2);
    }

    #[test]
    fn test_compress_empty() {
        let compressor = SpeculativeCompressor::new();
        let result = compressor.compress(&[]).unwrap();

        // Should produce valid frame
        assert_eq!(&result[0..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    }

    #[test]
    fn test_compress_small() {
        let compressor = SpeculativeCompressor::new();
        let input = b"Hello, World!";
        let result = compressor.compress(input).unwrap();

        // Should produce valid frame
        assert_eq!(&result[0..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    }

    #[test]
    fn test_compress_rle_data() {
        let compressor = SpeculativeCompressor::new();
        let input = vec![b'A'; 1000];
        let result = compressor.compress(&input).unwrap();

        // RLE should be very efficient
        assert!(
            result.len() < 50,
            "RLE compression too large: {} bytes",
            result.len()
        );
    }

    #[test]
    fn test_compress_repetitive_data() {
        let compressor = SpeculativeCompressor::new();
        let input: Vec<u8> = (0..1000).flat_map(|_| b"abcd".iter().copied()).collect();
        let result = compressor.compress(&input).unwrap();

        // Should compress well
        assert!(result.len() < input.len() / 2);
    }

    #[test]
    fn test_compress_random_data_fast_path() {
        let compressor = SpeculativeCompressor::new();
        // Pseudo-random data (high entropy)
        let input: Vec<u8> = (0..1000).map(|i| ((i * 17 + 31) % 256) as u8).collect();
        let result = compressor.compress(&input).unwrap();

        // Should produce valid frame (may expand slightly)
        assert_eq!(&result[0..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    }

    #[test]
    fn test_speculative_picks_best() {
        let compressor = SpeculativeCompressor::new();

        // Text-like data benefits from lazy matching
        let input = b"the quick brown fox jumps over the lazy dog. \
                     the quick brown fox jumps over the lazy dog.";

        let result = compressor.compress(input).unwrap();

        // Verify it's smaller than raw
        assert!(result.len() < input.len());
    }

    #[test]
    fn test_with_checksum() {
        let compressor = SpeculativeCompressor::new().with_checksum(true);
        let input = b"Hello, World!";
        let result = compressor.compress(input).unwrap();

        // Should have checksum flag in descriptor
        assert!(result.len() >= 8);
        // Descriptor byte has checksum bit set
        assert_eq!(result[4] & 0x04, 0x04);
    }

    #[test]
    fn test_custom_strategies() {
        let compressor = SpeculativeCompressor::new().with_strategies(&[
            SpeculativeStrategy::GreedyFast,
            SpeculativeStrategy::LiteralsOnly,
        ]);

        assert_eq!(compressor.strategies.len(), 2);
    }

    #[test]
    fn test_parallel_threshold() {
        let compressor = SpeculativeCompressor::new().with_parallel_threshold(1024);

        assert_eq!(compressor.parallel_threshold, 1024);
    }

    #[test]
    fn test_strategy_all() {
        let all = SpeculativeStrategy::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&SpeculativeStrategy::GreedyFast));
        assert!(all.contains(&SpeculativeStrategy::LazyBest));
    }

    #[test]
    fn test_strategy_fast() {
        let fast = SpeculativeStrategy::fast();
        assert_eq!(fast.len(), 3);
        assert!(fast.contains(&SpeculativeStrategy::GreedyFast));
        assert!(!fast.contains(&SpeculativeStrategy::LazyBest));
    }

    #[test]
    fn test_strategy_best() {
        let best = SpeculativeStrategy::best();
        assert_eq!(best.len(), 2);
        assert!(best.contains(&SpeculativeStrategy::LazyDefault));
        assert!(best.contains(&SpeculativeStrategy::LazyBest));
    }

    #[test]
    fn test_compress_medium_data() {
        let compressor = SpeculativeCompressor::new();

        // Generate semi-compressible data (mix of patterns)
        let mut input = Vec::with_capacity(8192);
        for i in 0..2048 {
            // Mix of repetitive and varied data
            if i % 4 == 0 {
                input.extend_from_slice(b"test");
            } else {
                input.push((i % 256) as u8);
            }
        }

        let result = compressor.compress(&input).unwrap();

        // Should compress
        assert!(result.len() < input.len());
        // Should be valid frame
        assert_eq!(&result[0..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    }

    #[test]
    fn test_compress_large_for_parallel() {
        let compressor = SpeculativeCompressor::new();

        // Large repetitive data (should benefit from parallel execution)
        let input: Vec<u8> = (0..16384).map(|i| (i % 256) as u8).collect();

        let result = compressor.compress(&input).unwrap();

        // Should produce valid frame
        assert_eq!(&result[0..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_compression() {
        use std::time::Instant;

        let compressor = SpeculativeCompressor::new().with_parallel_threshold(1024);

        // Large data to trigger parallel path
        let input: Vec<u8> = (0..65536)
            .flat_map(|_| b"pattern_".iter().copied())
            .collect();

        let start = Instant::now();
        let result = compressor.compress(&input).unwrap();
        let elapsed = start.elapsed();

        // Should complete reasonably fast (parallel)
        println!(
            "Parallel compression: {} bytes -> {} bytes in {:?}",
            input.len(),
            result.len(),
            elapsed
        );

        // Should compress well
        assert!(result.len() < input.len() / 2);
    }
}
