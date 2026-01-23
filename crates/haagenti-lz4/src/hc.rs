//! LZ4-HC (High Compression) implementation.
//!
//! LZ4-HC trades compression speed for better compression ratios by using
//! hash chains to find optimal matches instead of accepting the first match.
//!
//! ## Compression Levels
//!
//! - Level 1-3: Fast HC (short chain traversal)
//! - Level 4-6: Default HC (medium chain traversal)
//! - Level 7-9: Best HC (deep chain traversal, lazy matching)
//!
//! ## Algorithm
//!
//! Unlike standard LZ4 which uses a simple hash table (one entry per hash),
//! LZ4-HC uses hash chains:
//!
//! 1. Hash table maps hash → most recent position
//! 2. Chain table links each position → previous position with same hash
//! 3. Match search traverses the chain to find the longest match
//! 4. Chain depth controlled by compression level
//!
//! For levels 7+, lazy matching is enabled: we check if the next position
//! has a better match before committing to the current one.

use haagenti_core::{Error, Result};

use crate::block::{write_last_literals, MIN_MATCH, MAX_MATCH};

/// Hash table size (64KB = 2^16 entries).
const HASH_TABLE_SIZE: usize = 1 << 16;

/// Window size for back-references (64KB).
const WINDOW_SIZE: usize = 1 << 16;

/// Chain table size (matches window size).
const CHAIN_TABLE_SIZE: usize = WINDOW_SIZE;

/// Number of bytes at end of input that won't be compressed.
const LAST_LITERALS: usize = 5;

/// Minimum input size to attempt compression.
const MIN_INPUT_SIZE: usize = 13;

/// Maximum chain depth per compression level.
const CHAIN_DEPTHS: [usize; 10] = [
    4,    // Level 0: Minimal
    8,    // Level 1: Fast
    16,   // Level 2: Fast
    32,   // Level 3: Fast
    64,   // Level 4: Default
    128,  // Level 5: Default
    256,  // Level 6: Default
    512,  // Level 7: Best
    1024, // Level 8: Best
    4096, // Level 9: Ultra
];

/// Hash function for 5-byte sequence (better distribution for HC).
#[inline(always)]
fn hash5(data: &[u8], pos: usize) -> usize {
    let v = u64::from_le_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
        data[pos + 4],
        0,
        0,
        0,
    ]);
    ((v.wrapping_mul(889523592379_u64)) >> 24) as usize & (HASH_TABLE_SIZE - 1)
}

/// Count matching bytes between two positions.
#[inline]
fn count_match(data: &[u8], pos1: usize, pos2: usize, limit: usize) -> usize {
    let mut len = 0;
    let max_len = (limit - pos2).min(MAX_MATCH - MIN_MATCH);

    // Fast path: compare 8 bytes at a time
    while len + 8 <= max_len {
        let p1 = pos1 + len;
        let p2 = pos2 + len;

        if p1 + 8 > data.len() || p2 + 8 > data.len() {
            break;
        }

        let v1 = u64::from_le_bytes(data[p1..p1 + 8].try_into().unwrap());
        let v2 = u64::from_le_bytes(data[p2..p2 + 8].try_into().unwrap());

        if v1 != v2 {
            // Count matching bytes in the u64
            let diff = v1 ^ v2;
            len += (diff.trailing_zeros() / 8) as usize;
            return len;
        }
        len += 8;
    }

    // Byte-by-byte for remainder
    while len < max_len && pos1 + len < data.len() && pos2 + len < data.len() {
        if data[pos1 + len] != data[pos2 + len] {
            break;
        }
        len += 1;
    }

    len
}

/// LZ4-HC compression context.
pub struct Lz4HcContext {
    /// Hash table: maps hash → most recent position with that hash.
    hash_table: Vec<u32>,

    /// Chain table: maps position → previous position with same hash.
    chain_table: Vec<u32>,

    /// Base position for window calculations.
    base: usize,

    /// Maximum chain depth for this compression level.
    max_chain: usize,

    /// Enable lazy matching (levels 7+).
    lazy_matching: bool,
}

impl Lz4HcContext {
    /// Create a new HC context with the specified compression level.
    pub fn new(level: usize) -> Self {
        let level = level.min(9);
        Self {
            hash_table: vec![0; HASH_TABLE_SIZE],
            chain_table: vec![0; CHAIN_TABLE_SIZE],
            base: 0,
            max_chain: CHAIN_DEPTHS[level],
            lazy_matching: level >= 7,
        }
    }

    /// Reset the context for a new compression operation.
    pub fn reset(&mut self) {
        self.hash_table.fill(0);
        self.chain_table.fill(0);
        self.base = 0;
    }

    /// Insert a position into the hash chain.
    #[inline]
    fn insert(&mut self, data: &[u8], pos: usize) {
        if pos + 5 > data.len() {
            return;
        }

        let h = hash5(data, pos);
        let chain_pos = pos & (CHAIN_TABLE_SIZE - 1);

        // Link to previous position with same hash
        self.chain_table[chain_pos] = self.hash_table[h];
        self.hash_table[h] = pos as u32;
    }

    /// Insert multiple positions (used after a match).
    fn insert_many(&mut self, data: &[u8], start: usize, end: usize) {
        for pos in start..end {
            self.insert(data, pos);
        }
    }

    /// Find the best match at the current position.
    fn find_best_match(
        &self,
        data: &[u8],
        pos: usize,
        match_limit: usize,
    ) -> Option<(usize, usize)> {
        if pos + 5 > data.len() {
            return None;
        }

        let h = hash5(data, pos);
        let mut chain_pos = self.hash_table[h] as usize;

        let mut best_len = MIN_MATCH - 1;
        let mut best_offset = 0;
        let mut chain_count = 0;

        while chain_pos > 0 && chain_count < self.max_chain {
            // Check if position is valid (must come before subtraction to avoid overflow)
            if chain_pos >= pos {
                break;
            }

            // Check if position is within window
            if pos - chain_pos > WINDOW_SIZE - 1 {
                break;
            }

            // Quick check: first 4 bytes must match
            if data[chain_pos] == data[pos]
                && data[chain_pos + 1] == data[pos + 1]
                && data[chain_pos + 2] == data[pos + 2]
                && data[chain_pos + 3] == data[pos + 3]
            {
                // Count full match length
                let len = MIN_MATCH + count_match(
                    data,
                    chain_pos + MIN_MATCH,
                    pos + MIN_MATCH,
                    match_limit,
                );

                if len > best_len {
                    best_len = len;
                    best_offset = pos - chain_pos;

                    // Early exit if we found a very long match
                    if len >= 128 {
                        break;
                    }
                }
            }

            // Follow chain to previous position
            let next_chain = self.chain_table[chain_pos & (CHAIN_TABLE_SIZE - 1)] as usize;
            if next_chain >= chain_pos || next_chain == 0 {
                break;
            }
            chain_pos = next_chain;
            chain_count += 1;
        }

        if best_len >= MIN_MATCH && best_offset > 0 && best_offset <= WINDOW_SIZE - 1 {
            Some((best_offset, best_len))
        } else {
            None
        }
    }
}

/// Write a sequence (literals + match) to output.
fn write_sequence(
    input: &[u8],
    output: &mut [u8],
    mut pos: usize,
    literal_start: usize,
    literal_len: usize,
    offset: u16,
    match_len: usize,
) -> Result<usize> {
    // Calculate token
    let ll_token = literal_len.min(15);
    let ml_token = (match_len - MIN_MATCH).min(15);
    let token = ((ll_token << 4) | ml_token) as u8;

    // Check output space
    let needed = 1 + (literal_len / 255) + 1 + literal_len + 2 + ((match_len - MIN_MATCH) / 255) + 1;
    if pos + needed > output.len() {
        return Err(Error::buffer_too_small(pos + needed, output.len()));
    }

    // Write token
    output[pos] = token;
    pos += 1;

    // Write extra literal length bytes
    if literal_len >= 15 {
        let mut remaining = literal_len - 15;
        while remaining >= 255 {
            output[pos] = 255;
            pos += 1;
            remaining -= 255;
        }
        output[pos] = remaining as u8;
        pos += 1;
    }

    // Write literals
    output[pos..pos + literal_len].copy_from_slice(&input[literal_start..literal_start + literal_len]);
    pos += literal_len;

    // Write match offset
    let offset_bytes = offset.to_le_bytes();
    output[pos] = offset_bytes[0];
    output[pos + 1] = offset_bytes[1];
    pos += 2;

    // Write extra match length bytes
    if match_len - MIN_MATCH >= 15 {
        let mut remaining = match_len - MIN_MATCH - 15;
        while remaining >= 255 {
            output[pos] = 255;
            pos += 1;
            remaining -= 255;
        }
        output[pos] = remaining as u8;
        pos += 1;
    }

    Ok(pos)
}

/// Compress data using LZ4-HC algorithm.
///
/// Returns the number of bytes written to output.
pub fn compress_hc(input: &[u8], output: &mut [u8], level: usize) -> Result<usize> {
    let input_len = input.len();

    // Handle small inputs - use standard LZ4
    if input_len < MIN_INPUT_SIZE {
        return crate::block::compress_block(input, output);
    }

    let mut ctx = Lz4HcContext::new(level);
    let match_limit = input_len.saturating_sub(LAST_LITERALS);
    let mf_limit = match_limit.saturating_sub(MIN_MATCH);

    let mut input_pos = 0;
    let mut output_pos = 0;
    let mut anchor = 0;

    // Pre-populate hash table with initial positions
    for i in 0..input_len.min(WINDOW_SIZE).saturating_sub(MIN_MATCH) {
        ctx.insert(input, i);
    }

    while input_pos < mf_limit {
        // Find best match at current position
        let match_result = ctx.find_best_match(input, input_pos, match_limit);

        if let Some((offset, match_len)) = match_result {
            // Lazy matching: check if next position has better match
            let use_current = if ctx.lazy_matching && input_pos + 1 < mf_limit {
                if let Some((_, next_len)) = ctx.find_best_match(input, input_pos + 1, match_limit) {
                    // Use current match if it's at least as good as (next + 1)
                    match_len >= next_len + 1
                } else {
                    true
                }
            } else {
                true
            };

            if use_current {
                let literal_len = input_pos - anchor;

                // Write the sequence
                output_pos = write_sequence(
                    input,
                    output,
                    output_pos,
                    anchor,
                    literal_len,
                    offset as u16,
                    match_len,
                )?;

                // Update hash table with positions we're skipping
                let old_pos = input_pos;
                input_pos += match_len;
                anchor = input_pos;

                // Insert skipped positions into hash chain
                ctx.insert_many(input, old_pos + 1, input_pos.min(mf_limit));
            } else {
                // Skip this position, try next
                ctx.insert(input, input_pos);
                input_pos += 1;
            }
        } else {
            // No match found, advance
            ctx.insert(input, input_pos);
            input_pos += 1;
        }
    }

    // Write remaining literals
    let literal_len = input_len - anchor;
    if literal_len > 0 {
        output_pos = write_last_literals(input, output, output_pos, anchor, literal_len)?;
    }

    Ok(output_pos)
}

/// LZ4-HC compressor.
#[derive(Debug, Clone)]
pub struct Lz4HcCompressor {
    level: usize,
}

impl Lz4HcCompressor {
    /// Create a new LZ4-HC compressor with the specified level (1-9).
    pub fn new(level: usize) -> Self {
        Self {
            level: level.clamp(1, 9),
        }
    }

    /// Create with default level (4).
    pub fn default_level() -> Self {
        Self::new(4)
    }

    /// Compress data using LZ4-HC.
    pub fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        let max_size = crate::block::max_compressed_size(input.len());
        let mut output = vec![0u8; max_size];
        let len = compress_hc(input, &mut output, self.level)?;
        output.truncate(len);
        Ok(output)
    }

    /// Compress data into provided buffer.
    pub fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        compress_hc(input, output, self.level)
    }

    /// Get the compression level.
    pub fn level(&self) -> usize {
        self.level
    }
}

impl Default for Lz4HcCompressor {
    fn default() -> Self {
        Self::default_level()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hc_small_input() {
        let input = b"Hello, World!";
        let compressor = Lz4HcCompressor::new(4);
        let compressed = compressor.compress(input).unwrap();

        // Verify with standard LZ4 decompressor
        let mut decompressed = vec![0u8; input.len()];
        let len = crate::block::decompress_block(&compressed, &mut decompressed, input.len()).unwrap();

        assert_eq!(len, input.len());
        assert_eq!(&decompressed[..], input);
    }

    #[test]
    fn test_hc_repetitive() {
        let input = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let compressor = Lz4HcCompressor::new(4);
        let compressed = compressor.compress(input).unwrap();

        // Should compress well
        assert!(compressed.len() < input.len());

        // Verify roundtrip
        let mut decompressed = vec![0u8; input.len()];
        let len = crate::block::decompress_block(&compressed, &mut decompressed, input.len()).unwrap();

        assert_eq!(len, input.len());
        assert_eq!(&decompressed[..], input);
    }

    #[test]
    fn test_hc_pattern() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = pattern.iter().cycle().take(5000).copied().collect();

        let compressor = Lz4HcCompressor::new(9); // Best compression
        let compressed = compressor.compress(&input).unwrap();

        // HC should compress the data well
        assert!(compressed.len() < input.len());

        // Verify roundtrip
        let mut decompressed = vec![0u8; input.len()];
        let len = crate::block::decompress_block(&compressed, &mut decompressed, input.len()).unwrap();

        assert_eq!(len, input.len());
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_hc_levels() {
        let pattern = b"ABCDEFGHIJKLMNOP";
        let input: Vec<u8> = pattern.iter().cycle().take(10000).copied().collect();

        let mut sizes = Vec::new();

        for level in 1..=9 {
            let compressor = Lz4HcCompressor::new(level);
            let compressed = compressor.compress(&input).unwrap();
            sizes.push((level, compressed.len()));

            // Verify all levels decompress correctly
            let mut decompressed = vec![0u8; input.len()];
            let len = crate::block::decompress_block(&compressed, &mut decompressed, input.len()).unwrap();
            assert_eq!(len, input.len());
            assert_eq!(decompressed, input, "Level {} failed roundtrip", level);
        }

        // Higher levels should generally produce smaller output
        // (not strictly monotonic, but trend should be downward)
        let first_half_avg: f64 = sizes[0..4].iter().map(|(_, s)| *s as f64).sum::<f64>() / 4.0;
        let second_half_avg: f64 = sizes[5..9].iter().map(|(_, s)| *s as f64).sum::<f64>() / 4.0;
        assert!(second_half_avg <= first_half_avg, "Higher levels should compress better");
    }

    #[test]
    fn test_hc_interop_lz4flex() {
        let pattern = b"LZ4-HC interoperability test data with repeating patterns. ";
        let input: Vec<u8> = pattern.iter().cycle().take(2000).copied().collect();

        let compressor = Lz4HcCompressor::new(6);
        let compressed = compressor.compress(&input).unwrap();

        // lz4_flex should be able to decompress our HC output
        let decompressed = lz4_flex::decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_hc_vs_standard_ratio() {
        // Data that benefits from better match finding: longer repeating patterns
        // where HC's deeper search can find better matches
        let mut input = Vec::new();
        let phrases = [
            b"The quick brown fox jumps over the lazy dog. ".as_slice(),
            b"Pack my box with five dozen liquor jugs. ",
            b"How vexingly quick daft zebras jump! ",
            b"The five boxing wizards jump quickly. ",
        ];
        for i in 0..500 {
            input.extend_from_slice(phrases[i % phrases.len()]);
        }

        let hc = Lz4HcCompressor::new(9);
        let hc_compressed = hc.compress(&input).unwrap();

        let mut std_output = vec![0u8; crate::block::max_compressed_size(input.len())];
        let std_len = crate::block::compress_block(&input, &mut std_output).unwrap();

        // HC should achieve comparable or better compression
        // Allow small variance (< 1%) since some patterns may not benefit from deeper search
        let variance_allowed = (std_len as f64 * 0.01).ceil() as usize;
        assert!(
            hc_compressed.len() <= std_len + variance_allowed,
            "HC ({}) should be within 1% of standard ({})",
            hc_compressed.len(),
            std_len
        );
    }
}
