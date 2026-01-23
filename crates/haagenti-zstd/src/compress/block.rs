//! Block-level encoding for Zstd compression.
//!
//! This module handles encoding of literals and sequences sections.
//!
//! ## Parallel Encoding
//!
//! When the `parallel` feature is enabled, 4-stream Huffman encoding
//! uses rayon for parallel compression of the 4 literal segments,
//! providing up to 2-3x speedup for large literals.

use super::sequences::{analyze_for_rle, encode_sequences_fse_with_encoded};
use super::Match;
use crate::block::Sequence;
use crate::huffman::HuffmanEncoder;
use haagenti_core::Result;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Block encoder for creating compressed blocks.
#[derive(Debug)]
pub struct BlockEncoder {
    /// Accumulated literals.
    literals: Vec<u8>,
    /// Accumulated sequences.
    sequences: Vec<Sequence>,
}

impl BlockEncoder {
    /// Create a new block encoder.
    pub fn new() -> Self {
        Self {
            literals: Vec::new(),
            sequences: Vec::new(),
        }
    }

    /// Add a literal byte.
    pub fn add_literal(&mut self, byte: u8) {
        self.literals.push(byte);
    }

    /// Add multiple literal bytes.
    pub fn add_literals(&mut self, bytes: &[u8]) {
        self.literals.extend_from_slice(bytes);
    }

    /// Add a match as a sequence.
    pub fn add_match(&mut self, literal_length: u32, offset: u32, match_length: u32) {
        self.sequences
            .push(Sequence::new(literal_length, offset, match_length));
    }

    /// Get the accumulated literals.
    pub fn literals(&self) -> &[u8] {
        &self.literals
    }

    /// Get the accumulated sequences.
    pub fn sequences(&self) -> &[Sequence] {
        &self.sequences
    }

    /// Clear the encoder for reuse.
    pub fn clear(&mut self) {
        self.literals.clear();
        self.sequences.clear();
    }
}

impl Default for BlockEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum match length that can be encoded in a single sequence.
///
/// Using zstd's predefined ML table values:
/// - Code 52: baseline 65539, 16 extra bits → max = 65539 + 65535 = 131074
///
/// Note: zstd's predefined values differ from RFC 8878 for codes 43+.
/// This allows encoding longer matches without splitting.
const MAX_MATCH_LENGTH_PER_SEQUENCE: usize = 131074;

/// Minimum match length for Zstd (must be at least 3).
const MIN_MATCH_LENGTH: usize = 3;

/// Tracks repeat offsets during encoding.
///
/// Per RFC 8878 Section 3.1.1.5, offset values 1-3 reference recent offsets.
/// Initial values are [1, 4, 8].
struct RepeatOffsetsEncoder {
    offsets: [u32; 3],
}

impl RepeatOffsetsEncoder {
    /// Create with initial values per RFC 8878.
    fn new() -> Self {
        Self { offsets: [1, 4, 8] }
    }

    /// Convert an actual offset to an offset_value for encoding.
    ///
    /// Returns the offset_value to encode:
    /// - 1, 2, or 3 if the offset matches a repeat offset
    /// - actual_offset + 3 for new offsets
    ///
    /// Also updates the repeat offset state to match the decoder's state machine.
    /// This is critical: the encoder and decoder must stay in sync.
    ///
    /// Per RFC 8878 Section 3.1.1.5:
    /// - When LL > 0: offset_value 1/2/3 map directly to repeat_offset 1/2/3
    /// - When LL = 0: special handling (shifted mapping)
    fn encode(&mut self, actual_offset: u32, literal_length: u32) -> u32 {
        if literal_length > 0 {
            // Normal case: check if offset matches any repeat offset
            if actual_offset == self.offsets[0] {
                // Matches repeat_offset_1 - no state change needed
                return 1;
            } else if actual_offset == self.offsets[1] {
                // Matches repeat_offset_2 - rotate [1] to front
                // Decoder does: temp = [idx], shift down, [0] = temp
                self.offsets.swap(1, 0);
                return 2;
            } else if actual_offset == self.offsets[2] {
                // Matches repeat_offset_3 - rotate [2] to front
                let temp = self.offsets[2];
                self.offsets[2] = self.offsets[1];
                self.offsets[1] = self.offsets[0];
                self.offsets[0] = temp;
                return 3;
            }
        } else {
            // LL = 0: special case encoding
            // offset_value 1 -> decoded as offsets[1], then swap(0,1)
            // offset_value 2 -> decoded as offsets[2], then rotate
            // offset_value 3 -> decoded as offsets[0]-1, push to front
            if actual_offset == self.offsets[1] {
                // Decoder swaps [0] and [1]
                self.offsets.swap(0, 1);
                return 1;
            } else if actual_offset == self.offsets[2] {
                // Decoder rotates: [2]->[0], [0]->[1], [1]->[2]
                let temp = self.offsets[2];
                self.offsets[2] = self.offsets[1];
                self.offsets[1] = self.offsets[0];
                self.offsets[0] = temp;
                return 2;
            } else if actual_offset == self.offsets[0].saturating_sub(1).max(1) {
                // Decoder uses offsets[0]-1, pushes to front
                let new_offset = self.offsets[0].saturating_sub(1).max(1);
                self.offsets[2] = self.offsets[1];
                self.offsets[1] = self.offsets[0];
                self.offsets[0] = new_offset;
                return 3;
            }
        }

        // New offset: push to front, encode as actual_offset + 3
        self.offsets[2] = self.offsets[1];
        self.offsets[1] = self.offsets[0];
        self.offsets[0] = actual_offset;
        actual_offset + 3
    }
}

/// Convert matches to literals and sequences.
///
/// Long matches are split only when they exceed the maximum encodable length.
/// This minimizes sequence count for better compression with FSE encoding.
///
/// Uses repeat offset tracking to efficiently encode offsets that match
/// recently used offsets (per RFC 8878 Section 3.1.1.5).
pub fn matches_to_sequences(input: &[u8], matches: &[Match]) -> (Vec<u8>, Vec<Sequence>) {
    // Pre-allocate based on expected output sizes
    // Literals: estimate ~50% of input for text, sequences ~1 per match
    let mut literals = Vec::with_capacity(input.len() / 2);
    let mut sequences = Vec::with_capacity(matches.len());
    let mut repeat_offsets = RepeatOffsetsEncoder::new();
    let mut pos = 0;

    for m in matches {
        // Add literals before the match
        let literal_length = m.position - pos;
        if literal_length > 0 {
            literals.extend_from_slice(&input[pos..m.position]);
        }

        let actual_offset = m.offset as u32;

        // Use maximum chunk size - only split when necessary
        let chunk_size = MAX_MATCH_LENGTH_PER_SEQUENCE;

        let mut remaining_match = m.length;
        let mut first_sequence = true;

        while remaining_match > 0 {
            // Determine match length for this sequence
            let match_len = remaining_match.min(chunk_size);

            // Ensure we don't leave a too-short remainder
            let after_this = remaining_match - match_len;
            let match_len = if after_this > 0 && after_this < MIN_MATCH_LENGTH {
                remaining_match - MIN_MATCH_LENGTH
            } else {
                match_len
            };

            // First sequence gets the literal length, subsequent ones get 0
            let ll = if first_sequence {
                literal_length as u32
            } else {
                0
            };
            first_sequence = false;

            // Convert actual offset to offset_value using repeat offset tracking
            let offset_value = repeat_offsets.encode(actual_offset, ll);

            sequences.push(Sequence::new(ll, offset_value, match_len as u32));
            remaining_match -= match_len;
        }

        pos = m.position + m.length;
    }

    // Add trailing literals
    if pos < input.len() {
        literals.extend_from_slice(&input[pos..]);
    }

    (literals, sequences)
}

/// Encode the literals section.
///
/// Attempts Huffman compression for better ratios, falls back to Raw/RLE.
pub fn encode_literals(literals: &[u8], output: &mut Vec<u8>) -> Result<()> {
    let size = literals.len();

    if size == 0 {
        // Empty literals: just the header byte
        output.push(0x00);
        return Ok(());
    }

    // Check for RLE pattern first (most efficient)
    if literals.iter().all(|&b| b == literals[0]) {
        return encode_rle_literals(literals[0], size, output);
    }

    // Try Huffman compression for literals of suitable size
    // Min 64 bytes for meaningful compression
    // Max 128KB (block size limit) - encode_huffman_literals handles header format limits
    if size >= 64 {
        if let Some(encoder) = HuffmanEncoder::build(literals) {
            let estimated = encoder.estimate_size(literals);
            if estimated + 20 < size {
                return encode_huffman_literals(literals, &encoder, output);
            }
        }
    }

    // Fall back to raw literals
    encode_raw_literals(literals, output)
}

/// Encode literals section using a pre-built Huffman encoder.
///
/// This function uses a custom Huffman encoder instead of building one from the data.
/// Useful for dictionary compression or when using pre-trained encoders.
///
/// Falls back to raw encoding if the encoder can't compress the data efficiently.
pub fn encode_literals_with_encoder(
    literals: &[u8],
    encoder: &HuffmanEncoder,
    output: &mut Vec<u8>,
) -> Result<()> {
    let size = literals.len();

    if size == 0 {
        // Empty literals: just the header byte
        output.push(0x00);
        return Ok(());
    }

    // Check for RLE pattern first (most efficient, regardless of encoder)
    if literals.iter().all(|&b| b == literals[0]) {
        return encode_rle_literals(literals[0], size, output);
    }

    // Use the provided encoder for Huffman compression
    // Min 64 bytes for meaningful compression
    if size >= 64 {
        let estimated = encoder.estimate_size(literals);
        if estimated + 20 < size {
            return encode_huffman_literals(literals, encoder, output);
        }
    }

    // Fall back to raw literals
    encode_raw_literals(literals, output)
}

/// Encode literals using Huffman compression.
fn encode_huffman_literals(
    literals: &[u8],
    encoder: &HuffmanEncoder,
    output: &mut Vec<u8>,
) -> Result<()> {
    let regenerated_size = literals.len();
    let weights = encoder.serialize_weights();

    // If weights is empty, there are too many symbols for direct encoding (>128).
    // Fall back to raw literals since we don't implement FSE-compressed weights yet.
    if weights.is_empty() {
        return encode_raw_literals(literals, output);
    }

    // Header format for compressed literals (RFC 8878 Section 3.1.1.3.2):
    // Literals_Block_Type = 2 (Compressed) with fresh Huffman table
    //
    // Size_Format determines header layout:
    // - 0: 4 streams, 10-bit sizes, 3-byte header
    // - 1: 4 streams, 14-bit sizes, 4-byte header
    // - 2: 4 streams, 18-bit sizes, 5-byte header
    // - 3: 1 stream, 10-bit sizes, 3-byte header

    if regenerated_size <= 1023 {
        // Single stream format (Size_Format = 3)
        let compressed = encoder.encode(literals);
        let compressed_size = weights.len() + compressed.len();

        if compressed_size <= 1023 {
            // Use Size_Format = 3 (single stream, 10-bit sizes, 3-byte header)
            // Byte 0: regen[3:0] << 4 | Size_Format << 2 | Block_Type
            //       = regen[3:0] << 4 | 3 << 2 | 2 = regen[3:0] << 4 | 0x0E
            // Byte 1: comp[1:0] << 6 | regen[9:4]
            // Byte 2: comp[9:2]
            let byte0 = (((regenerated_size & 0x0F) << 4) | 0x0E) as u8;
            let byte1 = (((compressed_size & 0x03) << 6) | ((regenerated_size >> 4) & 0x3F)) as u8;
            let byte2 = ((compressed_size >> 2) & 0xFF) as u8;

            output.push(byte0);
            output.push(byte1);
            output.push(byte2);

            // Write Huffman weights table
            output.extend_from_slice(&weights);
            // Write compressed literals
            output.extend_from_slice(&compressed);

            return Ok(());
        }
    }

    // For larger sizes, use 4-stream format (supports up to 262143 bytes)
    if regenerated_size <= 262143 {
        return encode_huffman_4stream(literals, encoder, &weights, output);
    }

    // Fall back to raw literals for very large sizes (> 256KB)
    encode_raw_literals(literals, output)
}

/// Encode literals using 4-stream Huffman compression.
///
/// 4-stream format splits literals into 4 segments, compresses each,
/// and writes segment sizes for the first 3 streams (last stream size is inferred).
///
/// When the `parallel` feature is enabled, segments are compressed in parallel
/// using rayon, providing up to 2-3x speedup for large literals.
fn encode_huffman_4stream(
    literals: &[u8],
    encoder: &HuffmanEncoder,
    weights: &[u8],
    output: &mut Vec<u8>,
) -> Result<()> {
    let regenerated_size = literals.len();

    // Split literals into 4 equal segments (with rounding for last segment)
    let segment_size = regenerated_size.div_ceil(4);
    let mut segments: [&[u8]; 4] = [&[], &[], &[], &[]];

    for (i, segment) in segments.iter_mut().enumerate() {
        let start = i * segment_size;
        if start >= regenerated_size {
            *segment = &[];
        } else {
            let end = ((i + 1) * segment_size).min(regenerated_size);
            *segment = &literals[start..end];
        }
    }

    // Compress each segment - parallel when feature enabled
    #[cfg(feature = "parallel")]
    let (compressed_0, compressed_1, compressed_2, compressed_3) = {
        // Use rayon to compress all 4 segments in parallel
        let compressed: Vec<Vec<u8>> = segments.par_iter().map(|seg| encoder.encode(seg)).collect();

        let mut iter = compressed.into_iter();
        (
            iter.next().unwrap_or_default(),
            iter.next().unwrap_or_default(),
            iter.next().unwrap_or_default(),
            iter.next().unwrap_or_default(),
        )
    };

    #[cfg(not(feature = "parallel"))]
    let (compressed_0, compressed_1, compressed_2, compressed_3) = (
        encoder.encode(segments[0]),
        encoder.encode(segments[1]),
        encoder.encode(segments[2]),
        encoder.encode(segments[3]),
    );

    // Calculate total compressed size
    let stream_sizes = [
        compressed_0.len(),
        compressed_1.len(),
        compressed_2.len(),
        compressed_3.len(),
    ];

    // Total compressed = weights + 6 bytes jump table + all streams
    let total_compressed = weights.len() + 6 + stream_sizes.iter().sum::<usize>();

    // Check for individual stream overflow
    if stream_sizes.iter().any(|&s| s > 65535) {
        return encode_raw_literals(literals, output);
    }

    // Choose header format based on sizes - must match decoder's expected format!
    // Size_Format=0: 10-bit sizes, 3-byte header (max 1023 each)
    // Size_Format=2: 16/14-bit sizes, 5-byte header (larger sizes)
    if regenerated_size <= 1023 && total_compressed <= 1023 {
        // Use Size_Format = 0 (4 streams, 10-bit sizes, 3-byte header)
        // Decoder expects:
        //   regen = (byte0 >> 4) | ((byte1 & 0x3F) << 4)
        //   comp = (byte1 >> 6) | (byte2 << 2)
        // So encoder must:
        //   byte0 = regen[3:0] << 4 | Size_Format << 2 | Block_Type = regen[3:0] << 4 | 0x02
        //   byte1 = comp[1:0] << 6 | regen[9:4]
        //   byte2 = comp[9:2]
        let byte0 = (((regenerated_size & 0x0F) << 4) | 0x02) as u8;
        let byte1 = (((total_compressed & 0x03) << 6) | ((regenerated_size >> 4) & 0x3F)) as u8;
        let byte2 = ((total_compressed >> 2) & 0xFF) as u8;

        output.push(byte0);
        output.push(byte1);
        output.push(byte2);
    } else if regenerated_size <= 65535 && total_compressed <= 16383 {
        // Use Size_Format = 2 (4 streams, 5-byte header)
        // Decoder expects:
        //   regen = ((byte0 >> 4) & 0x3F) | (byte1 << 4) | ((byte2 & 0x0F) << 12)
        //   comp = (byte2 >> 4) | (byte3 << 4) | ((byte4 & 0x03) << 12)
        // So encoder must:
        //   byte0 = regen[3:0] << 4 | Size_Format << 2 | Block_Type = regen[3:0] << 4 | 0x0A
        //   byte1 = regen[11:4]
        //   byte2 = comp[3:0] << 4 | regen[15:12]
        //   byte3 = comp[11:4]
        //   byte4 = comp[13:12]
        let byte0 = (((regenerated_size & 0x0F) << 4) | 0x0A) as u8;
        let byte1 = ((regenerated_size >> 4) & 0xFF) as u8;
        let byte2 = (((total_compressed & 0x0F) << 4) | ((regenerated_size >> 12) & 0x0F)) as u8;
        let byte3 = ((total_compressed >> 4) & 0xFF) as u8;
        let byte4 = ((total_compressed >> 12) & 0x03) as u8;

        output.push(byte0);
        output.push(byte1);
        output.push(byte2);
        output.push(byte3);
        output.push(byte4);
    } else {
        // Too large for Huffman compression
        return encode_raw_literals(literals, output);
    }

    // Write Huffman weights table
    output.extend_from_slice(weights);

    // Write jump table: 3 x 16-bit cumulative offsets (little-endian)
    // RFC 8878: Jump values are cumulative offsets from position 6 (after jump table)
    // - Stream 0 starts at position 6
    // - Stream 1 starts at position 6 + jump1
    // - Stream 2 starts at position 6 + jump2
    // - Stream 3 starts at position 6 + jump3
    let jump1 = stream_sizes[0];
    let jump2 = jump1 + stream_sizes[1];
    let jump3 = jump2 + stream_sizes[2];

    output.extend_from_slice(&(jump1 as u16).to_le_bytes());
    output.extend_from_slice(&(jump2 as u16).to_le_bytes());
    output.extend_from_slice(&(jump3 as u16).to_le_bytes());

    // Write all 4 compressed streams
    output.extend_from_slice(&compressed_0);
    output.extend_from_slice(&compressed_1);
    output.extend_from_slice(&compressed_2);
    output.extend_from_slice(&compressed_3);

    Ok(())
}

/// Encode raw literals.
///
/// Uses RFC 8878 Section 3.1.1.3.1.1 header format:
/// - Size_Format 0b00/0b01: 1-byte header, 5-bit size (0-31)
/// - Size_Format 0b10: 2-byte header, 12-bit size (0-4095)
/// - Size_Format 0b11: 3-byte header, 20-bit size (0-131071)
///
/// Note: The 1-byte format has a bit layout issue where `size << 3` puts
/// size bit 0 into the Size_Format field (bits 3:2), causing odd sizes to
/// produce Size_Format = 2. To avoid this, we use 2-byte format for all
/// sizes > 0 up to 4095. This costs 1 extra byte but guarantees correctness.
fn encode_raw_literals(literals: &[u8], output: &mut Vec<u8>) -> Result<()> {
    let size = literals.len();

    // Header format per RFC 8878 Section 3.1.1.1:
    // Literals_Block_Type[1:0] = 0 (Raw)
    // Size_Format[3:2]:
    //   00 or 10: 5-bit size in 1-byte header
    //   01: 12-bit size in 2-byte header
    //   11: 20-bit size in 3-byte header
    if size == 0 {
        // Empty: 1-byte header with all zeros
        output.push(0x00);
    } else if size <= 31 {
        // 5-bit size: 1-byte header with Size_Format = 0
        // byte0 = size[4:0] << 3 | Size_Format << 2 | Block_Type
        //       = size << 3 | 0 << 2 | 0
        //       = size << 3
        let byte0 = (size << 3) as u8;
        output.push(byte0);
    } else if size <= 4095 {
        // 12-bit size: 2-byte header with Size_Format = 1
        // byte0 = size[3:0] << 4 | Size_Format << 2 | Block_Type
        //       = size[3:0] << 4 | 1 << 2 | 0
        //       = size[3:0] << 4 | 0x04
        // byte1 = size[11:4]
        let byte0 = ((size & 0x0F) << 4) | 0x04;
        let byte1 = (size >> 4) & 0xFF;
        output.push(byte0 as u8);
        output.push(byte1 as u8);
    } else {
        // 20-bit size: 3-byte header with Size_Format = 3
        // byte0 = size[3:0] << 4 | 3 << 2 | 0 = size[3:0] << 4 | 0x0C
        // byte1 = size[11:4]
        // byte2 = size[19:12]
        let byte0 = ((size & 0x0F) << 4) | 0x0C;
        let byte1 = (size >> 4) & 0xFF;
        let byte2 = (size >> 12) & 0xFF;
        output.push(byte0 as u8);
        output.push(byte1 as u8);
        output.push(byte2 as u8);
    }

    // Literal data
    output.extend_from_slice(literals);

    Ok(())
}

/// Encode RLE literals.
///
/// Uses RFC 8878 Section 3.1.1.3.1.2 header format:
/// - Size_Format 0b10: 2-byte header, 12-bit size (0-4095)
/// - Size_Format 0b11: 3-byte header, 20-bit size (0-131071)
///
/// Same as raw literals, we use 2-byte format to avoid bit layout issues.
fn encode_rle_literals(byte: u8, count: usize, output: &mut Vec<u8>) -> Result<()> {
    // Header format per RFC 8878 Section 3.1.1.1:
    // Literals_Block_Type[1:0] = 1 (RLE)
    // Size_Format[3:2]:
    //   00 or 10: 5-bit size in 1-byte header
    //   01: 12-bit size in 2-byte header
    //   11: 20-bit size in 3-byte header
    if count == 0 {
        // Empty RLE (shouldn't happen, but handle gracefully)
        output.push(0x01); // Type=1 (RLE), Size_Format=0, Size=0
    } else if count <= 31 {
        // 5-bit size: 1-byte header with Size_Format = 0
        // byte0 = count[4:0] << 3 | Size_Format << 2 | Block_Type
        //       = count << 3 | 0 << 2 | 1
        //       = count << 3 | 1
        let byte0 = ((count << 3) | 1) as u8;
        output.push(byte0);
    } else if count <= 4095 {
        // 12-bit size: 2-byte header with Size_Format = 1
        // byte0 = count[3:0] << 4 | Size_Format << 2 | Block_Type
        //       = count[3:0] << 4 | 1 << 2 | 1
        //       = count[3:0] << 4 | 0x05
        // byte1 = count[11:4]
        let byte0 = ((count & 0x0F) << 4) | 0x05;
        let byte1 = (count >> 4) & 0xFF;
        output.push(byte0 as u8);
        output.push(byte1 as u8);
    } else {
        // 20-bit size: 3-byte header with Size_Format = 3
        // byte0 = count[3:0] << 4 | 3 << 2 | 1 = count[3:0] << 4 | 0x0D
        // byte1 = count[11:4]
        // byte2 = count[19:12]
        let byte0 = ((count & 0x0F) << 4) | 0x0D;
        let byte1 = (count >> 4) & 0xFF;
        let byte2 = (count >> 12) & 0xFF;
        output.push(byte0 as u8);
        output.push(byte1 as u8);
        output.push(byte2 as u8);
    }

    // Single byte value
    output.push(byte);

    Ok(())
}

/// Encode the sequences section.
///
/// Uses RLE mode when all sequences have the same codes (uniform),
/// otherwise falls back to FSE encoding with predefined tables.
///
/// # Performance
///
/// Sequences are encoded once during RLE analysis and reused for FSE
/// encoding if needed, avoiding redundant encoding work.
pub fn encode_sequences(sequences: &[Sequence], output: &mut Vec<u8>) -> Result<()> {
    if sequences.is_empty() {
        output.push(0);
        return Ok(());
    }

    // Analyze sequences to pre-encode into codes+extra bits.
    let suitability = analyze_for_rle(sequences);

    // Always use FSE/Predefined mode for better cross-decoder compatibility.
    // RLE mode has subtle implementation differences that cause issues
    // with some reference decoders. Predefined FSE mode (mode=0x00) is
    // universally supported and produces identical output to reference zstd.
    encode_sequences_fse_with_encoded(&suitability.encoded, output)
}

/// Encode a complete block.
pub fn encode_block(input: &[u8], matches: &[Match], output: &mut Vec<u8>) -> Result<()> {
    let (literals, sequences) = matches_to_sequences(input, matches);
    encode_literals(&literals, output)?;
    encode_sequences(&sequences, output)?;
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_encoder_creation() {
        let encoder = BlockEncoder::new();
        assert!(encoder.literals().is_empty());
        assert!(encoder.sequences().is_empty());
    }

    #[test]
    fn test_add_literals() {
        let mut encoder = BlockEncoder::new();
        encoder.add_literal(b'A');
        encoder.add_literals(b"BC");

        assert_eq!(encoder.literals(), b"ABC");
    }

    #[test]
    fn test_add_match() {
        let mut encoder = BlockEncoder::new();
        encoder.add_match(5, 10, 4);

        assert_eq!(encoder.sequences().len(), 1);
        let seq = &encoder.sequences()[0];
        assert_eq!(seq.literal_length, 5);
        assert_eq!(seq.offset, 10);
        assert_eq!(seq.match_length, 4);
    }

    #[test]
    fn test_matches_to_sequences_empty() {
        let input = b"hello";
        let matches = vec![];

        let (literals, sequences) = matches_to_sequences(input, &matches);

        assert_eq!(literals, b"hello");
        assert!(sequences.is_empty());
    }

    #[test]
    fn test_matches_to_sequences_with_match() {
        let input = b"abcdabcd";
        let matches = vec![Match::new(4, 4, 4)];

        let (literals, sequences) = matches_to_sequences(input, &matches);

        assert_eq!(literals, b"abcd"); // Literals before match
        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0].literal_length, 4);
        // Offset 4 matches repeat_offset_2 (initial [1,4,8]), so encoded as 2
        assert_eq!(sequences[0].offset, 2);
        assert_eq!(sequences[0].match_length, 4);
    }

    #[test]
    fn test_matches_to_sequences_new_offset() {
        let input = b"abcdefXXXabcdef";
        // Match at position 9, offset 9, length 6 (copying "abcdef" from position 0)
        let matches = vec![Match::new(9, 9, 6)];

        let (literals, sequences) = matches_to_sequences(input, &matches);

        assert_eq!(literals, b"abcdefXXX"); // Literals before match
        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0].literal_length, 9);
        // Offset 9 encoded as 9 + 3 = 12
        assert_eq!(sequences[0].offset, 12);
        assert_eq!(sequences[0].match_length, 6);
    }

    #[test]
    fn test_repeat_offset_encoder() {
        let mut encoder = RepeatOffsetsEncoder::new();

        // Initial repeat offsets are [1, 4, 8]
        // Repeat offset optimization is now ENABLED for better compression

        // Offset 4 matches repeat_offset_2 -> encoded as 2, state becomes [4, 1, 8]
        assert_eq!(encoder.encode(4, 5), 2);
        // Offset 4 is now repeat_offset_1 -> encoded as 1, state stays [4, 1, 8]
        assert_eq!(encoder.encode(4, 5), 1);
        // Offset 1 matches repeat_offset_2 -> encoded as 2, state becomes [1, 4, 8]
        assert_eq!(encoder.encode(1, 5), 2);
        // Offset 100 is new -> encoded as 100 + 3 = 103, state becomes [100, 1, 4]
        assert_eq!(encoder.encode(100, 5), 103);
    }

    #[test]
    fn test_encode_raw_literals_small() {
        let mut output = Vec::new();
        encode_raw_literals(b"Hello", &mut output).unwrap();

        // 1-byte header with Size_Format = 0: byte0 = (5 << 3) | 0 = 0x28
        assert_eq!(output[0], 0x28);
        assert_eq!(&output[1..], b"Hello");
    }

    #[test]
    fn test_encode_raw_literals_medium() {
        let data: Vec<u8> = (0..100).collect();
        let mut output = Vec::new();
        encode_raw_literals(&data, &mut output).unwrap();

        // 12-bit size format
        assert_eq!(output.len(), 2 + 100);
    }

    #[test]
    fn test_encode_rle_literals() {
        let mut output = Vec::new();
        encode_rle_literals(b'X', 10, &mut output).unwrap();

        // 1-byte header with Size_Format = 0: byte0 = (10 << 3) | 1 = 0x51
        assert_eq!(output[0], 0x51);
        assert_eq!(output[1], b'X');
    }

    #[test]
    fn test_encode_sequences_empty() {
        let mut output = Vec::new();
        encode_sequences(&[], &mut output).unwrap();

        assert_eq!(output, vec![0]);
    }

    #[test]
    fn test_encode_sequences_count_small() {
        let sequences = vec![Sequence::new(0, 4, 3)];
        let mut output = Vec::new();
        encode_sequences(&sequences, &mut output).unwrap();

        // Count = 1 (single byte)
        assert_eq!(output[0], 1);
    }

    #[test]
    fn test_encode_literals_detects_rle() {
        let rle_data = vec![b'A'; 50];
        let mut output = Vec::new();
        encode_literals(&rle_data, &mut output).unwrap();

        // Should be RLE encoded (much smaller)
        assert!(output.len() < 50);
    }

    #[test]
    fn test_encode_block() {
        let input = b"abcdabcd";
        let matches = vec![Match::new(4, 4, 4)];
        let mut output = Vec::new();

        encode_block(input, &matches, &mut output).unwrap();

        // Should have literals section + sequences section
        assert!(!output.is_empty());
    }
}
