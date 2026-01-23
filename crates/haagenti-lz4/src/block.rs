//! LZ4 block format encoding and decoding.
//!
//! LZ4 block format is a sequence of:
//! - Token byte: (literal_length: 4 bits, match_length: 4 bits)
//! - Optional additional literal length bytes (if literal_length == 15)
//! - Literal bytes
//! - Match offset (2 bytes, little-endian)
//! - Optional additional match length bytes (if match_length == 15)
//!
//! The last sequence has no match (just literals).

use haagenti_core::{Error, Result};

/// Minimum match length for LZ4 (matches must be at least 4 bytes).
pub const MIN_MATCH: usize = 4;

/// Maximum match length we'll look for.
pub const MAX_MATCH: usize = 65535 + MIN_MATCH;

/// Hash table size (64KB = 2^16 entries).
const HASH_TABLE_SIZE: usize = 1 << 16;

/// Acceleration factor for fast scanning (skip bytes when no match found).
const ACCELERATION: usize = 1;

/// Number of bytes at end of input that won't be compressed (safety margin).
const LAST_LITERALS: usize = 5;

/// Minimum input size to attempt compression.
const MIN_INPUT_SIZE: usize = 13;

/// Hash function for 4-byte sequence.
#[inline(always)]
fn hash(data: u32) -> usize {
    // Multiplicative hash with good distribution
    ((data.wrapping_mul(2654435761)) >> 16) as usize & (HASH_TABLE_SIZE - 1)
}

/// Read 4 bytes as u32 (little-endian).
#[inline(always)]
fn read_u32_le(data: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

/// Read 2 bytes as u16 (little-endian).
#[inline(always)]
fn read_u16_le(data: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes([data[pos], data[pos + 1]])
}

/// Write u16 as 2 bytes (little-endian).
#[inline(always)]
fn write_u16_le(data: &mut [u8], pos: usize, val: u16) {
    let bytes = val.to_le_bytes();
    data[pos] = bytes[0];
    data[pos + 1] = bytes[1];
}

/// Count matching bytes between two positions.
#[inline]
fn count_match(data: &[u8], mut pos1: usize, mut pos2: usize, limit: usize) -> usize {
    let start = pos2;
    while pos2 < limit && data[pos1] == data[pos2] {
        pos1 += 1;
        pos2 += 1;
    }
    pos2 - start
}

/// Compress data using LZ4 block format.
///
/// Returns the number of bytes written to output.
pub fn compress_block(input: &[u8], output: &mut [u8]) -> Result<usize> {
    let input_len = input.len();

    // Handle small inputs
    if input_len < MIN_INPUT_SIZE {
        return compress_literals_only(input, output);
    }

    // Hash table: maps hash -> position in input
    let mut hash_table = [0usize; HASH_TABLE_SIZE];

    let mut input_pos = 0;
    let mut output_pos = 0;
    let mut anchor = 0; // Start of current literal run

    let match_limit = input_len.saturating_sub(LAST_LITERALS);
    let mf_limit = match_limit.saturating_sub(MIN_MATCH);

    // Main compression loop
    while input_pos < mf_limit {
        // Hash current position
        let h = hash(read_u32_le(input, input_pos));
        let match_pos = hash_table[h];
        hash_table[h] = input_pos;

        // Check for match
        if match_pos > 0
            && input_pos - match_pos <= 65535
            && read_u32_le(input, match_pos) == read_u32_le(input, input_pos)
        {
            // Found a match! Extend it.
            let match_len = MIN_MATCH
                + count_match(
                    input,
                    match_pos + MIN_MATCH,
                    input_pos + MIN_MATCH,
                    match_limit,
                );

            let literal_len = input_pos - anchor;
            let offset = (input_pos - match_pos) as u16;

            // Write sequence
            output_pos = write_sequence(
                input,
                output,
                output_pos,
                anchor,
                literal_len,
                offset,
                match_len,
            )?;

            // Move past the match
            input_pos += match_len;
            anchor = input_pos;

            // Update hash table for positions we skipped
            if input_pos < mf_limit {
                hash_table[hash(read_u32_le(input, input_pos - 2))] = input_pos - 2;
            }
        } else {
            // No match, advance
            input_pos += ACCELERATION;
        }
    }

    // Write remaining literals
    let literal_len = input_len - anchor;
    if literal_len > 0 {
        output_pos = write_last_literals(input, output, output_pos, anchor, literal_len)?;
    }

    Ok(output_pos)
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

    // Check output space (rough estimate)
    let needed =
        1 + (literal_len / 255) + 1 + literal_len + 2 + ((match_len - MIN_MATCH) / 255) + 1;
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
    output[pos..pos + literal_len]
        .copy_from_slice(&input[literal_start..literal_start + literal_len]);
    pos += literal_len;

    // Write match offset
    write_u16_le(output, pos, offset);
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

/// Write final literals (no match follows).
pub(crate) fn write_last_literals(
    input: &[u8],
    output: &mut [u8],
    mut pos: usize,
    literal_start: usize,
    literal_len: usize,
) -> Result<usize> {
    // Token with match_length = 0 (but we don't write offset)
    let ll_token = literal_len.min(15);
    let token = (ll_token << 4) as u8;

    // Check output space
    let needed = 1 + (literal_len / 255) + 1 + literal_len;
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
    output[pos..pos + literal_len]
        .copy_from_slice(&input[literal_start..literal_start + literal_len]);
    pos += literal_len;

    Ok(pos)
}

/// Compress input that's too small to have matches.
fn compress_literals_only(input: &[u8], output: &mut [u8]) -> Result<usize> {
    write_last_literals(input, output, 0, 0, input.len())
}

/// Decompress LZ4 block format.
///
/// `output_size` is the expected decompressed size (must be known).
pub fn decompress_block(input: &[u8], output: &mut [u8], output_size: usize) -> Result<usize> {
    let mut input_pos = 0;
    let mut output_pos = 0;

    while input_pos < input.len() {
        // Read token
        if input_pos >= input.len() {
            return Err(Error::unexpected_eof(input_pos));
        }
        let token = input[input_pos];
        input_pos += 1;

        let mut literal_len = (token >> 4) as usize;
        let mut match_len = (token & 0x0F) as usize;

        // Read extra literal length bytes
        if literal_len == 15 {
            loop {
                if input_pos >= input.len() {
                    return Err(Error::unexpected_eof(input_pos));
                }
                let byte = input[input_pos];
                input_pos += 1;
                literal_len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }

        // Copy literals
        if literal_len > 0 {
            if input_pos + literal_len > input.len() {
                return Err(Error::unexpected_eof(input_pos));
            }
            if output_pos + literal_len > output.len() {
                return Err(Error::buffer_too_small(
                    output_pos + literal_len,
                    output.len(),
                ));
            }
            output[output_pos..output_pos + literal_len]
                .copy_from_slice(&input[input_pos..input_pos + literal_len]);
            input_pos += literal_len;
            output_pos += literal_len;
        }

        // Check if this is the last sequence (no match)
        if output_pos >= output_size {
            break;
        }

        // Read match offset
        if input_pos + 2 > input.len() {
            return Err(Error::unexpected_eof(input_pos));
        }
        let offset = read_u16_le(input, input_pos) as usize;
        input_pos += 2;

        if offset == 0 {
            return Err(Error::corrupted("invalid zero offset"));
        }
        if offset > output_pos {
            return Err(Error::corrupted_at("offset beyond output", output_pos));
        }

        // Read extra match length bytes
        match_len += MIN_MATCH;
        if (token & 0x0F) == 15 {
            loop {
                if input_pos >= input.len() {
                    return Err(Error::unexpected_eof(input_pos));
                }
                let byte = input[input_pos];
                input_pos += 1;
                match_len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }

        // Copy match (may overlap!)
        let match_start = output_pos - offset;
        if output_pos + match_len > output.len() {
            return Err(Error::buffer_too_small(
                output_pos + match_len,
                output.len(),
            ));
        }

        // Handle overlapping copy
        if offset >= match_len {
            // Non-overlapping: can use copy_from_slice
            output.copy_within(match_start..match_start + match_len, output_pos);
        } else {
            // Overlapping: copy byte by byte
            for i in 0..match_len {
                output[output_pos + i] = output[match_start + i];
            }
        }
        output_pos += match_len;
    }

    Ok(output_pos)
}

/// Calculate maximum compressed size for given input length.
/// LZ4 guarantees output never exceeds this.
pub fn max_compressed_size(input_len: usize) -> usize {
    // LZ4 formula: input_len + (input_len / 255) + 16
    input_len + (input_len / 255) + 16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let input = b"";
        let mut output = vec![0u8; max_compressed_size(input.len())];
        let compressed_len = compress_block(input, &mut output).unwrap();
        assert!(compressed_len <= output.len());
    }

    #[test]
    fn test_small_input() {
        let input = b"Hello";
        let mut output = vec![0u8; max_compressed_size(input.len())];
        let compressed_len = compress_block(input, &mut output).unwrap();

        let mut decompressed = vec![0u8; input.len()];
        let decompressed_len =
            decompress_block(&output[..compressed_len], &mut decompressed, input.len()).unwrap();

        assert_eq!(decompressed_len, input.len());
        assert_eq!(&decompressed[..], input);
    }

    #[test]
    fn test_repetitive_input() {
        let input = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // 40 A's
        let mut output = vec![0u8; max_compressed_size(input.len())];
        let compressed_len = compress_block(input, &mut output).unwrap();

        // Should compress well
        assert!(compressed_len < input.len());

        let mut decompressed = vec![0u8; input.len()];
        let decompressed_len =
            decompress_block(&output[..compressed_len], &mut decompressed, input.len()).unwrap();

        assert_eq!(decompressed_len, input.len());
        assert_eq!(&decompressed[..], input);
    }

    #[test]
    fn test_mixed_input() {
        let input = b"Hello, World! Hello, World! Hello, World! This is a test.";
        let mut output = vec![0u8; max_compressed_size(input.len())];
        let compressed_len = compress_block(input, &mut output).unwrap();

        let mut decompressed = vec![0u8; input.len()];
        let decompressed_len =
            decompress_block(&output[..compressed_len], &mut decompressed, input.len()).unwrap();

        assert_eq!(decompressed_len, input.len());
        assert_eq!(&decompressed[..], input);
    }

    #[test]
    fn test_large_input() {
        // Create input with repeating pattern
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = pattern.iter().cycle().take(10000).copied().collect();

        let mut output = vec![0u8; max_compressed_size(input.len())];
        let compressed_len = compress_block(&input, &mut output).unwrap();

        // Should compress well
        assert!(compressed_len < input.len());

        let mut decompressed = vec![0u8; input.len()];
        let decompressed_len =
            decompress_block(&output[..compressed_len], &mut decompressed, input.len()).unwrap();

        assert_eq!(decompressed_len, input.len());
        assert_eq!(decompressed, input);
    }

    /// Test interoperability: decompress lz4_flex output with our decompressor
    #[test]
    fn test_interop_decompress_lz4flex() {
        let input = b"Hello, LZ4! This is a test for interoperability with lz4_flex.";

        // Compress with lz4_flex
        let compressed = lz4_flex::compress(input);

        // Decompress with our implementation
        let mut decompressed = vec![0u8; input.len()];
        let len = decompress_block(&compressed, &mut decompressed, input.len()).unwrap();

        assert_eq!(len, input.len());
        assert_eq!(&decompressed[..], input);
    }

    /// Test interoperability: compress with us, decompress with lz4_flex
    #[test]
    fn test_interop_compress_for_lz4flex() {
        let input = b"Testing compression output compatibility with lz4_flex reference.";

        // Compress with our implementation
        let mut output = vec![0u8; max_compressed_size(input.len())];
        let compressed_len = compress_block(input, &mut output).unwrap();
        let compressed = &output[..compressed_len];

        // Decompress with lz4_flex
        let decompressed = lz4_flex::decompress(compressed, input.len()).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    /// Test interop with repetitive data (exercises match copying)
    #[test]
    fn test_interop_repetitive() {
        let pattern = b"ABCDEFGH";
        let input: Vec<u8> = pattern.iter().cycle().take(1000).copied().collect();

        // Our compress -> lz4_flex decompress
        let mut output = vec![0u8; max_compressed_size(input.len())];
        let compressed_len = compress_block(&input, &mut output).unwrap();
        let decompressed = lz4_flex::decompress(&output[..compressed_len], input.len()).unwrap();
        assert_eq!(decompressed, input);

        // lz4_flex compress -> our decompress
        let compressed = lz4_flex::compress(&input);
        let mut decompressed2 = vec![0u8; input.len()];
        let len = decompress_block(&compressed, &mut decompressed2, input.len()).unwrap();
        assert_eq!(len, input.len());
        assert_eq!(decompressed2, input);
    }
}
