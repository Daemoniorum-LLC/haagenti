//! Zstd block decoding.
//!
//! This module implements the decoding of Zstd compressed blocks,
//! including literals and sequences sections.
//!
//! ## Block Structure
//!
//! A compressed block contains:
//! 1. Literals Section - raw byte data
//! 2. Sequences Section - LZ77 commands (literal length, offset, match length)
//!
//! ## References
//!
//! - [RFC 8878 Section 3.1.1](https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1)

mod literals;
mod sequences;

pub use literals::{LiteralsSection, LiteralsBlockType};
pub use sequences::{
    SequencesSection, Sequence,
    LITERAL_LENGTH_BASELINE, MATCH_LENGTH_BASELINE,
};

use haagenti_core::{Error, Result};

/// Decode a raw block (uncompressed).
pub fn decode_raw_block(input: &[u8], output: &mut Vec<u8>) -> Result<()> {
    output.extend_from_slice(input);
    Ok(())
}

/// Decode an RLE block (single byte repeated).
pub fn decode_rle_block(input: &[u8], size: usize, output: &mut Vec<u8>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::corrupted("RLE block missing byte"));
    }
    let byte = input[0];
    output.reserve(size);
    for _ in 0..size {
        output.push(byte);
    }
    Ok(())
}

/// Decode a compressed block.
pub fn decode_compressed_block(
    input: &[u8],
    output: &mut Vec<u8>,
    window: &[u8],
) -> Result<()> {
    if input.is_empty() {
        return Err(Error::corrupted("Empty compressed block"));
    }

    // Parse literals section
    let (literals, literals_size) = LiteralsSection::parse(input)?;

    // Parse sequences section
    let sequences_data = &input[literals_size..];
    let sequences = SequencesSection::parse(sequences_data, &literals)?;

    // Execute sequences
    execute_sequences(&literals, &sequences.sequences, output, window)?;

    Ok(())
}

/// Execute decoded sequences to produce output.
fn execute_sequences(
    literals: &LiteralsSection,
    sequences: &[Sequence],
    output: &mut Vec<u8>,
    _window: &[u8],
) -> Result<()> {
    let literal_bytes = literals.data();
    let mut literal_pos = 0;

    for seq in sequences {
        // Copy literal_length bytes from literals
        let literal_end = literal_pos + seq.literal_length as usize;
        if literal_end > literal_bytes.len() {
            return Err(Error::corrupted("Literal length exceeds available literals"));
        }
        output.extend_from_slice(&literal_bytes[literal_pos..literal_end]);
        literal_pos = literal_end;

        // Copy match_length bytes from offset back in output
        if seq.match_length > 0 {
            let out_len = output.len();
            if seq.offset as usize > out_len {
                return Err(Error::corrupted(format!(
                    "Match offset {} exceeds output size {}",
                    seq.offset, out_len
                )));
            }

            let match_start = out_len - seq.offset as usize;

            // Handle overlapping matches (RLE-like patterns)
            for i in 0..seq.match_length as usize {
                let byte = output[match_start + (i % seq.offset as usize)];
                output.push(byte);
            }
        }
    }

    // Copy any remaining literals
    if literal_pos < literal_bytes.len() {
        output.extend_from_slice(&literal_bytes[literal_pos..]);
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_raw_block() {
        let input = b"Hello, World!";
        let mut output = Vec::new();
        decode_raw_block(input, &mut output).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_decode_rle_block() {
        let input = [b'A'];
        let mut output = Vec::new();
        decode_rle_block(&input, 5, &mut output).unwrap();
        assert_eq!(output, b"AAAAA");
    }

    #[test]
    fn test_decode_rle_block_empty_error() {
        let input = [];
        let mut output = Vec::new();
        let result = decode_rle_block(&input, 5, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_rle_block_large() {
        let input = [0xFF];
        let mut output = Vec::new();
        decode_rle_block(&input, 1000, &mut output).unwrap();
        assert_eq!(output.len(), 1000);
        assert!(output.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_execute_sequences_literals_only() {
        // No sequences, just literals
        let literals = LiteralsSection::new_raw(b"Hello".to_vec());
        let sequences = vec![];
        let mut output = Vec::new();

        execute_sequences(&literals, &sequences, &mut output, &[]).unwrap();
        assert_eq!(output, b"Hello");
    }

    #[test]
    fn test_execute_sequences_with_match() {
        // Literal "ab", then match (offset=2, length=4) to repeat "ab" twice
        let literals = LiteralsSection::new_raw(b"ab".to_vec());
        let sequences = vec![
            Sequence {
                literal_length: 2,
                offset: 2,
                match_length: 4,
            },
        ];
        let mut output = Vec::new();

        execute_sequences(&literals, &sequences, &mut output, &[]).unwrap();
        // Output: "ab" + "abab" = "ababab"
        assert_eq!(output, b"ababab");
    }

    #[test]
    fn test_execute_sequences_rle_pattern() {
        // Literal "a", then match (offset=1, length=4) to repeat "a" four times
        let literals = LiteralsSection::new_raw(b"a".to_vec());
        let sequences = vec![
            Sequence {
                literal_length: 1,
                offset: 1,
                match_length: 4,
            },
        ];
        let mut output = Vec::new();

        execute_sequences(&literals, &sequences, &mut output, &[]).unwrap();
        // Output: "a" + "aaaa" = "aaaaa"
        assert_eq!(output, b"aaaaa");
    }

    #[test]
    fn test_execute_sequences_multiple() {
        // Two sequences
        let literals = LiteralsSection::new_raw(b"abcXYZ".to_vec());
        let sequences = vec![
            Sequence {
                literal_length: 3, // "abc"
                offset: 3,
                match_length: 3,   // copy "abc"
            },
            Sequence {
                literal_length: 3, // "XYZ"
                offset: 0,
                match_length: 0,
            },
        ];
        let mut output = Vec::new();

        execute_sequences(&literals, &sequences, &mut output, &[]).unwrap();
        // Output: "abc" + "abc" + "XYZ" = "abcabcXYZ"
        assert_eq!(output, b"abcabcXYZ");
    }
}
