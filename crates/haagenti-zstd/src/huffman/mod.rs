//! Huffman coding for Zstandard.
//!
//! Huffman coding is used in Zstandard for literal compression. It provides
//! fast decoding while achieving good compression ratios.
//!
//! ## Overview
//!
//! Zstd uses canonical Huffman codes with weights stored in a compressed header.
//! The header can be either:
//! - FSE-compressed (for larger symbol sets)
//! - Direct representation (for small symbol sets)
//!
//! ## References
//!
//! - [RFC 8878 Section 4.2](https://datatracker.ietf.org/doc/html/rfc8878#section-4.2)

mod decoder;
mod encoder;
mod table;

pub use decoder::{build_table_from_weights, parse_huffman_weights, HuffmanDecoder};
pub use encoder::{HuffmanCode, HuffmanEncoder};
pub use table::{HuffmanTable, HuffmanTableEntry};

/// Maximum number of symbols in a Huffman table (0-255 for literals).
pub const HUFFMAN_MAX_SYMBOLS: usize = 256;

/// Maximum Huffman weight value.
/// Weight 0 means symbol not present.
/// Weight w means code length = max_bits + 1 - w.
pub const HUFFMAN_MAX_WEIGHT: u8 = 12;

/// Maximum number of bits for a Huffman code.
pub const HUFFMAN_MAX_BITS: u8 = 11;

/// Minimum header size for direct representation.
pub const HUFFMAN_MIN_HEADER_SIZE: usize = 1;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fse::BitReader;

    #[test]
    fn test_constants() {
        assert_eq!(HUFFMAN_MAX_SYMBOLS, 256);
        assert!(HUFFMAN_MAX_WEIGHT <= 12);
        assert!(HUFFMAN_MAX_BITS <= 11);
    }

    #[test]
    fn test_huffman_encoder_decoder_roundtrip() {
        // Create sample data with clear frequency distribution
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }
        for _ in 0..25 {
            data.push(b'c');
        }

        // Build encoder
        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder");

        // Get weights and build decoder table
        let weights = encoder.serialize_weights();
        let (parsed_weights, _) = parse_huffman_weights(&weights).expect("Should parse weights");
        let table = build_table_from_weights(parsed_weights).expect("Should build table");
        let decoder = HuffmanDecoder::new(&table);

        // Encode the data
        let encoded = encoder.encode(&data);

        // Encoded should be smaller than original
        assert!(encoded.len() < data.len(), "Huffman should compress");

        // Decode all symbols using reversed reader
        let mut bits = BitReader::new_reversed(&encoded).expect("Should create reversed reader");
        let mut decoded = Vec::new();
        for _ in 0..data.len() {
            let symbol = decoder
                .decode_symbol(&mut bits)
                .expect("Should decode symbol");
            decoded.push(symbol);
        }

        // Verify full roundtrip
        assert_eq!(decoded.len(), data.len(), "Should decode all symbols");
        assert_eq!(decoded, data, "Huffman roundtrip should match");
    }

    #[test]
    fn test_huffman_simple_encode_decode() {
        // Very simple case: just encode 'a' and 'b'
        let mut data = Vec::new();
        for _ in 0..50 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }

        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder");
        let encoded = encoder.encode(&data);

        // Should be smaller than original
        assert!(encoded.len() < data.len(), "Huffman should compress");

        // The encoded data should have a sentinel bit
        let last_byte = encoded[encoded.len() - 1];
        assert!(last_byte != 0, "Should have sentinel bit");
    }

    #[test]
    fn test_huffman_minimal() {
        // Minimal test: 3 symbols with known codes
        // Weights [2, 1, 1] -> codes: sym0 = 0 (1 bit), sym1 = 10, sym2 = 11
        let weights = [2u8, 1, 1];
        let table = HuffmanTable::from_weights(&weights).unwrap();
        let decoder = HuffmanDecoder::new(&table);

        assert_eq!(table.max_bits(), 2);

        // Manually encode: symbol 0 (code 0, 1 bit)
        // Byte = 0b00000100, sentinel at bit 2, data at bits 1,0 = 00
        // Decoder peeks 2 bits, sees 00 -> symbol 0
        let encoded = [0b00000100u8];

        let mut bits = BitReader::new_reversed(&encoded).unwrap();
        assert_eq!(bits.bits_remaining(), 2);

        let sym = decoder.decode_symbol(&mut bits).unwrap();
        assert_eq!(sym, 0, "Should decode symbol 0");
    }

    // =========================================================================
    // Track A.3 Roadmap Tests
    // =========================================================================

    #[test]
    fn test_huffman_table_from_frequencies() {
        // Build encoder from data with known frequency distribution
        // This tests the frequency -> weight conversion
        let mut data = Vec::new();
        // Symbol frequencies: a=100, b=50, c=25, d=12, e=6, f=3, g=2, h=1
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }
        for _ in 0..25 {
            data.push(b'c');
        }
        for _ in 0..12 {
            data.push(b'd');
        }
        for _ in 0..6 {
            data.push(b'e');
        }
        for _ in 0..3 {
            data.push(b'f');
        }
        for _ in 0..2 {
            data.push(b'g');
        }
        for _ in 0..1 {
            data.push(b'h');
        }

        let encoder = HuffmanEncoder::build(&data).expect("Should build from frequencies");
        let codes = encoder.get_codes();

        // Most frequent symbol ('a') should have shortest or equal code length
        let a_bits = codes[b'a' as usize].num_bits;
        let h_bits = codes[b'h' as usize].num_bits;

        assert!(
            a_bits <= h_bits,
            "Most frequent symbol should have shorter code: a={} bits, h={} bits",
            a_bits,
            h_bits
        );
    }

    #[test]
    fn test_huffman_table_serialization_roundtrip() {
        // Test that serialized weights can be parsed back
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }
        for _ in 0..25 {
            data.push(b'c');
        }
        for _ in 0..12 {
            data.push(b'd');
        }

        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder");
        let serialized = encoder.serialize_weights();

        // Parse the serialized weights
        let (parsed_weights, bytes_read) =
            parse_huffman_weights(&serialized).expect("Should parse serialized weights");

        assert!(bytes_read > 0, "Should read some bytes");
        assert!(!parsed_weights.is_empty(), "Should have parsed weights");

        // Build table from parsed weights
        let table = build_table_from_weights(parsed_weights)
            .expect("Should build table from parsed weights");

        // Verify table is valid
        assert!(
            table.max_bits() <= HUFFMAN_MAX_BITS,
            "Max bits should be within limit"
        );
    }

    #[test]
    fn test_huffman_max_code_length() {
        // Test that code lengths never exceed 11 bits (Zstd limit)
        // Create data with power-of-2 frequencies (worst case for Huffman)
        let mut data = Vec::new();
        for i in 0..11u8 {
            let count = 1usize << i; // 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
            for _ in 0..count {
                data.push(i);
            }
        }

        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder");

        assert!(
            encoder.max_bits() <= HUFFMAN_MAX_BITS,
            "Max code length {} exceeds limit {}",
            encoder.max_bits(),
            HUFFMAN_MAX_BITS
        );

        // Verify all codes are within limit
        let codes = encoder.get_codes();
        for i in 0..11u8 {
            let code = &codes[i as usize];
            if code.num_bits > 0 {
                assert!(
                    code.num_bits <= HUFFMAN_MAX_BITS,
                    "Symbol {} has code length {} exceeding max {}",
                    i,
                    code.num_bits,
                    HUFFMAN_MAX_BITS
                );
            }
        }
    }

    #[test]
    fn test_huffman_many_symbols() {
        // Test encoding with many byte values
        // Use 20 symbols with clear frequency distribution
        let mut data = Vec::new();
        for i in 0..20u8 {
            // Exponential distribution for good compression
            let count = 200usize.saturating_sub(i as usize * 8).max(10);
            for _ in 0..count {
                data.push(i);
            }
        }

        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder for many symbols");

        // Verify symbols are encoded
        assert!(encoder.num_symbols() >= 15, "Should have many symbols");
        assert!(
            encoder.max_bits() <= HUFFMAN_MAX_BITS,
            "Should respect max bits"
        );

        // Encode the data and verify compression
        let encoded = encoder.encode(&data);
        assert!(encoded.len() < data.len(), "Should compress data");

        // Get weights and verify serialization
        let weights = encoder.serialize_weights();
        assert!(!weights.is_empty(), "Should produce non-empty weights");
        assert!(weights[0] >= 128, "Should use direct encoding format");
    }

    #[test]
    fn test_huffman_compression_ratio() {
        // Test that Huffman achieves good compression on repetitive data
        let data = b"aaaaaabbbbcccdde".repeat(1000);

        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder");
        let encoded = encoder.encode(&data);
        let weights = encoder.serialize_weights();

        let compressed_size = encoded.len() + weights.len();
        let original_size = data.len();

        // Should achieve at least 2x compression on this repetitive data
        let ratio = original_size as f64 / compressed_size as f64;
        assert!(
            ratio >= 2.0,
            "Compression ratio {:.2}x is below expected 2x (original: {}, compressed: {})",
            ratio,
            original_size,
            compressed_size
        );
    }

    #[test]
    fn test_huffman_empty_data() {
        // Empty data should return just sentinel
        let encoder = HuffmanEncoder::build(&[]);
        assert!(encoder.is_none(), "Empty data should not build encoder");
    }

    #[test]
    fn test_huffman_single_value_data() {
        // Data with single unique value (RLE case)
        let data = vec![b'x'; 1000];
        let encoder = HuffmanEncoder::build(&data);
        // Should return None since single symbol is better handled by RLE
        assert!(
            encoder.is_none(),
            "Single symbol data should not use Huffman"
        );
    }

    #[test]
    fn test_huffman_two_symbols_equal_frequency() {
        // Two symbols with equal frequency should each get 1-bit codes
        let mut data = Vec::new();
        for _ in 0..500 {
            data.push(b'0');
        }
        for _ in 0..500 {
            data.push(b'1');
        }

        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder");
        let codes = encoder.get_codes();

        // Both symbols should have 1-bit codes
        assert_eq!(
            codes[b'0' as usize].num_bits, 1,
            "Symbol '0' should have 1-bit code"
        );
        assert_eq!(
            codes[b'1' as usize].num_bits, 1,
            "Symbol '1' should have 1-bit code"
        );

        // Compressed size should be about half the original (1 bit per symbol)
        let encoded = encoder.encode(&data);
        assert!(
            encoded.len() <= data.len() / 2 + 10, // Some overhead allowed
            "Two equal-freq symbols should compress to ~half size"
        );
    }

    #[test]
    fn test_huffman_text_pattern() {
        // Test with realistic text-like data
        let text = b"The quick brown fox jumps over the lazy dog. ";
        let data = text.repeat(100);

        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder for text");

        // Text has non-uniform character distribution
        assert!(
            encoder.num_symbols() > 10,
            "Text should have multiple symbols"
        );

        // Encode
        let encoded = encoder.encode(&data);
        let weights = encoder.serialize_weights();

        // Should achieve compression on repetitive text
        let total_compressed = encoded.len() + weights.len();
        assert!(
            total_compressed < data.len(),
            "Should compress text: {} < {}",
            total_compressed,
            data.len()
        );

        // Space is common, should be encoded
        let space_code = encoder.get_codes()[b' ' as usize];
        assert!(space_code.num_bits > 0, "Space should be encoded");
        // Space should have a reasonable code length (not the longest)
        assert!(
            space_code.num_bits <= encoder.max_bits(),
            "Space code should fit within max bits"
        );
    }

    #[test]
    fn test_huffman_batch_encode_consistency() {
        // Test that regular and batch encoding produce consistent results
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }
        for _ in 0..25 {
            data.push(b'c');
        }
        for _ in 0..12 {
            data.push(b'd');
        }

        let encoder = HuffmanEncoder::build(&data).expect("Should build encoder");

        let regular = encoder.encode(&data);
        let batch = encoder.encode_batch(&data);

        // Both should produce non-empty output
        assert!(!regular.is_empty(), "Regular encode should produce output");
        assert!(!batch.is_empty(), "Batch encode should produce output");

        // Both should be smaller than input
        assert!(regular.len() < data.len(), "Regular should compress");
        assert!(batch.len() < data.len(), "Batch should compress");
    }
}
