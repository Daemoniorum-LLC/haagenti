//! Literals section decoding.
//!
//! The literals section contains raw byte data that is copied to the output.

use crate::fse::BitReader;
use crate::huffman::{build_table_from_weights, parse_huffman_weights, HuffmanDecoder};
use haagenti_core::{Error, Result};

/// Literals block type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralsBlockType {
    /// Raw literals - uncompressed bytes.
    Raw,
    /// RLE literals - single byte repeated.
    Rle,
    /// Huffman compressed literals with new tree.
    Compressed,
    /// Huffman compressed using previous tree.
    Treeless,
}

impl LiteralsBlockType {
    /// Parse block type from 2-bit field.
    pub fn from_field(field: u8) -> Self {
        match field {
            0 => LiteralsBlockType::Raw,
            1 => LiteralsBlockType::Rle,
            2 => LiteralsBlockType::Compressed,
            3 => LiteralsBlockType::Treeless,
            _ => unreachable!(),
        }
    }
}

/// Parsed literals section.
#[derive(Debug, Clone)]
pub struct LiteralsSection {
    /// Block type.
    pub block_type: LiteralsBlockType,
    /// Regenerated (uncompressed) size.
    pub regenerated_size: usize,
    /// Compressed size (for compressed modes).
    pub compressed_size: usize,
    /// The literal data.
    data: Vec<u8>,
}

impl LiteralsSection {
    /// Create a new raw literals section for testing.
    pub fn new_raw(data: Vec<u8>) -> Self {
        let size = data.len();
        Self {
            block_type: LiteralsBlockType::Raw,
            regenerated_size: size,
            compressed_size: size,
            data,
        }
    }

    /// Parse a literals section from input.
    ///
    /// Returns the parsed section and the number of bytes consumed.
    pub fn parse(input: &[u8]) -> Result<(Self, usize)> {
        if input.is_empty() {
            return Err(Error::corrupted("Empty literals section"));
        }

        let header_byte = input[0];
        let block_type = LiteralsBlockType::from_field(header_byte & 0x03);
        let size_format = (header_byte >> 2) & 0x03;

        match block_type {
            LiteralsBlockType::Raw | LiteralsBlockType::Rle => {
                Self::parse_raw_rle(input, block_type, size_format)
            }
            LiteralsBlockType::Compressed | LiteralsBlockType::Treeless => {
                Self::parse_compressed(input, block_type, size_format)
            }
        }
    }

    /// Parse raw or RLE literals.
    fn parse_raw_rle(
        input: &[u8],
        block_type: LiteralsBlockType,
        size_format: u8,
    ) -> Result<(Self, usize)> {
        let (regenerated_size, header_size) = match size_format {
            // Size_Format = 0b00 or 0b10: 5-bit size
            0 | 2 => {
                let size = (input[0] >> 3) as usize;
                (size, 1)
            }
            // Size_Format = 0b01: 12-bit size
            1 => {
                if input.len() < 2 {
                    return Err(Error::corrupted("Literals header truncated"));
                }
                let size = ((input[0] >> 4) as usize) | ((input[1] as usize) << 4);
                (size, 2)
            }
            // Size_Format = 0b11: 20-bit size
            3 => {
                if input.len() < 3 {
                    return Err(Error::corrupted("Literals header truncated"));
                }
                let size = ((input[0] >> 4) as usize)
                    | ((input[1] as usize) << 4)
                    | ((input[2] as usize) << 12);
                (size, 3)
            }
            _ => unreachable!(),
        };

        let data_start = header_size;
        let data = match block_type {
            LiteralsBlockType::Raw => {
                if input.len() < data_start + regenerated_size {
                    return Err(Error::corrupted("Raw literals truncated"));
                }
                input[data_start..data_start + regenerated_size].to_vec()
            }
            LiteralsBlockType::Rle => {
                if input.len() < data_start + 1 {
                    return Err(Error::corrupted("RLE literals missing byte"));
                }
                vec![input[data_start]; regenerated_size]
            }
            _ => unreachable!(),
        };

        let total_size = match block_type {
            LiteralsBlockType::Raw => header_size + regenerated_size,
            LiteralsBlockType::Rle => header_size + 1,
            _ => unreachable!(),
        };

        Ok((
            Self {
                block_type,
                regenerated_size,
                compressed_size: match block_type {
                    LiteralsBlockType::Raw => regenerated_size,
                    LiteralsBlockType::Rle => 1,
                    _ => unreachable!(),
                },
                data,
            },
            total_size,
        ))
    }

    /// Parse compressed literals (Huffman).
    fn parse_compressed(
        input: &[u8],
        block_type: LiteralsBlockType,
        size_format: u8,
    ) -> Result<(Self, usize)> {
        // Determine stream count and parse sizes
        let is_single_stream = size_format == 3;

        // Parse sizes based on format
        let (regenerated_size, compressed_size, header_size) = match size_format {
            // 4 streams, 10-bit sizes (3-byte header)
            // RFC 8878: regen[3:0] = byte0[7:4], regen[9:4] = byte1[5:0]
            //           comp[1:0] = byte1[7:6], comp[9:2] = byte2[7:0]
            0 => {
                if input.len() < 3 {
                    return Err(Error::corrupted("Compressed literals header truncated"));
                }
                let regen = ((input[0] >> 4) as usize) | (((input[1] & 0x3F) as usize) << 4);
                let comp = ((input[1] >> 6) as usize) | ((input[2] as usize) << 2);
                (regen, comp, 3)
            }
            // 4 streams, 14-bit regen size, 10-bit comp size (4-byte header)
            // RFC 8878: byte0[7:4]=regen[3:0], byte1=regen[11:4], byte2[1:0]=regen[13:12]
            //           byte2[7:2]=comp[5:0], byte3=comp[9:2]? No...
            // Actually: byte2[7:6]=comp[1:0], byte3=comp[9:2]
            1 => {
                if input.len() < 4 {
                    return Err(Error::corrupted("Compressed literals header truncated"));
                }
                let regen = ((input[0] >> 4) as usize)
                    | ((input[1] as usize) << 4)
                    | (((input[2] & 0x03) as usize) << 12);
                let comp = ((input[2] >> 6) as usize) | ((input[3] as usize) << 2);
                (regen, comp, 4)
            }
            // 4 streams, 18-bit sizes
            2 => {
                if input.len() < 5 {
                    return Err(Error::corrupted("Compressed literals header truncated"));
                }
                let regen = (((input[0] >> 4) & 0x3F) as usize)
                    | ((input[1] as usize) << 4)
                    | (((input[2] & 0x0F) as usize) << 12);
                let comp = ((input[2] >> 4) as usize)
                    | ((input[3] as usize) << 4)
                    | (((input[4] & 0x03) as usize) << 12);
                (regen, comp, 5)
            }
            // 1 stream, 10-bit sizes (3-byte header, single stream)
            // Same format as Size_Format=0 but single stream instead of 4
            3 => {
                if input.len() < 3 {
                    return Err(Error::corrupted("Compressed literals header truncated"));
                }
                let regen = ((input[0] >> 4) as usize) | (((input[1] & 0x3F) as usize) << 4);
                let comp = ((input[1] >> 6) as usize) | ((input[2] as usize) << 2);
                (regen, comp, 3)
            }
            _ => unreachable!(),
        };

        if input.len() < header_size + compressed_size {
            return Err(Error::corrupted("Compressed literals data truncated"));
        }

        let compressed_data = &input[header_size..header_size + compressed_size];

        // For treeless mode, we'd need a previously stored Huffman table
        if block_type == LiteralsBlockType::Treeless {
            return Err(Error::Unsupported(
                "Treeless Huffman literals require previous table state".into(),
            ));
        }

        // Decode Huffman-compressed literals
        let data =
            Self::decode_huffman_literals(compressed_data, regenerated_size, is_single_stream)?;

        let total_size = header_size + compressed_size;

        Ok((
            Self {
                block_type,
                regenerated_size,
                compressed_size,
                data,
            },
            total_size,
        ))
    }

    /// Decode Huffman-compressed literals.
    fn decode_huffman_literals(
        data: &[u8],
        regenerated_size: usize,
        is_single_stream: bool,
    ) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Err(Error::corrupted("Empty Huffman literals data"));
        }

        // Parse Huffman weights from the beginning of data
        let (weights, weights_consumed) = parse_huffman_weights(data)?;

        // Build Huffman table
        let table = build_table_from_weights(weights)?;
        let decoder = HuffmanDecoder::new(&table);

        let stream_data = &data[weights_consumed..];

        if is_single_stream {
            Self::decode_single_stream(&decoder, stream_data, regenerated_size)
        } else {
            Self::decode_four_streams(&decoder, stream_data, regenerated_size)
        }
    }

    /// Decode a single Huffman stream.
    fn decode_single_stream(
        decoder: &HuffmanDecoder,
        data: &[u8],
        regenerated_size: usize,
    ) -> Result<Vec<u8>> {
        if data.is_empty() {
            if regenerated_size == 0 {
                return Ok(Vec::new());
            }
            return Err(Error::corrupted("Empty stream data for Huffman decoding"));
        }

        // Huffman streams are read backwards (from end to start)
        let mut output = Vec::with_capacity(regenerated_size);
        let mut bits = BitReader::new_reversed(data)?;

        for _ in 0..regenerated_size {
            let symbol = decoder.decode_symbol(&mut bits)?;
            output.push(symbol);
        }

        Ok(output)
    }

    /// Decode four parallel Huffman streams.
    fn decode_four_streams(
        decoder: &HuffmanDecoder,
        data: &[u8],
        regenerated_size: usize,
    ) -> Result<Vec<u8>> {
        // 4-stream format has 6-byte header with stream sizes
        if data.len() < 6 {
            return Err(Error::corrupted("4-stream header too short"));
        }

        // Read jump table: 3 x 2-byte offsets (little-endian)
        let jump1 = u16::from_le_bytes([data[0], data[1]]) as usize;
        let jump2 = u16::from_le_bytes([data[2], data[3]]) as usize;
        let jump3 = u16::from_le_bytes([data[4], data[5]]) as usize;

        // Stream boundaries
        let stream1_start = 6;
        let stream2_start = 6 + jump1;
        let stream3_start = 6 + jump2;
        let stream4_start = 6 + jump3;
        let stream4_end = data.len();

        // Validate boundaries
        if stream2_start > data.len() || stream3_start > data.len() || stream4_start > data.len() {
            return Err(Error::corrupted(
                "Invalid stream boundaries in 4-stream literals",
            ));
        }

        // Calculate output size per stream (regenerated_size split into 4)
        let base_size = regenerated_size / 4;
        let remainder = regenerated_size % 4;

        let sizes = [
            base_size + if remainder > 0 { 1 } else { 0 },
            base_size + if remainder > 1 { 1 } else { 0 },
            base_size + if remainder > 2 { 1 } else { 0 },
            base_size,
        ];

        let stream_ranges = [
            (stream1_start, stream2_start),
            (stream2_start, stream3_start),
            (stream3_start, stream4_start),
            (stream4_start, stream4_end),
        ];

        let mut output = Vec::with_capacity(regenerated_size);

        // Decode each stream
        for (i, &(start, end)) in stream_ranges.iter().enumerate() {
            if start >= end {
                // Empty stream
                if sizes[i] > 0 {
                    return Err(Error::corrupted(format!(
                        "Stream {} is empty but expects {} symbols",
                        i, sizes[i]
                    )));
                }
                continue;
            }

            let stream_data = &data[start..end];
            let stream_output = Self::decode_single_stream(decoder, stream_data, sizes[i])?;
            output.extend(stream_output);
        }

        Ok(output)
    }

    /// Get the literal data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literals_block_type_parsing() {
        assert_eq!(LiteralsBlockType::from_field(0), LiteralsBlockType::Raw);
        assert_eq!(LiteralsBlockType::from_field(1), LiteralsBlockType::Rle);
        assert_eq!(
            LiteralsBlockType::from_field(2),
            LiteralsBlockType::Compressed
        );
        assert_eq!(
            LiteralsBlockType::from_field(3),
            LiteralsBlockType::Treeless
        );
    }

    #[test]
    fn test_raw_literals_5bit_size() {
        // Raw, size_format=0, size=5 (5 << 3 = 40, type=0 -> 0b00101000 = 0x28)
        // Actually: header byte = (size << 3) | (size_format << 2) | type
        // size=5: (5 << 3) | (0 << 2) | 0 = 0x28
        let mut input = vec![0x28]; // size=5, format=0, type=Raw
        input.extend_from_slice(b"Hello");

        let (section, consumed) = LiteralsSection::parse(&input).unwrap();
        assert_eq!(section.block_type, LiteralsBlockType::Raw);
        assert_eq!(section.regenerated_size, 5);
        assert_eq!(section.data, b"Hello");
        assert_eq!(consumed, 6); // 1 header + 5 data
    }

    #[test]
    fn test_rle_literals_5bit_size() {
        // RLE, size_format=0, size=10
        // header = (10 << 3) | (0 << 2) | 1 = 0x51
        let input = vec![0x51, b'X']; // size=10, format=0, type=RLE, byte='X'

        let (section, consumed) = LiteralsSection::parse(&input).unwrap();
        assert_eq!(section.block_type, LiteralsBlockType::Rle);
        assert_eq!(section.regenerated_size, 10);
        assert_eq!(section.data, vec![b'X'; 10]);
        assert_eq!(consumed, 2); // 1 header + 1 byte
    }

    #[test]
    fn test_raw_literals_12bit_size() {
        // Raw, size_format=1, size=256
        // byte0: (size_low << 4) | (1 << 2) | 0
        // size_low = size & 0x0F = 0
        // size_high = size >> 4 = 16
        // byte0 = (0 << 4) | (1 << 2) | 0 = 0x04
        // byte1 = size_high = 16
        let mut input = vec![0x04, 0x10]; // size=256
        input.resize(2 + 256, b'A');

        let (section, consumed) = LiteralsSection::parse(&input).unwrap();
        assert_eq!(section.block_type, LiteralsBlockType::Raw);
        assert_eq!(section.regenerated_size, 256);
        assert_eq!(consumed, 2 + 256);
    }

    #[test]
    fn test_empty_input_error() {
        let result = LiteralsSection::parse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_raw_error() {
        // Raw, size=10, but only 5 bytes of data
        let input = vec![0x50, b'H', b'e', b'l', b'l', b'o'];
        let result = LiteralsSection::parse(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_raw_helper() {
        let section = LiteralsSection::new_raw(b"test".to_vec());
        assert_eq!(section.block_type, LiteralsBlockType::Raw);
        assert_eq!(section.regenerated_size, 4);
        assert_eq!(section.data(), b"test");
    }

    #[test]
    fn test_compressed_header_type_detection() {
        // Test that compressed literals type is detected correctly
        // Type=Compressed (2), any size format
        let header_byte = 0x0E; // Type=2 (Compressed), Size_Format=3
        let block_type = LiteralsBlockType::from_field(header_byte & 0x03);
        assert_eq!(block_type, LiteralsBlockType::Compressed);

        let header_byte = 0x02; // Type=2 (Compressed), Size_Format=0
        let block_type = LiteralsBlockType::from_field(header_byte & 0x03);
        assert_eq!(block_type, LiteralsBlockType::Compressed);
    }

    #[test]
    fn test_treeless_requires_previous_table() {
        // Treeless mode (type=3) should fail without previous table state
        // Construct minimal treeless header: type=3, size_format=3, regen=5, comp=10
        // byte0 = (5 << 4) | (3 << 2) | 3 = 0x5F
        // byte1 = ((5 >> 4) & 0x03) | ((10 & 0x3F) << 2) = 0 | 0x28 = 0x28
        // byte2 = (10 >> 6) = 0
        // Then add fake compressed data
        let mut input = vec![0x5F, 0x28, 0x00];
        input.extend(vec![0x80; 10]); // Fake compressed data with sentinel

        let result = LiteralsSection::parse(&input);

        // Should fail with "requires previous table" error
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("previous table") || msg.contains("Treeless"),
                "Expected 'previous table' or 'Treeless' error, got: {}",
                msg
            );
        }
    }

    #[test]
    fn test_compressed_literals_truncated_data_error() {
        // Compressed literals with data shorter than declared
        // Type=2, size_format=3 (single stream), regen=10, comp=20
        // But only provide 5 bytes of data
        let input = vec![0xA2, 0x50, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
        let result = LiteralsSection::parse(&input);

        // Should fail due to truncated data
        assert!(result.is_err());
    }

    #[test]
    fn test_size_format_detection() {
        // Verify size_format extraction from header
        for size_format in 0..4u8 {
            let header_byte = 0x02 | (size_format << 2); // Compressed type with various formats
            let extracted = (header_byte >> 2) & 0x03;
            assert_eq!(extracted, size_format);
        }
    }
}
