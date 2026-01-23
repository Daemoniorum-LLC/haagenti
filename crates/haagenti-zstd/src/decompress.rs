//! Full Zstd decompression pipeline.
//!
//! This module integrates all components to provide complete decompression.

use crate::block::{decode_raw_block, decode_rle_block, LiteralsSection, SequencesSection};
use crate::frame::{xxhash64, BlockHeader, BlockType, FrameHeader, ZSTD_MAGIC};
use haagenti_core::{Error, Result};

/// Decompression context holding state across blocks.
#[derive(Debug)]
pub struct DecompressContext {
    /// Output buffer (history window).
    output: Vec<u8>,
    /// Window size for back-references (used when history exceeds window).
    #[allow(dead_code)]
    window_size: usize,
    /// Repeat offsets (1-3).
    repeat_offsets: [u32; 3],
}

impl DecompressContext {
    /// Create a new decompression context.
    pub fn new(window_size: usize) -> Self {
        Self {
            output: Vec::with_capacity(window_size.min(1024 * 1024)),
            window_size,
            repeat_offsets: [1, 4, 8], // Default repeat offsets
        }
    }

    /// Get the decompressed output.
    pub fn output(&self) -> &[u8] {
        &self.output
    }

    /// Take ownership of the output.
    pub fn into_output(self) -> Vec<u8> {
        self.output
    }

    /// Update repeat offsets after a match.
    pub fn update_offsets(&mut self, offset: u32) {
        if offset != self.repeat_offsets[0] {
            self.repeat_offsets[2] = self.repeat_offsets[1];
            self.repeat_offsets[1] = self.repeat_offsets[0];
            self.repeat_offsets[0] = offset;
        }
    }

    /// Get a repeat offset by code (1, 2, or 3).
    pub fn get_repeat_offset(&self, code: u32) -> u32 {
        match code {
            1 => self.repeat_offsets[0],
            2 => self.repeat_offsets[1],
            3 => self.repeat_offsets[2],
            _ => code, // Not a repeat offset
        }
    }
}

/// Decompress a complete Zstd frame.
///
/// # Arguments
/// * `input` - The compressed data including magic number
///
/// # Returns
/// The decompressed data.
pub fn decompress_frame(input: &[u8]) -> Result<Vec<u8>> {
    // Validate minimum size
    if input.len() < 4 {
        return Err(Error::corrupted("Input too short for Zstd frame"));
    }

    // Validate magic number
    let magic = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if magic != ZSTD_MAGIC {
        return Err(Error::corrupted(format!(
            "Invalid Zstd magic: expected 0x{:08X}, got 0x{:08X}",
            ZSTD_MAGIC, magic
        )));
    }

    // Parse frame header
    let header = FrameHeader::parse(&input[4..])?;
    let mut ctx = DecompressContext::new(header.window_size);

    // Process blocks
    let mut pos = header.header_size;
    loop {
        if pos + BlockHeader::SIZE > input.len() {
            return Err(Error::corrupted("Frame truncated at block header"));
        }

        let block_header = BlockHeader::parse(&input[pos..])?;
        pos += BlockHeader::SIZE;

        let compressed_size = block_header.compressed_size();
        if pos + compressed_size > input.len() {
            return Err(Error::corrupted("Frame truncated at block data"));
        }

        let block_data = &input[pos..pos + compressed_size];
        pos += compressed_size;

        // Decode block
        match block_header.block_type {
            BlockType::Raw => {
                decode_raw_block(block_data, &mut ctx.output)?;
            }
            BlockType::Rle => {
                decode_rle_block(
                    block_data,
                    block_header.decompressed_size(),
                    &mut ctx.output,
                )?;
            }
            BlockType::Compressed => {
                decode_compressed_block(block_data, &mut ctx)?;
            }
            BlockType::Reserved => {
                return Err(Error::corrupted("Reserved block type"));
            }
        }

        if block_header.last_block {
            break;
        }
    }

    // Verify checksum if present
    if header.has_checksum {
        if pos + 4 > input.len() {
            return Err(Error::corrupted("Frame truncated at checksum"));
        }
        let expected =
            u32::from_le_bytes([input[pos], input[pos + 1], input[pos + 2], input[pos + 3]]);
        let actual = (xxhash64(&ctx.output, 0) & 0xFFFFFFFF) as u32;

        if expected != actual {
            return Err(Error::corrupted(format!(
                "Checksum mismatch: expected 0x{:08X}, got 0x{:08X}",
                expected, actual
            )));
        }
    }

    // Verify content size if specified
    if let Some(expected_size) = header.frame_content_size {
        if ctx.output.len() as u64 != expected_size {
            return Err(Error::corrupted(format!(
                "Content size mismatch: expected {}, got {}",
                expected_size,
                ctx.output.len()
            )));
        }
    }

    Ok(ctx.into_output())
}

/// Decompress a Zstd frame with dictionary support.
///
/// # Arguments
/// * `input` - The compressed data including magic number
/// * `dict` - Optional dictionary for decompression
///
/// # Returns
/// The decompressed data.
pub fn decompress_frame_with_dict(
    input: &[u8],
    dict: Option<&crate::dictionary::ZstdDictionary>,
) -> Result<Vec<u8>> {
    // For now, dictionary support is partial - we verify the ID and use
    // dictionary content as initial window. Full dictionary decompression
    // would require using dictionary's Huffman/FSE tables.

    if dict.is_none() {
        return decompress_frame(input);
    }

    let dictionary = dict.unwrap();

    // Validate minimum size
    if input.len() < 4 {
        return Err(Error::corrupted("Input too short for Zstd frame"));
    }

    // Validate magic number
    let magic = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if magic != ZSTD_MAGIC {
        return Err(Error::corrupted(format!(
            "Invalid Zstd magic: expected 0x{:08X}, got 0x{:08X}",
            ZSTD_MAGIC, magic
        )));
    }

    // Parse frame header
    let header = FrameHeader::parse(&input[4..])?;
    let mut ctx = DecompressContext::new(header.window_size);

    // Pre-fill context with dictionary content for back-references
    ctx.output.extend_from_slice(dictionary.content());
    let dict_len = dictionary.content().len();

    // Process blocks
    let mut pos = header.header_size;
    loop {
        if pos + BlockHeader::SIZE > input.len() {
            return Err(Error::corrupted("Frame truncated at block header"));
        }

        let block_header = BlockHeader::parse(&input[pos..])?;
        pos += BlockHeader::SIZE;

        let compressed_size = block_header.compressed_size();
        if pos + compressed_size > input.len() {
            return Err(Error::corrupted("Frame truncated at block data"));
        }

        let block_data = &input[pos..pos + compressed_size];
        pos += compressed_size;

        // Decode block
        match block_header.block_type {
            BlockType::Raw => {
                decode_raw_block(block_data, &mut ctx.output)?;
            }
            BlockType::Rle => {
                decode_rle_block(
                    block_data,
                    block_header.decompressed_size(),
                    &mut ctx.output,
                )?;
            }
            BlockType::Compressed => {
                decode_compressed_block(block_data, &mut ctx)?;
            }
            BlockType::Reserved => {
                return Err(Error::corrupted("Reserved block type"));
            }
        }

        if block_header.last_block {
            break;
        }
    }

    // Verify checksum if present (on content without dictionary prefix)
    if header.has_checksum {
        if pos + 4 > input.len() {
            return Err(Error::corrupted("Frame truncated at checksum"));
        }
        let expected =
            u32::from_le_bytes([input[pos], input[pos + 1], input[pos + 2], input[pos + 3]]);
        // Checksum is computed on the actual decompressed content (without dict prefix)
        let content = &ctx.output[dict_len..];
        let actual = (xxhash64(content, 0) & 0xFFFFFFFF) as u32;

        if expected != actual {
            return Err(Error::corrupted(format!(
                "Checksum mismatch: expected 0x{:08X}, got 0x{:08X}",
                expected, actual
            )));
        }
    }

    // Verify content size if specified
    if let Some(expected_size) = header.frame_content_size {
        let actual_size = (ctx.output.len() - dict_len) as u64;
        if actual_size != expected_size {
            return Err(Error::corrupted(format!(
                "Content size mismatch: expected {}, got {}",
                expected_size, actual_size
            )));
        }
    }

    // Return only the actual decompressed content (without dict prefix)
    Ok(ctx.output[dict_len..].to_vec())
}

/// Decode a compressed block.
fn decode_compressed_block(input: &[u8], ctx: &mut DecompressContext) -> Result<()> {
    if input.is_empty() {
        return Err(Error::corrupted("Empty compressed block"));
    }

    // Parse literals section
    let (literals, literals_consumed) = LiteralsSection::parse(input)?;

    // Parse sequences section
    let sequences_data = &input[literals_consumed..];
    let sequences = SequencesSection::parse(sequences_data, &literals)?;

    // Execute sequences
    execute_sequences(&literals, &sequences, ctx)?;

    Ok(())
}

/// Execute decoded sequences to produce output.
///
/// Note: The sequences already have actual offsets resolved by the sequence decoder
/// (SequencesSection::parse handles repeat offset logic internally).
fn execute_sequences(
    literals: &LiteralsSection,
    sequences: &SequencesSection,
    ctx: &mut DecompressContext,
) -> Result<()> {
    let literal_bytes = literals.data();
    let mut literal_pos = 0;

    // Pre-reserve capacity to avoid reallocations
    let total_output: usize = sequences
        .sequences
        .iter()
        .map(|s| s.literal_length as usize + s.match_length as usize)
        .sum();
    ctx.output
        .reserve(total_output + literal_bytes.len() - literal_pos);

    for seq in &sequences.sequences {
        // Copy literal_length bytes from literals
        let literal_end = literal_pos + seq.literal_length as usize;
        if literal_end > literal_bytes.len() {
            return Err(Error::corrupted(
                "Literal length exceeds available literals",
            ));
        }
        ctx.output
            .extend_from_slice(&literal_bytes[literal_pos..literal_end]);
        literal_pos = literal_end;

        // Offset is already resolved to actual byte offset by sequence decoder
        // (repeat offset handling is done in SequencesSection::parse)
        let offset = seq.offset as usize;
        let match_length = seq.match_length as usize;

        // Copy match_length bytes from offset back in output
        if match_length > 0 && offset > 0 {
            let out_len = ctx.output.len();
            if offset > out_len {
                return Err(Error::corrupted(format!(
                    "Match offset {} exceeds output size {}",
                    offset, out_len
                )));
            }

            let match_start = out_len - offset;

            // Fast path: non-overlapping copy (offset >= match_length)
            if offset >= match_length {
                // Safe to use extend_from_within for non-overlapping
                ctx.output
                    .extend_from_within(match_start..match_start + match_length);
            } else {
                // Overlapping copy - need special handling
                copy_match_overlapping(&mut ctx.output, match_start, offset, match_length);
            }
        }
    }

    // Copy any remaining literals
    if literal_pos < literal_bytes.len() {
        ctx.output.extend_from_slice(&literal_bytes[literal_pos..]);
    }

    Ok(())
}

/// Fast overlapping match copy.
///
/// When offset < match_length, the source and destination overlap.
/// This handles the RLE-like pattern efficiently.
#[inline(always)]
fn copy_match_overlapping(
    output: &mut Vec<u8>,
    match_start: usize,
    offset: usize,
    match_length: usize,
) {
    // Reserve space
    output.reserve(match_length);
    let out_len = output.len();

    // SAFETY: We've reserved space and will write exactly match_length bytes
    unsafe {
        output.set_len(out_len + match_length);
        let dst = output.as_mut_ptr().add(out_len);
        let src_base = output.as_ptr().add(match_start);

        match offset {
            1 => {
                // RLE: single byte repeated
                let byte = *src_base;
                core::ptr::write_bytes(dst, byte, match_length);
            }
            2 => {
                // 2-byte pattern
                let pattern = core::ptr::read_unaligned(src_base as *const u16);
                let mut i = 0;
                while i + 2 <= match_length {
                    core::ptr::write_unaligned(dst.add(i) as *mut u16, pattern);
                    i += 2;
                }
                if i < match_length {
                    *dst.add(i) = *src_base;
                }
            }
            3 => {
                // 3-byte pattern - copy byte by byte for simplicity
                for i in 0..match_length {
                    *dst.add(i) = *src_base.add(i % 3);
                }
            }
            4 => {
                // 4-byte pattern
                let pattern = core::ptr::read_unaligned(src_base as *const u32);
                let mut i = 0;
                while i + 4 <= match_length {
                    core::ptr::write_unaligned(dst.add(i) as *mut u32, pattern);
                    i += 4;
                }
                while i < match_length {
                    *dst.add(i) = *src_base.add(i % 4);
                    i += 1;
                }
            }
            5..=7 => {
                // 5-7 byte patterns - copy in chunks
                for i in 0..match_length {
                    *dst.add(i) = *src_base.add(i % offset);
                }
            }
            _ => {
                // offset >= 8: copy in 8-byte chunks where possible
                let mut i = 0;
                // Copy full offset-sized chunks
                while i + offset <= match_length {
                    core::ptr::copy_nonoverlapping(src_base, dst.add(i), offset);
                    i += offset;
                }
                // Copy remaining bytes
                if i < match_length {
                    core::ptr::copy_nonoverlapping(src_base, dst.add(i), match_length - i);
                }
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompress_context_creation() {
        let ctx = DecompressContext::new(1024);
        assert_eq!(ctx.window_size, 1024);
        assert!(ctx.output.is_empty());
    }

    #[test]
    fn test_repeat_offsets() {
        let mut ctx = DecompressContext::new(1024);

        // Initial offsets
        assert_eq!(ctx.get_repeat_offset(1), 1);
        assert_eq!(ctx.get_repeat_offset(2), 4);
        assert_eq!(ctx.get_repeat_offset(3), 8);

        // Update with new offset
        ctx.update_offsets(100);
        assert_eq!(ctx.get_repeat_offset(1), 100);
        assert_eq!(ctx.get_repeat_offset(2), 1);
        assert_eq!(ctx.get_repeat_offset(3), 4);

        // Update again
        ctx.update_offsets(200);
        assert_eq!(ctx.get_repeat_offset(1), 200);
        assert_eq!(ctx.get_repeat_offset(2), 100);
        assert_eq!(ctx.get_repeat_offset(3), 1);
    }

    #[test]
    fn test_repeat_offset_same_value() {
        let mut ctx = DecompressContext::new(1024);
        ctx.update_offsets(100);

        // Same offset shouldn't shift
        ctx.update_offsets(100);
        assert_eq!(ctx.get_repeat_offset(1), 100);
        assert_eq!(ctx.get_repeat_offset(2), 1);
    }

    #[test]
    fn test_magic_validation() {
        // Invalid magic
        let result = decompress_frame(&[0x00, 0x00, 0x00, 0x00]);
        assert!(result.is_err());

        // Too short
        let result = decompress_frame(&[0x28, 0xB5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_magic() {
        // Valid magic but incomplete frame
        let data = [0x28, 0xB5, 0x2F, 0xFD, 0x00];
        let result = decompress_frame(&data);
        // Should fail for truncated header, not magic
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_raw_frame() {
        // Construct a minimal valid frame with a raw block
        // Magic: 0xFD2FB528
        // Frame header: single segment, 1-byte FCS, no dict, no checksum
        // Block: raw, last, size = 5
        // Data: "Hello"

        let mut frame = vec![];

        // Magic number (little-endian)
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: FCS=0, single_segment=1, checksum=0, dict=0
        // 0b00100000 = 0x20
        frame.push(0x20);

        // FCS (1 byte): size = 5
        frame.push(5);

        // Block header: last=1, type=Raw(0), size=5
        // Header = (5 << 3) | (0 << 1) | 1 = 41 = 0x29
        // 3 bytes little-endian
        frame.extend_from_slice(&[0x29, 0x00, 0x00]);

        // Raw block data
        frame.extend_from_slice(b"Hello");

        let result = decompress_frame(&frame).unwrap();
        assert_eq!(result, b"Hello");
    }

    #[test]
    fn test_rle_frame() {
        // Frame with RLE block
        let mut frame = vec![];

        // Magic
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: single segment, FCS=1 byte
        frame.push(0x20);

        // FCS: size = 10
        frame.push(10);

        // Block header: last=1, type=RLE(1), size=10
        // Header = (10 << 3) | (1 << 1) | 1 = 83 = 0x53
        frame.extend_from_slice(&[0x53, 0x00, 0x00]);

        // RLE byte
        frame.push(b'X');

        let result = decompress_frame(&frame).unwrap();
        assert_eq!(result, vec![b'X'; 10]);
    }

    #[test]
    fn test_multi_block_frame() {
        // Frame with multiple blocks
        let mut frame = vec![];

        // Magic
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: single segment, FCS=1 byte
        frame.push(0x20);

        // FCS: size = 8 (5 + 3)
        frame.push(8);

        // Block 1: not last, type=Raw, size=5
        // Header = (5 << 3) | (0 << 1) | 0 = 40 = 0x28
        frame.extend_from_slice(&[0x28, 0x00, 0x00]);
        frame.extend_from_slice(b"Hello");

        // Block 2: last, type=Raw, size=3
        // Header = (3 << 3) | (0 << 1) | 1 = 25 = 0x19
        frame.extend_from_slice(&[0x19, 0x00, 0x00]);
        frame.extend_from_slice(b"!!!");

        let result = decompress_frame(&frame).unwrap();
        assert_eq!(result, b"Hello!!!");
    }

    #[test]
    fn test_content_size_mismatch() {
        // Frame declaring wrong size
        let mut frame = vec![];

        // Magic
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: single segment, FCS=1 byte
        frame.push(0x20);

        // FCS: says size = 10, but actual is 5
        frame.push(10);

        // Block: raw, last, size=5
        frame.extend_from_slice(&[0x29, 0x00, 0x00]);
        frame.extend_from_slice(b"Hello");

        let result = decompress_frame(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn test_frame_with_checksum() {
        // Frame with checksum enabled
        let mut frame = vec![];

        // Magic
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: single segment, FCS=1 byte, checksum=1
        // 0b00100100 = 0x24
        frame.push(0x24);

        // FCS: size = 5
        frame.push(5);

        // Block: raw, last, size=5
        frame.extend_from_slice(&[0x29, 0x00, 0x00]);
        frame.extend_from_slice(b"Hello");

        // Checksum: XXHash64 of "Hello", low 32 bits
        let hash = xxhash64(b"Hello", 0);
        let checksum = (hash & 0xFFFFFFFF) as u32;
        frame.extend_from_slice(&checksum.to_le_bytes());

        let result = decompress_frame(&frame).unwrap();
        assert_eq!(result, b"Hello");
    }

    #[test]
    fn test_checksum_mismatch() {
        // Frame with wrong checksum
        let mut frame = vec![];

        // Magic
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: single segment, FCS=1 byte, checksum=1
        frame.push(0x24);

        // FCS: size = 5
        frame.push(5);

        // Block: raw, last, size=5
        frame.extend_from_slice(&[0x29, 0x00, 0x00]);
        frame.extend_from_slice(b"Hello");

        // Wrong checksum
        frame.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        let result = decompress_frame(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn test_compressed_block_literals_only() {
        // Compressed block with only raw literals (no sequences)
        let mut frame = vec![];

        // Magic
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: single segment, FCS=1 byte
        frame.push(0x20);

        // FCS: size = 5
        frame.push(5);

        // Block header: last=1, type=Compressed(2), compressed_size
        // Compressed block data:
        // - Literals section: Raw, 5 bytes
        // - Sequences section: 0 sequences
        let literals = b"Hello";
        let compressed_block = build_compressed_block_literals_only(literals);

        let block_size = compressed_block.len();
        // Header = (size << 3) | (2 << 1) | 1 = (size << 3) | 5
        let header = (block_size << 3) | 5;
        frame.push((header & 0xFF) as u8);
        frame.push(((header >> 8) & 0xFF) as u8);
        frame.push(((header >> 16) & 0xFF) as u8);

        frame.extend_from_slice(&compressed_block);

        let result = decompress_frame(&frame).unwrap();
        assert_eq!(result, b"Hello");
    }

    /// Build a compressed block with only raw literals (no sequences).
    fn build_compressed_block_literals_only(literals: &[u8]) -> Vec<u8> {
        let mut block = vec![];

        // Literals section header (Raw type)
        // Block type = 0 (Raw), size_format based on size
        let size = literals.len();

        if size <= 31 {
            // 5-bit size: header = (size << 3) | (0 << 2) | 0
            block.push(((size << 3) | 0) as u8);
        } else if size <= 4095 {
            // 12-bit size: 2 bytes
            // byte0 = (size[3:0] << 4) | (1 << 2) | 0
            // byte1 = size[11:4]
            let byte0 = ((size & 0xF) << 4) | (1 << 2);
            let byte1 = (size >> 4) & 0xFF;
            block.push(byte0 as u8);
            block.push(byte1 as u8);
        } else {
            // 20-bit size: 3 bytes (not testing this case)
            unreachable!("Size too large for test");
        }

        // Literals data
        block.extend_from_slice(literals);

        // Sequences section: 0 sequences
        block.push(0);

        block
    }

    #[test]
    fn test_compressed_block_with_rle_literals() {
        // Compressed block with RLE literals (no sequences)
        let mut frame = vec![];

        // Magic
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: single segment, FCS=1 byte
        frame.push(0x20);

        // FCS: size = 10 (RLE of 'A' repeated 10 times)
        frame.push(10);

        // Build compressed block with RLE literals
        let compressed_block = build_compressed_block_rle_literals(b'A', 10);

        let block_size = compressed_block.len();
        let header = (block_size << 3) | 5; // type=2, last=1
        frame.push((header & 0xFF) as u8);
        frame.push(((header >> 8) & 0xFF) as u8);
        frame.push(((header >> 16) & 0xFF) as u8);

        frame.extend_from_slice(&compressed_block);

        let result = decompress_frame(&frame).unwrap();
        assert_eq!(result, vec![b'A'; 10]);
    }

    /// Build a compressed block with RLE literals.
    fn build_compressed_block_rle_literals(byte: u8, repeat_count: usize) -> Vec<u8> {
        let mut block = vec![];

        // Literals section header (RLE type = 1)
        // Block type = 1 (RLE), size_format based on repeat_count
        if repeat_count <= 31 {
            // 5-bit size: header = (size << 3) | (0 << 2) | 1
            block.push(((repeat_count << 3) | 1) as u8);
        } else if repeat_count <= 4095 {
            // 12-bit size
            let byte0 = ((repeat_count & 0xF) << 4) | (1 << 2) | 1;
            let byte1 = (repeat_count >> 4) & 0xFF;
            block.push(byte0 as u8);
            block.push(byte1 as u8);
        } else {
            unreachable!("Size too large for test");
        }

        // RLE byte
        block.push(byte);

        // Sequences section: 0 sequences
        block.push(0);

        block
    }

    #[test]
    fn test_compressed_block_multi_literals() {
        // Test with larger raw literals (12-bit size format)
        let mut frame = vec![];

        // Magic
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);

        // Frame descriptor: single segment, FCS=2 bytes (for sizes 256-65791)
        frame.push(0x40);

        // Create 100-byte literal data
        let literals: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();

        // FCS: size = 100 (2 bytes: size-256 for FCS field 1, or just size for single_segment)
        // Actually for FCS_Field_Size = 2, it's (size - 256) stored
        // But with single_segment, it's just the raw value
        // Let me check the frame header...
        // FCS_Field_Size=1 means 2 bytes, and value is stored + 256
        // So for size=100, we need FCS_Field_Size=0 (1 byte) which means 0x20
        // Let me fix this

        // Actually, let me use FCS=1 byte which supports 0-255
        let mut frame = vec![];
        frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);
        frame.push(0x20); // single segment, 1-byte FCS
        frame.push(100); // FCS = 100

        let compressed_block = build_compressed_block_literals_only(&literals);

        let block_size = compressed_block.len();
        let header = (block_size << 3) | 5;
        frame.push((header & 0xFF) as u8);
        frame.push(((header >> 8) & 0xFF) as u8);
        frame.push(((header >> 16) & 0xFF) as u8);

        frame.extend_from_slice(&compressed_block);

        let result = decompress_frame(&frame).unwrap();
        assert_eq!(result, literals);
    }
}
