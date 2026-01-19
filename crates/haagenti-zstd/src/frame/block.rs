//! Zstd block header parsing.
//!
//! Each data block in a Zstd frame has a 3-byte header.

use haagenti_core::{Error, Result};

/// Block types in Zstd.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    /// Raw block - uncompressed data.
    Raw,
    /// RLE block - single byte repeated.
    Rle,
    /// Compressed block - uses Zstd compression.
    Compressed,
    /// Reserved - invalid, should not appear.
    Reserved,
}

impl BlockType {
    /// Parse block type from the type field (2 bits).
    pub fn from_field(field: u8) -> Result<Self> {
        match field {
            0 => Ok(BlockType::Raw),
            1 => Ok(BlockType::Rle),
            2 => Ok(BlockType::Compressed),
            3 => Err(Error::corrupted("Reserved block type")),
            _ => unreachable!(),
        }
    }
}

/// Parsed block header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHeader {
    /// Whether this is the last block in the frame.
    pub last_block: bool,
    /// Block type.
    pub block_type: BlockType,
    /// Block size in bytes.
    /// For Raw: size of uncompressed data
    /// For RLE: decompressed size (compressed is 1 byte)
    /// For Compressed: size of compressed data
    pub block_size: usize,
}

impl BlockHeader {
    /// Block header size in bytes.
    pub const SIZE: usize = 3;

    /// Maximum block size (128 KB - 1).
    pub const MAX_BLOCK_SIZE: usize = (1 << 17) - 1;

    /// Parse a block header from 3 bytes.
    ///
    /// ```text
    /// Byte 0-2 (little-endian):
    ///   Bit 0:     Last_Block flag
    ///   Bits 1-2:  Block_Type
    ///   Bits 3-23: Block_Size (21 bits)
    /// ```
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SIZE {
            return Err(Error::corrupted(format!(
                "Block header too short: {} bytes, need {}",
                data.len(),
                Self::SIZE
            )));
        }

        // Read 3 bytes as little-endian 24-bit integer
        let header = data[0] as u32 | ((data[1] as u32) << 8) | ((data[2] as u32) << 16);

        let last_block = (header & 0x01) != 0;
        let block_type_field = ((header >> 1) & 0x03) as u8;
        let block_size = (header >> 3) as usize;

        let block_type = BlockType::from_field(block_type_field)?;

        if block_size > Self::MAX_BLOCK_SIZE {
            return Err(Error::corrupted(format!(
                "Block size {} exceeds maximum {}",
                block_size,
                Self::MAX_BLOCK_SIZE
            )));
        }

        Ok(Self {
            last_block,
            block_type,
            block_size,
        })
    }

    /// Get the size of compressed data to read.
    /// For RLE blocks, this is 1 (the byte to repeat).
    pub fn compressed_size(&self) -> usize {
        match self.block_type {
            BlockType::Raw => self.block_size,
            BlockType::Rle => 1,
            BlockType::Compressed => self.block_size,
            BlockType::Reserved => 0,
        }
    }

    /// Get the size of decompressed output.
    pub fn decompressed_size(&self) -> usize {
        self.block_size
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_type_parsing() {
        assert_eq!(BlockType::from_field(0).unwrap(), BlockType::Raw);
        assert_eq!(BlockType::from_field(1).unwrap(), BlockType::Rle);
        assert_eq!(BlockType::from_field(2).unwrap(), BlockType::Compressed);
        assert!(BlockType::from_field(3).is_err());
    }

    #[test]
    fn test_raw_block_header() {
        // Raw block, not last, size = 100
        // Encoding: last=0, type=00, size=100
        // Header = (100 << 3) | (0 << 1) | 0 = 800 = 0x320
        // Little-endian: [0x20, 0x03, 0x00]
        let data = [0x20, 0x03, 0x00];
        let header = BlockHeader::parse(&data).unwrap();

        assert!(!header.last_block);
        assert_eq!(header.block_type, BlockType::Raw);
        assert_eq!(header.block_size, 100);
        assert_eq!(header.compressed_size(), 100);
        assert_eq!(header.decompressed_size(), 100);
    }

    #[test]
    fn test_rle_block_header() {
        // RLE block, last block, size = 1000
        // Encoding: last=1, type=01, size=1000
        // Header = (1000 << 3) | (1 << 1) | 1 = 8003 = 0x1F43
        // Little-endian: [0x43, 0x1F, 0x00]
        let data = [0x43, 0x1F, 0x00];
        let header = BlockHeader::parse(&data).unwrap();

        assert!(header.last_block);
        assert_eq!(header.block_type, BlockType::Rle);
        assert_eq!(header.block_size, 1000);
        assert_eq!(header.compressed_size(), 1); // RLE is always 1 byte compressed
        assert_eq!(header.decompressed_size(), 1000);
    }

    #[test]
    fn test_compressed_block_header() {
        // Compressed block, not last, size = 50000
        // Encoding: last=0, type=10, size=50000
        // Header = (50000 << 3) | (2 << 1) | 0 = 400004 = 0x61A84
        // Little-endian: [0x84, 0x1A, 0x06]
        let data = [0x84, 0x1A, 0x06];
        let header = BlockHeader::parse(&data).unwrap();

        assert!(!header.last_block);
        assert_eq!(header.block_type, BlockType::Compressed);
        assert_eq!(header.block_size, 50000);
        assert_eq!(header.compressed_size(), 50000);
    }

    #[test]
    fn test_last_block_flag() {
        // Same as raw block but with last_block = 1
        // Header = (100 << 3) | (0 << 1) | 1 = 801 = 0x321
        let data = [0x21, 0x03, 0x00];
        let header = BlockHeader::parse(&data).unwrap();

        assert!(header.last_block);
        assert_eq!(header.block_type, BlockType::Raw);
        assert_eq!(header.block_size, 100);
    }

    #[test]
    fn test_max_block_size() {
        // Maximum size: 2^17 - 1 = 131071
        // Header = (131071 << 3) | (0 << 1) | 0 = 1048568 = 0xFFFF8
        let data = [0xF8, 0xFF, 0x0F];
        let header = BlockHeader::parse(&data).unwrap();

        assert_eq!(header.block_size, 131071);
        assert_eq!(header.block_size, BlockHeader::MAX_BLOCK_SIZE);
    }

    #[test]
    fn test_block_size_too_large() {
        // Size = 131072 (one more than max)
        // Header = (131072 << 3) | 0 = 1048576 = 0x100000
        let data = [0x00, 0x00, 0x10];
        let result = BlockHeader::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_reserved_block_type_error() {
        // Reserved block type (type = 3)
        // Header = (0 << 3) | (3 << 1) | 0 = 6
        let data = [0x06, 0x00, 0x00];
        let result = BlockHeader::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_too_short() {
        let result = BlockHeader::parse(&[0x00, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_size_block() {
        // Zero-size raw block
        let data = [0x00, 0x00, 0x00];
        let header = BlockHeader::parse(&data).unwrap();

        assert_eq!(header.block_size, 0);
        assert_eq!(header.compressed_size(), 0);
    }
}
