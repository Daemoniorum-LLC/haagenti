//! Zstd frame header parsing.
//!
//! The frame header contains the frame descriptor and optional fields.

use haagenti_core::{Error, Result};

/// Frame header descriptor byte.
///
/// ```text
/// Bit 7-6: Frame_Content_Size_flag
/// Bit 5:   Single_Segment_flag
/// Bit 4:   Unused_bit (must be 0)
/// Bit 3:   Reserved_bit (must be 0)
/// Bit 2:   Content_Checksum_flag
/// Bit 1-0: Dictionary_ID_flag
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDescriptor {
    /// Raw descriptor byte.
    raw: u8,
}

impl FrameDescriptor {
    /// Parse a frame descriptor from a byte.
    pub fn new(byte: u8) -> Result<Self> {
        // Check reserved bit (bit 3)
        if byte & 0x08 != 0 {
            return Err(Error::corrupted(
                "Reserved bit in frame descriptor must be 0",
            ));
        }

        Ok(Self { raw: byte })
    }

    /// Get the Frame_Content_Size_flag (bits 7-6).
    /// Returns the number of bytes used for frame content size:
    /// 0 = 0 bytes, 1 = 1 byte, 2 = 2 bytes, 3 = 8 bytes
    #[inline]
    pub fn frame_content_size_flag(&self) -> u8 {
        (self.raw >> 6) & 0x03
    }

    /// Get the number of bytes for the frame content size field.
    pub fn frame_content_size_bytes(&self) -> usize {
        match self.frame_content_size_flag() {
            0 => {
                if self.single_segment_flag() {
                    1 // Single segment mode uses 1 byte
                } else {
                    0 // No FCS field
                }
            }
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        }
    }

    /// Get the Single_Segment_flag (bit 5).
    /// When set, window size is derived from frame content size.
    #[inline]
    pub fn single_segment_flag(&self) -> bool {
        (self.raw & 0x20) != 0
    }

    /// Get the Content_Checksum_flag (bit 2).
    /// When set, a 4-byte XXHash64 checksum follows the last block.
    #[inline]
    pub fn content_checksum_flag(&self) -> bool {
        (self.raw & 0x04) != 0
    }

    /// Get the Dictionary_ID_flag (bits 1-0).
    /// Returns the number of bytes for dictionary ID: 0, 1, 2, or 4.
    #[inline]
    pub fn dictionary_id_flag(&self) -> u8 {
        self.raw & 0x03
    }

    /// Get the number of bytes for the dictionary ID field.
    pub fn dictionary_id_bytes(&self) -> usize {
        match self.dictionary_id_flag() {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 4,
            _ => unreachable!(),
        }
    }

    /// Whether this frame requires a window descriptor byte.
    #[inline]
    pub fn has_window_descriptor(&self) -> bool {
        !self.single_segment_flag()
    }
}

/// Parsed Zstd frame header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    /// Frame descriptor.
    pub descriptor: FrameDescriptor,
    /// Window size in bytes.
    pub window_size: usize,
    /// Dictionary ID (0 if not present).
    pub dictionary_id: u32,
    /// Frame content size (None if not present).
    pub frame_content_size: Option<u64>,
    /// Whether content checksum is present.
    pub has_checksum: bool,
    /// Total header size in bytes (including magic number).
    pub header_size: usize,
}

impl FrameHeader {
    /// Parse a frame header from the input buffer.
    ///
    /// The buffer should start at the frame descriptor (after magic number).
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::corrupted("Empty frame header"));
        }

        let descriptor = FrameDescriptor::new(data[0])?;
        let mut offset = 1;

        // Parse window descriptor if present
        let window_size = if descriptor.has_window_descriptor() {
            if data.len() < offset + 1 {
                return Err(Error::corrupted(
                    "Frame header truncated at window descriptor",
                ));
            }
            let window_byte = data[offset];
            offset += 1;
            Self::decode_window_size(window_byte)?
        } else {
            // Will be determined from frame content size
            0
        };

        // Parse dictionary ID if present
        let dict_bytes = descriptor.dictionary_id_bytes();
        let dictionary_id = if dict_bytes > 0 {
            if data.len() < offset + dict_bytes {
                return Err(Error::corrupted("Frame header truncated at dictionary ID"));
            }
            let dict_id = Self::read_le_uint(&data[offset..], dict_bytes)?;
            offset += dict_bytes;
            dict_id as u32
        } else {
            0
        };

        // Parse frame content size if present
        let fcs_bytes = descriptor.frame_content_size_bytes();
        let frame_content_size = if fcs_bytes > 0 {
            if data.len() < offset + fcs_bytes {
                return Err(Error::corrupted(
                    "Frame header truncated at frame content size",
                ));
            }
            let mut fcs = Self::read_le_uint(&data[offset..], fcs_bytes)?;
            // For 2-byte FCS, add 256
            if fcs_bytes == 2 {
                fcs += 256;
            }
            offset += fcs_bytes;
            Some(fcs)
        } else {
            None
        };

        // Determine final window size
        let final_window_size = if descriptor.single_segment_flag() {
            frame_content_size.unwrap_or(0) as usize
        } else {
            window_size
        };

        Ok(Self {
            descriptor,
            window_size: final_window_size,
            dictionary_id,
            frame_content_size,
            has_checksum: descriptor.content_checksum_flag(),
            header_size: 4 + offset, // 4 bytes for magic + header bytes
        })
    }

    /// Decode window size from the window descriptor byte.
    fn decode_window_size(byte: u8) -> Result<usize> {
        let exponent = (byte >> 3) as u32;
        let mantissa = (byte & 0x07) as usize;

        if exponent > 41 {
            return Err(Error::corrupted(format!(
                "Window size exponent {} exceeds maximum",
                exponent
            )));
        }

        // window_base = 1 << (10 + exponent)
        // window_add = (window_base / 8) * mantissa
        // window_size = window_base + window_add

        let window_base = 1usize << (10 + exponent);
        let window_add = (window_base >> 3) * mantissa;
        let window_size = window_base + window_add;

        if window_size > super::MAX_WINDOW_SIZE {
            return Err(Error::corrupted(format!(
                "Window size {} exceeds maximum {}",
                window_size,
                super::MAX_WINDOW_SIZE
            )));
        }

        Ok(window_size)
    }

    /// Read a little-endian unsigned integer of the given size.
    fn read_le_uint(data: &[u8], size: usize) -> Result<u64> {
        if data.len() < size {
            return Err(Error::corrupted("Insufficient data for integer"));
        }

        let mut result = 0u64;
        for (i, &byte) in data.iter().enumerate().take(size) {
            result |= (byte as u64) << (8 * i);
        }
        Ok(result)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_descriptor_flags() {
        // Test descriptor with all flags
        // FCS=3 (bits 7-6 = 11), Single=1 (bit 5), Checksum=1 (bit 2), DictID=3 (bits 1-0)
        let desc = FrameDescriptor::new(0b11100111).unwrap();
        assert_eq!(desc.frame_content_size_flag(), 3);
        assert!(desc.single_segment_flag());
        assert!(desc.content_checksum_flag());
        assert_eq!(desc.dictionary_id_flag(), 3);
    }

    #[test]
    fn test_frame_descriptor_reserved_bit_error() {
        // Reserved bit (bit 3) must be 0
        let result = FrameDescriptor::new(0b00001000);
        assert!(result.is_err());
    }

    #[test]
    fn test_frame_descriptor_fcs_bytes() {
        // FCS flag 0, no single segment -> 0 bytes
        let desc = FrameDescriptor::new(0b00000000).unwrap();
        assert_eq!(desc.frame_content_size_bytes(), 0);

        // FCS flag 0, single segment -> 1 byte
        let desc = FrameDescriptor::new(0b00100000).unwrap();
        assert_eq!(desc.frame_content_size_bytes(), 1);

        // FCS flag 1 -> 2 bytes
        let desc = FrameDescriptor::new(0b01000000).unwrap();
        assert_eq!(desc.frame_content_size_bytes(), 2);

        // FCS flag 2 -> 4 bytes
        let desc = FrameDescriptor::new(0b10000000).unwrap();
        assert_eq!(desc.frame_content_size_bytes(), 4);

        // FCS flag 3 -> 8 bytes
        let desc = FrameDescriptor::new(0b11000000).unwrap();
        assert_eq!(desc.frame_content_size_bytes(), 8);
    }

    #[test]
    fn test_frame_descriptor_dict_bytes() {
        // Dict ID flag 0 -> 0 bytes
        let desc = FrameDescriptor::new(0b00000000).unwrap();
        assert_eq!(desc.dictionary_id_bytes(), 0);

        // Dict ID flag 1 -> 1 byte
        let desc = FrameDescriptor::new(0b00000001).unwrap();
        assert_eq!(desc.dictionary_id_bytes(), 1);

        // Dict ID flag 2 -> 2 bytes
        let desc = FrameDescriptor::new(0b00000010).unwrap();
        assert_eq!(desc.dictionary_id_bytes(), 2);

        // Dict ID flag 3 -> 4 bytes
        let desc = FrameDescriptor::new(0b00000011).unwrap();
        assert_eq!(desc.dictionary_id_bytes(), 4);
    }

    #[test]
    fn test_window_descriptor_has() {
        // No single segment -> has window descriptor
        let desc = FrameDescriptor::new(0b00000000).unwrap();
        assert!(desc.has_window_descriptor());

        // Single segment -> no window descriptor
        let desc = FrameDescriptor::new(0b00100000).unwrap();
        assert!(!desc.has_window_descriptor());
    }

    #[test]
    fn test_frame_header_minimal() {
        // Minimal header: single segment, FCS=1 byte, no dict, no checksum
        // Descriptor: 0b00100000 = 0x20
        // FCS: 0x00 (size = 0)
        let data = [0x20, 0x00];
        let header = FrameHeader::parse(&data).unwrap();

        assert!(header.descriptor.single_segment_flag());
        assert_eq!(header.frame_content_size, Some(0));
        assert_eq!(header.dictionary_id, 0);
        assert!(!header.has_checksum);
        assert_eq!(header.header_size, 4 + 2); // magic + descriptor + fcs
    }

    #[test]
    fn test_frame_header_with_window() {
        // Header with window descriptor
        // Descriptor: 0b00000000 = 0x00 (has window, no FCS, no dict, no checksum)
        // Window: exponent=0, mantissa=0 -> 1KB
        let data = [0x00, 0x00];
        let header = FrameHeader::parse(&data).unwrap();

        assert_eq!(header.window_size, 1024);
        assert_eq!(header.frame_content_size, None);
        assert_eq!(header.header_size, 4 + 2);
    }

    #[test]
    fn test_frame_header_with_dictionary() {
        // Header with 4-byte dictionary ID
        // Descriptor: 0b00100011 = 0x23 (single segment, FCS=1, dict=4 bytes)
        // Dict ID: 0x12345678
        // FCS: 0x10
        let data = [0x23, 0x78, 0x56, 0x34, 0x12, 0x10];
        let header = FrameHeader::parse(&data).unwrap();

        assert_eq!(header.dictionary_id, 0x12345678);
        assert_eq!(header.frame_content_size, Some(0x10));
    }

    #[test]
    fn test_frame_header_with_checksum() {
        // Header with checksum flag
        // Descriptor: 0b00100100 = 0x24 (single segment, checksum)
        let data = [0x24, 0x00]; // FCS = 0
        let header = FrameHeader::parse(&data).unwrap();

        assert!(header.has_checksum);
    }

    #[test]
    fn test_frame_header_2byte_fcs() {
        // 2-byte FCS: actual size = value + 256
        // Descriptor: 0b01100000 = 0x60 (single segment, FCS=2 bytes)
        // FCS: 0x0100 (little-endian) = 256 -> actual = 256 + 256 = 512
        let data = [0x60, 0x00, 0x01];
        let header = FrameHeader::parse(&data).unwrap();

        assert_eq!(header.frame_content_size, Some(256 + 256));
    }

    #[test]
    fn test_frame_header_8byte_fcs() {
        // 8-byte FCS
        // Descriptor: 0b11100000 = 0xE0 (single segment, FCS=8 bytes)
        let data = [
            0xE0, // descriptor
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // FCS
        ];
        let header = FrameHeader::parse(&data).unwrap();

        assert_eq!(header.frame_content_size, Some(0x0807060504030201));
    }

    #[test]
    fn test_window_size_decoding() {
        // Exponent 0, mantissa 0: 1KB
        assert_eq!(FrameHeader::decode_window_size(0x00).unwrap(), 1024);

        // Exponent 0, mantissa 7: 1KB + 7/8*1KB = 1KB + 896 = 1920
        assert_eq!(FrameHeader::decode_window_size(0x07).unwrap(), 1024 + 896);

        // Exponent 10, mantissa 0: 1MB
        assert_eq!(FrameHeader::decode_window_size(0x50).unwrap(), 1024 * 1024);

        // Exponent 17, mantissa 0: 128MB (max)
        assert_eq!(
            FrameHeader::decode_window_size(0x88).unwrap(),
            128 * 1024 * 1024
        );
    }

    #[test]
    fn test_window_size_too_large() {
        // Exponent 18 would give 256MB, exceeding max
        // 0x90 = exponent 18, mantissa 0
        let result = FrameHeader::decode_window_size(0x90);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_header_error() {
        let result = FrameHeader::parse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_header_error() {
        // Descriptor requires window byte but it's missing
        let result = FrameHeader::parse(&[0x00]);
        assert!(result.is_err());
    }
}
