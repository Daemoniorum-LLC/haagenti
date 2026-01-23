//! Zstandard frame format.
//!
//! Implements parsing for Zstd frames and blocks according to RFC 8878.
//!
//! ## Frame Structure
//!
//! ```text
//! +-------------------+
//! | Magic_Number      | 4 bytes (0xFD2FB528)
//! +-------------------+
//! | Frame_Header      | 2-14 bytes
//! +-------------------+
//! | Data_Block(s)     | variable
//! +-------------------+
//! | Content_Checksum  | 0-4 bytes (optional)
//! +-------------------+
//! ```
//!
//! ## References
//!
//! - [RFC 8878 Section 3.1](https://datatracker.ietf.org/doc/html/rfc8878#section-3.1)

mod block;
mod checksum;
mod header;

pub use block::{BlockHeader, BlockType};
pub use checksum::xxhash64;
pub use header::{FrameDescriptor, FrameHeader};

/// Zstd magic number (little-endian: 0xFD2FB528).
pub const ZSTD_MAGIC: u32 = 0xFD2FB528;

/// Skippable frame magic range: 0x184D2A50 to 0x184D2A5F.
pub const SKIPPABLE_MAGIC_LOW: u32 = 0x184D2A50;
pub const SKIPPABLE_MAGIC_HIGH: u32 = 0x184D2A5F;

/// Maximum window size (128 MB).
pub const MAX_WINDOW_SIZE: usize = 1 << 27;

/// Minimum window size (1 KB).
pub const MIN_WINDOW_SIZE: usize = 1 << 10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_constants() {
        assert_eq!(ZSTD_MAGIC, 0xFD2FB528);
        assert_eq!(SKIPPABLE_MAGIC_LOW, 0x184D2A50);
        assert_eq!(SKIPPABLE_MAGIC_HIGH, 0x184D2A5F);
    }

    #[test]
    fn test_window_size_constants() {
        assert_eq!(MAX_WINDOW_SIZE, 128 * 1024 * 1024);
        assert_eq!(MIN_WINDOW_SIZE, 1024);
    }

    #[test]
    fn test_skippable_magic_range() {
        // 16 possible skippable frame magic values
        for i in 0..16 {
            let magic = SKIPPABLE_MAGIC_LOW + i;
            assert!(magic >= SKIPPABLE_MAGIC_LOW);
            assert!(magic <= SKIPPABLE_MAGIC_HIGH);
        }
    }
}
