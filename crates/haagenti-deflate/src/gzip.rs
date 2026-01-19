//! Gzip format wrapper (RFC 1952).
//!
//! Gzip adds a header with metadata and CRC-32 checksum around DEFLATE data.
//! Used for HTTP compression, .gz files, and tar archives.

use haagenti_core::{CompressionLevel, Error, Result};

use crate::deflate::deflate;
use crate::inflate::inflate;

/// Gzip magic number.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Compression method: DEFLATE.
const CM_DEFLATE: u8 = 8;

/// Header flags.
#[allow(dead_code)]
const FTEXT: u8 = 1;
const FHCRC: u8 = 2;
const FEXTRA: u8 = 4;
const FNAME: u8 = 8;
const FCOMMENT: u8 = 16;

/// OS identifier: Unix.
const OS_UNIX: u8 = 3;

/// Compress data with gzip wrapper.
pub fn gzip_compress(input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
    let compressed = deflate(input, level)?;

    let xfl = match level {
        CompressionLevel::Best | CompressionLevel::Ultra => 2,
        CompressionLevel::Fast => 4,
        _ => 0,
    };

    // Build header
    let mut output = Vec::with_capacity(10 + compressed.len() + 8);

    // Magic number
    output.extend_from_slice(&GZIP_MAGIC);

    // Compression method
    output.push(CM_DEFLATE);

    // Flags (none set)
    output.push(0);

    // Modification time (0 = not available)
    output.extend_from_slice(&[0, 0, 0, 0]);

    // Extra flags
    output.push(xfl);

    // Operating system
    output.push(OS_UNIX);

    // Compressed data
    output.extend_from_slice(&compressed);

    // Trailer: CRC-32 and original size
    let crc = crc32(input);
    let size = input.len() as u32;

    output.push(crc as u8);
    output.push((crc >> 8) as u8);
    output.push((crc >> 16) as u8);
    output.push((crc >> 24) as u8);

    output.push(size as u8);
    output.push((size >> 8) as u8);
    output.push((size >> 16) as u8);
    output.push((size >> 24) as u8);

    Ok(output)
}

/// Decompress gzip-wrapped data.
pub fn gzip_decompress(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() < 18 {
        return Err(Error::corrupted("gzip data too short"));
    }

    // Verify magic number
    if input[0] != GZIP_MAGIC[0] || input[1] != GZIP_MAGIC[1] {
        return Err(Error::corrupted("invalid gzip magic"));
    }

    // Check compression method
    if input[2] != CM_DEFLATE {
        return Err(Error::corrupted("unsupported compression method"));
    }

    let flags = input[3];
    let mut pos = 10; // Skip fixed header

    // Skip extra field if present
    if flags & FEXTRA != 0 {
        if pos + 2 > input.len() {
            return Err(Error::unexpected_eof(pos));
        }
        let xlen = (input[pos] as usize) | ((input[pos + 1] as usize) << 8);
        pos += 2 + xlen;
    }

    // Skip filename if present
    if flags & FNAME != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1; // Skip null terminator
    }

    // Skip comment if present
    if flags & FCOMMENT != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1; // Skip null terminator
    }

    // Skip header CRC if present
    if flags & FHCRC != 0 {
        pos += 2;
    }

    if pos + 8 > input.len() {
        return Err(Error::corrupted("gzip data truncated"));
    }

    // Extract compressed data (leave 8 bytes for trailer)
    let data_end = input.len() - 8;
    let compressed = &input[pos..data_end];

    // Decompress
    let mut output = Vec::new();
    inflate(compressed, &mut output)?;

    // Verify CRC-32
    let stored_crc = (input[data_end] as u32)
        | ((input[data_end + 1] as u32) << 8)
        | ((input[data_end + 2] as u32) << 16)
        | ((input[data_end + 3] as u32) << 24);

    let computed_crc = crc32(&output);

    if stored_crc != computed_crc {
        return Err(Error::corrupted("gzip CRC mismatch"));
    }

    // Verify size (mod 2^32)
    let stored_size = (input[data_end + 4] as u32)
        | ((input[data_end + 5] as u32) << 8)
        | ((input[data_end + 6] as u32) << 16)
        | ((input[data_end + 7] as u32) << 24);

    if stored_size != (output.len() as u32) {
        return Err(Error::corrupted("gzip size mismatch"));
    }

    Ok(output)
}

/// Decompress gzip data with known output size.
pub fn gzip_decompress_to(input: &[u8], output: &mut [u8]) -> Result<usize> {
    let decompressed = gzip_decompress(input)?;

    if decompressed.len() > output.len() {
        return Err(Error::buffer_too_small(decompressed.len(), output.len()));
    }

    output[..decompressed.len()].copy_from_slice(&decompressed);
    Ok(decompressed.len())
}

/// CRC-32 lookup table.
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB88320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Calculate CRC-32 checksum.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF_u32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32() {
        // Known test vectors
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_gzip_roundtrip_empty() {
        let input = b"";
        let compressed = gzip_compress(input, CompressionLevel::Default).unwrap();
        let decompressed = gzip_decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_gzip_roundtrip_small() {
        let input = b"Hello, World!";
        let compressed = gzip_compress(input, CompressionLevel::Default).unwrap();
        let decompressed = gzip_decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_gzip_roundtrip_large() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = pattern.iter().cycle().take(10000).copied().collect();

        let compressed = gzip_compress(&input, CompressionLevel::Default).unwrap();

        // Should compress
        assert!(compressed.len() < input.len());

        let decompressed = gzip_decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_gzip_interop_decompress() {
        // Compress with flate2, decompress with us
        let input = b"Testing gzip interoperability with flate2 reference.";

        let compressed: Vec<u8> = {
            let mut c = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            std::io::Write::write_all(&mut c, input).unwrap();
            c.finish().unwrap()
        };

        let decompressed = gzip_decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_gzip_interop_compress() {
        // Compress with us, decompress with flate2
        let input = b"Testing our gzip compression against flate2.";

        let compressed = gzip_compress(input, CompressionLevel::Default).unwrap();

        let decompressed: Vec<u8> = {
            let mut d = flate2::read::GzDecoder::new(&compressed[..]);
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut d, &mut out).unwrap();
            out
        };

        assert_eq!(decompressed.as_slice(), input);
    }
}
