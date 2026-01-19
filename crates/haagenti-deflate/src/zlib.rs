//! Zlib format wrapper (RFC 1950).
//!
//! Zlib adds a header and Adler-32 checksum around DEFLATE data.
//! Used by PNG, git objects, and many other formats.

use haagenti_core::{CompressionLevel, Error, Result};

use crate::deflate::deflate;
use crate::inflate::inflate;

/// Zlib compression method: DEFLATE.
const CM_DEFLATE: u8 = 8;

/// Maximum window size (32K).
const CINFO_32K: u8 = 7;

/// Compress data with zlib wrapper.
pub fn zlib_compress(input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
    let compressed = deflate(input, level)?;

    // Calculate header
    let cmf = CM_DEFLATE | (CINFO_32K << 4); // CM=8, CINFO=7
    let flevel = match level {
        CompressionLevel::None => 0,
        CompressionLevel::Fast => 1,
        CompressionLevel::Default => 2,
        CompressionLevel::Best | CompressionLevel::Ultra => 3,
        CompressionLevel::Custom(l) => {
            if l <= 1 {
                0
            } else if l <= 4 {
                1
            } else if l <= 6 {
                2
            } else {
                3
            }
        }
    };

    let flg_base = flevel << 6;
    // FCHECK: make (CMF * 256 + FLG) divisible by 31
    let check = ((cmf as u16) * 256 + flg_base as u16) % 31;
    let fcheck = if check == 0 { 0 } else { 31 - check } as u8;
    let flg = flg_base | fcheck;

    // Calculate Adler-32 of uncompressed data
    let adler = adler32(input);

    // Build output
    let mut output = Vec::with_capacity(2 + compressed.len() + 4);
    output.push(cmf);
    output.push(flg);
    output.extend_from_slice(&compressed);
    output.push((adler >> 24) as u8);
    output.push((adler >> 16) as u8);
    output.push((adler >> 8) as u8);
    output.push(adler as u8);

    Ok(output)
}

/// Decompress zlib-wrapped data.
pub fn zlib_decompress(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() < 6 {
        return Err(Error::corrupted("zlib data too short"));
    }

    // Parse header
    let cmf = input[0];
    let flg = input[1];

    // Verify header
    let cm = cmf & 0x0F;
    if cm != CM_DEFLATE {
        return Err(Error::corrupted("unsupported compression method"));
    }

    let cinfo = cmf >> 4;
    if cinfo > 7 {
        return Err(Error::corrupted("invalid window size"));
    }

    // Check header checksum
    let check = ((cmf as u16) * 256 + (flg as u16)) % 31;
    if check != 0 {
        return Err(Error::corrupted("zlib header checksum failed"));
    }

    // Check for preset dictionary (not supported)
    if flg & 0x20 != 0 {
        return Err(Error::corrupted("preset dictionary not supported"));
    }

    // Extract compressed data (skip header, leave 4 bytes for Adler-32)
    let data_end = input.len() - 4;
    let compressed = &input[2..data_end];

    // Decompress
    let mut output = Vec::new();
    inflate(compressed, &mut output)?;

    // Verify Adler-32
    let stored_adler = ((input[data_end] as u32) << 24)
        | ((input[data_end + 1] as u32) << 16)
        | ((input[data_end + 2] as u32) << 8)
        | (input[data_end + 3] as u32);

    let computed_adler = adler32(&output);

    if stored_adler != computed_adler {
        return Err(Error::corrupted("zlib checksum mismatch"));
    }

    Ok(output)
}

/// Decompress zlib data with known output size.
pub fn zlib_decompress_to(input: &[u8], output: &mut [u8]) -> Result<usize> {
    let decompressed = zlib_decompress(input)?;

    if decompressed.len() > output.len() {
        return Err(Error::buffer_too_small(decompressed.len(), output.len()));
    }

    output[..decompressed.len()].copy_from_slice(&decompressed);
    Ok(decompressed.len())
}

/// Calculate Adler-32 checksum.
fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;

    let mut a: u32 = 1;
    let mut b: u32 = 0;

    for &byte in data {
        a = (a + byte as u32) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }

    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adler32() {
        // Known test vectors
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"a"), 0x00620062);
        assert_eq!(adler32(b"abc"), 0x024d0127);
    }

    #[test]
    fn test_zlib_roundtrip_empty() {
        let input = b"";
        let compressed = zlib_compress(input, CompressionLevel::Default).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_zlib_roundtrip_small() {
        let input = b"Hello, World!";
        let compressed = zlib_compress(input, CompressionLevel::Default).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_zlib_roundtrip_large() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = pattern.iter().cycle().take(10000).copied().collect();

        let compressed = zlib_compress(&input, CompressionLevel::Default).unwrap();

        // Should compress
        assert!(compressed.len() < input.len());

        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_zlib_interop_decompress() {
        // Compress with flate2, decompress with us
        let input = b"Testing zlib interoperability with flate2 reference.";

        let compressed: Vec<u8> = {
            let mut c = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            std::io::Write::write_all(&mut c, input).unwrap();
            c.finish().unwrap()
        };

        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_zlib_interop_compress() {
        // Compress with us, decompress with flate2
        let input = b"Testing our zlib compression against flate2.";

        let compressed = zlib_compress(input, CompressionLevel::Default).unwrap();

        let decompressed: Vec<u8> = {
            let mut d = flate2::read::ZlibDecoder::new(&compressed[..]);
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut d, &mut out).unwrap();
            out
        };

        assert_eq!(decompressed.as_slice(), input);
    }
}
