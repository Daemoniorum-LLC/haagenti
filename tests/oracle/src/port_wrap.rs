//! haagenti-deflate's `zlib_decompress` and `gzip_decompress` at 28eb2cb, copied unchanged except that
//! they call the patched `port_inflate::inflate` (spec §6 D1). Their checksums are copied too: Rust's
//! are private.

use crate::port_inflate::inflate;
use haagenti_core::{Error, Result};

pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &x in data {
        crc ^= x as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { 0xEDB8_8320 ^ (crc >> 1) } else { crc >> 1 };
        }
    }
    !crc
}

pub fn zlib_decompress(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() < 6 {
        return Err(Error::corrupted("zlib data too short"));
    }
    let (cmf, flg) = (input[0], input[1]);
    if cmf & 0x0F != 8 {
        return Err(Error::corrupted("unsupported compression method"));
    }
    if cmf >> 4 > 7 {
        return Err(Error::corrupted("invalid window size"));
    }
    if ((cmf as u16) * 256 + flg as u16) % 31 != 0 {
        return Err(Error::corrupted("zlib header checksum failed"));
    }
    if flg & 0x20 != 0 {
        return Err(Error::corrupted("preset dictionary not supported"));
    }
    let end = input.len() - 4;
    let mut out = Vec::new();
    inflate(&input[2..end], &mut out)?;
    let stored = u32::from_be_bytes([input[end], input[end + 1], input[end + 2], input[end + 3]]);
    if stored != adler32(&out) {
        return Err(Error::corrupted("zlib checksum mismatch"));
    }
    Ok(out)
}

pub fn gzip_decompress(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() < 18 {
        return Err(Error::corrupted("gzip data too short"));
    }
    if input[0] != 0x1f || input[1] != 0x8b {
        return Err(Error::corrupted("invalid gzip magic"));
    }
    if input[2] != 8 {
        return Err(Error::corrupted("unsupported compression method"));
    }
    let flags = input[3];
    let mut pos = 10;
    if flags & 4 != 0 {
        if pos + 2 > input.len() {
            return Err(Error::unexpected_eof(pos));
        }
        pos += 2 + ((input[pos] as usize) | ((input[pos + 1] as usize) << 8));
    }
    for flag in [8u8, 16] {
        if flags & flag != 0 {
            while pos < input.len() && input[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
    }
    if flags & 2 != 0 {
        pos += 2;
    }
    if pos + 8 > input.len() {
        return Err(Error::corrupted("gzip data truncated"));
    }
    let end = input.len() - 8;
    let mut out = Vec::new();
    inflate(&input[pos..end], &mut out)?;
    let t = &input[end..];
    if u32::from_le_bytes([t[0], t[1], t[2], t[3]]) != crc32(&out) {
        return Err(Error::corrupted("gzip CRC mismatch"));
    }
    if u32::from_le_bytes([t[4], t[5], t[6], t[7]]) != out.len() as u32 {
        return Err(Error::corrupted("gzip size mismatch"));
    }
    Ok(out)
}
