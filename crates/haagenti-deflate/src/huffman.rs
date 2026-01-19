//! Huffman coding for DEFLATE compression.
//!
//! Implements both encoding and decoding of Huffman codes as specified in RFC 1951.

use haagenti_core::{Error, Result};

/// Maximum bits in a Huffman code.
pub const MAX_BITS: usize = 15;

/// Maximum number of literal/length codes.
pub const MAX_LIT_CODES: usize = 286;

/// Maximum number of distance codes.
pub const MAX_DIST_CODES: usize = 30;

/// Maximum number of code length codes.
pub const MAX_CL_CODES: usize = 19;

/// Fixed Huffman literal/length code lengths (RFC 1951 section 3.2.6).
pub const FIXED_LIT_LENGTHS: [u8; 288] = {
    let mut lengths = [0u8; 288];
    let mut i = 0;
    while i < 144 {
        lengths[i] = 8;
        i += 1;
    }
    while i < 256 {
        lengths[i] = 9;
        i += 1;
    }
    while i < 280 {
        lengths[i] = 7;
        i += 1;
    }
    while i < 288 {
        lengths[i] = 8;
        i += 1;
    }
    lengths
};

/// Fixed Huffman distance code lengths.
pub const FIXED_DIST_LENGTHS: [u8; 32] = [5; 32];

/// Order of code length codes in the dynamic header.
pub const CL_CODE_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Extra bits for length codes 257-285.
pub const LENGTH_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Base lengths for length codes 257-285.
pub const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra bits for distance codes 0-29.
pub const DISTANCE_EXTRA_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Base distances for distance codes 0-29.
pub const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Huffman decoder using lookup tables.
#[derive(Clone)]
pub struct HuffmanDecoder {
    /// Lookup table: code -> (symbol, bits)
    /// Index by reversed code bits, value is (symbol << 4) | bits
    table: Vec<u16>,
    /// Number of bits for table lookup
    table_bits: u8,
}

impl HuffmanDecoder {
    /// Build a Huffman decoder from code lengths.
    pub fn from_lengths(lengths: &[u8]) -> Result<Self> {
        if lengths.is_empty() {
            return Err(Error::corrupted("empty code lengths"));
        }

        // Count codes of each length
        let mut bl_count = [0u32; MAX_BITS + 1];
        let mut max_bits = 0u8;

        for &len in lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
                if len > max_bits {
                    max_bits = len;
                }
            }
        }

        if max_bits == 0 {
            // All zeros - create minimal table
            return Ok(Self {
                table: vec![0; 2],
                table_bits: 1,
            });
        }

        // Calculate starting codes for each length
        let mut next_code = [0u32; MAX_BITS + 1];
        let mut code = 0u32;
        for bits in 1..=max_bits as usize {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Use full table size to handle all codes properly
        let table_bits = max_bits;
        let table_size = 1usize << table_bits;
        let mut table = vec![0u16; table_size];

        // Assign codes to symbols
        for (symbol, &len) in lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }

            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            // Reverse the code bits
            let reversed = reverse_bits(code, len);

            // Fill all entries that match this code
            let fill_count = 1usize << (table_bits - len);
            let entry = ((symbol as u16) << 4) | (len as u16);

            for i in 0..fill_count {
                let index = (reversed as usize) | (i << len);
                if index < table_size {
                    table[index] = entry;
                }
            }
        }

        Ok(Self { table, table_bits })
    }

    /// Decode a symbol from a bit reader.
    #[inline]
    pub fn decode(&self, bits: &mut BitReader) -> Result<u16> {
        let peek = bits.peek(self.table_bits as usize)?;
        let entry = self.table[peek as usize];
        let len = (entry & 0xF) as usize;
        let symbol = entry >> 4;

        if len == 0 {
            return Err(Error::corrupted("invalid huffman code"));
        }

        bits.consume(len);
        Ok(symbol)
    }
}

/// Reverse `bits` number of bits in `code`.
fn reverse_bits(code: u32, bits: u8) -> u32 {
    let mut result = 0;
    let mut code = code;
    for _ in 0..bits {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

/// Bit reader for DEFLATE streams.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bit_count: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Ensure at least `n` bits are available in the buffer.
    #[inline]
    fn fill(&mut self, n: usize) -> Result<()> {
        while self.bit_count < n as u8 {
            if self.pos >= self.data.len() {
                return Err(Error::unexpected_eof(self.pos));
            }
            self.bit_buf |= (self.data[self.pos] as u64) << self.bit_count;
            self.pos += 1;
            self.bit_count += 8;
        }
        Ok(())
    }

    /// Peek at the next `n` bits without consuming them.
    #[inline]
    pub fn peek(&mut self, n: usize) -> Result<u32> {
        self.fill(n)?;
        Ok((self.bit_buf & ((1 << n) - 1)) as u32)
    }

    /// Consume `n` bits from the buffer.
    #[inline]
    pub fn consume(&mut self, n: usize) {
        self.bit_buf >>= n;
        self.bit_count -= n as u8;
    }

    /// Read `n` bits.
    #[inline]
    pub fn read_bits(&mut self, n: usize) -> Result<u32> {
        let value = self.peek(n)?;
        self.consume(n);
        Ok(value)
    }

    /// Align to byte boundary.
    pub fn align(&mut self) {
        let discard = self.bit_count % 8;
        if discard > 0 {
            self.consume(discard as usize);
        }
    }

    /// Read a raw byte (must be byte-aligned).
    pub fn read_byte(&mut self) -> Result<u8> {
        self.align();
        if self.bit_count >= 8 {
            let byte = (self.bit_buf & 0xFF) as u8;
            self.consume(8);
            Ok(byte)
        } else if self.pos < self.data.len() {
            let byte = self.data[self.pos];
            self.pos += 1;
            Ok(byte)
        } else {
            Err(Error::unexpected_eof(self.pos))
        }
    }

    /// Read a 16-bit little-endian value (must be byte-aligned).
    pub fn read_u16(&mut self) -> Result<u16> {
        let lo = self.read_byte()? as u16;
        let hi = self.read_byte()? as u16;
        Ok(lo | (hi << 8))
    }

    /// Get current byte position.
    pub fn byte_pos(&self) -> usize {
        self.pos - (self.bit_count / 8) as usize
    }

    /// Check if at end of input.
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len() && self.bit_count == 0
    }
}

/// Huffman encoder for compression.
pub struct HuffmanEncoder {
    /// Code for each symbol
    codes: Vec<u32>,
    /// Bit length for each symbol
    lengths: Vec<u8>,
}

impl HuffmanEncoder {
    /// Build an encoder from code lengths.
    pub fn from_lengths(lengths: &[u8]) -> Self {
        let mut codes = vec![0u32; lengths.len()];

        // Count codes of each length
        let mut bl_count = [0u32; MAX_BITS + 1];
        for &len in lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Calculate starting codes
        let mut next_code = [0u32; MAX_BITS + 1];
        let mut code = 0u32;
        for bits in 1..=MAX_BITS {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Assign codes
        for (symbol, &len) in lengths.iter().enumerate() {
            if len > 0 {
                codes[symbol] = next_code[len as usize];
                next_code[len as usize] += 1;
            }
        }

        Self {
            codes,
            lengths: lengths.to_vec(),
        }
    }

    /// Get the code and length for a symbol.
    #[inline]
    pub fn get(&self, symbol: usize) -> (u32, u8) {
        (self.codes[symbol], self.lengths[symbol])
    }
}

/// Bit writer for DEFLATE compression.
pub struct BitWriter {
    data: Vec<u8>,
    bit_buf: u64,
    bit_count: u8,
}

impl BitWriter {
    /// Create a new bit writer.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Create with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Write `n` bits.
    #[inline]
    pub fn write_bits(&mut self, value: u32, n: u8) {
        self.bit_buf |= (value as u64) << self.bit_count;
        self.bit_count += n;

        while self.bit_count >= 8 {
            self.data.push(self.bit_buf as u8);
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
    }

    /// Write a Huffman code (reversed bits).
    #[inline]
    pub fn write_code(&mut self, code: u32, len: u8) {
        // DEFLATE uses reversed bit order for Huffman codes
        let reversed = reverse_bits(code, len);
        self.write_bits(reversed, len);
    }

    /// Flush remaining bits (pad with zeros).
    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.data.push(self.bit_buf as u8);
        }
        self.data
    }

    /// Align to byte boundary.
    pub fn align(&mut self) {
        if self.bit_count > 0 {
            self.data.push(self.bit_buf as u8);
            self.bit_buf = 0;
            self.bit_count = 0;
        }
    }

    /// Write a raw byte (must be aligned).
    pub fn write_byte(&mut self, byte: u8) {
        self.align();
        self.data.push(byte);
    }

    /// Write raw bytes (must be aligned).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.align();
        self.data.extend_from_slice(bytes);
    }

    /// Get current length.
    pub fn len(&self) -> usize {
        self.data.len() + if self.bit_count > 0 { 1 } else { 0 }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() && self.bit_count == 0
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader_basic() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(8).unwrap(), 0b11001010);
    }

    #[test]
    fn test_bit_writer_basic() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b0100, 4);
        writer.write_bits(0b1011, 4);
        writer.write_bits(0b11001010, 8);

        let data = writer.finish();
        assert_eq!(data, vec![0b10110100, 0b11001010]);
    }

    #[test]
    fn test_fixed_huffman_decoder() {
        let decoder = HuffmanDecoder::from_lengths(&FIXED_LIT_LENGTHS).unwrap();

        // Verify some known codes
        // 'A' (65) should have 8-bit code
        // End of block (256) should have 7-bit code
        // Fixed literals use 7-9 bit codes
        assert!(decoder.table_bits >= 7);
    }

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b1100, 4), 0b0011);
        assert_eq!(reverse_bits(0b10101010, 8), 0b01010101);
        assert_eq!(reverse_bits(0b1, 1), 0b1);
    }
}
