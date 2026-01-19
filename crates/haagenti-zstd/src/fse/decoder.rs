//! FSE stream decoder.
//!
//! Implements the Finite State Entropy decoder for Zstandard.

use super::table::FseTable;
use haagenti_core::{Error, Result};

/// FSE bitstream decoder.
///
/// Reads symbols from a backward bitstream using an FSE table.
/// Zstd uses backward bitstreams - bits are read from high to low.
#[derive(Debug)]
pub struct FseDecoder<'a> {
    /// The FSE decoding table.
    table: &'a FseTable,
    /// Current decoder state.
    state: usize,
}

impl<'a> FseDecoder<'a> {
    /// Create a new FSE decoder with the given table.
    pub fn new(table: &'a FseTable) -> Self {
        Self { table, state: 0 }
    }

    /// Initialize the decoder state from the bitstream.
    ///
    /// Reads `accuracy_log` bits to set the initial state.
    pub fn init_state(&mut self, bits: &mut BitReader) -> Result<()> {
        let accuracy_log = self.table.accuracy_log();
        self.state = bits.read_bits(accuracy_log as usize)? as usize;
        Ok(())
    }

    /// Decode the current symbol and update state.
    ///
    /// Returns the decoded symbol.
    pub fn decode_symbol(&mut self, bits: &mut BitReader) -> Result<u8> {
        let entry = self.table.decode(self.state);
        let symbol = entry.symbol;

        // Read bits for next state
        let add_bits = bits.read_bits(entry.num_bits as usize)? as u16;
        self.state = (entry.baseline + add_bits) as usize;

        Ok(symbol)
    }

    /// Peek at the current symbol without advancing state.
    pub fn peek_symbol(&self) -> u8 {
        self.table.decode(self.state).symbol
    }

    /// Peek at how many bits the next decode operation needs.
    pub fn peek_num_bits(&self) -> u8 {
        self.table.decode(self.state).num_bits
    }

    /// Peek at the direct sequence base value for the current state.
    /// Used by predefined sequence tables that store baseValue directly.
    pub fn peek_seq_base(&self) -> u32 {
        self.table.decode(self.state).seq_base
    }

    /// Peek at the number of extra bits for sequence decoding.
    /// Used by predefined sequence tables.
    pub fn peek_seq_extra_bits(&self) -> u8 {
        self.table.decode(self.state).seq_extra_bits
    }

    /// Update state by reading FSE transition bits (without returning symbol).
    ///
    /// Call this AFTER reading extra bits, to update state for next symbol.
    /// For the last sequence, skip this call (no state update needed).
    pub fn update_state(&mut self, bits: &mut BitReader) -> Result<()> {
        let entry = self.table.decode(self.state);
        let add_bits = bits.read_bits(entry.num_bits as usize)? as u16;
        self.state = (entry.baseline + add_bits) as usize;
        Ok(())
    }

    /// Get the current state (for debugging/testing).
    pub fn state(&self) -> usize {
        self.state
    }

    /// Manually set state (for testing).
    #[cfg(test)]
    pub fn set_state(&mut self, state: usize) {
        self.state = state;
    }
}

/// Forward bitstream reader.
///
/// Reads bits from a byte buffer in LSB-first order.
/// This is the simpler forward direction used for testing.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    /// The data buffer.
    data: &'a [u8],
    /// Current byte position.
    byte_pos: usize,
    /// Current bit position within current byte (0-7, 0 is LSB).
    bit_pos: u8,
    /// Whether this reader is in reversed mode (MSB-first, for Huffman).
    reversed: bool,
    /// For reversed mode: current byte index (counts down).
    rev_byte_idx: usize,
    /// For reversed mode: current bit position (7 = MSB, counts down).
    rev_bit_pos: i8,
    /// For reversed mode: total bits available (excluding sentinel).
    rev_total_bits: usize,
    /// For FSE mode: bit container loaded from bytes (little-endian).
    fse_container: u64,
    /// For FSE mode: bits consumed from container (0-63).
    fse_bits_consumed: usize,
    /// For FSE mode: total bits available in stream (excluding sentinel).
    fse_total_bits: usize,
    /// For FSE mode: total bits consumed from stream overall.
    fse_stream_bits_consumed: usize,
    /// For FSE mode: byte position of first byte in current container.
    fse_byte_pos: usize,
    /// Whether using FSE mode (LSB-first from little-endian container).
    fse_mode: bool,
}

impl<'a> BitReader<'a> {
    /// Create a new bitstream reader (forward mode).
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
            reversed: false,
            rev_byte_idx: 0,
            rev_bit_pos: 0,
            rev_total_bits: 0,
            fse_container: 0,
            fse_bits_consumed: 0,
            fse_total_bits: 0,
            fse_stream_bits_consumed: 0,
            fse_byte_pos: 0,
            fse_mode: false,
        }
    }

    /// Create a new bitstream reader for reversed Huffman streams.
    ///
    /// Zstd Huffman streams are stored in reverse order with a sentinel '1' bit
    /// at the end (MSB position). This constructor finds the sentinel and
    /// positions the reader to start just below it.
    ///
    /// Bits are read from the sentinel position downward, moving to previous
    /// bytes as needed. This is the standard Zstd Huffman stream format.
    pub fn new_reversed(data: &'a [u8]) -> Result<BitReader<'a>> {
        if data.is_empty() {
            return Err(Error::corrupted("Empty bitstream"));
        }

        // Find the sentinel '1' bit in the last byte (highest set bit)
        let last_byte = data[data.len() - 1];
        if last_byte == 0 {
            return Err(Error::corrupted("Invalid bitstream: no sentinel bit"));
        }

        // Sentinel position: highest set bit (0-7, 7 is MSB)
        let sentinel_pos = 7 - last_byte.leading_zeros() as i8;

        // Total bits available: all bits in previous bytes + bits below sentinel
        let prev_bytes_bits = (data.len() - 1) * 8;
        let last_byte_bits = sentinel_pos as usize; // bits 0..(sentinel_pos-1)
        let total_bits = prev_bytes_bits + last_byte_bits;

        // Start reading from just below the sentinel
        let start_byte_idx = data.len() - 1;
        let start_bit_pos = sentinel_pos - 1; // -1 can be negative if sentinel at bit 0

        Ok(Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
            reversed: true,
            rev_byte_idx: start_byte_idx,
            rev_bit_pos: start_bit_pos,
            rev_total_bits: total_bits,
            fse_container: 0,
            fse_bits_consumed: 0,
            fse_total_bits: 0,
            fse_stream_bits_consumed: 0,
            fse_byte_pos: 0,
            fse_mode: false,
        })
    }

    /// Initialize from the last byte, finding the sentinel bit.
    ///
    /// Zstd bitstreams have a sentinel '1' bit at the end to mark the boundary.
    /// This sets up reversed mode for reading from the sentinel position downward.
    pub fn init_from_end(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Err(Error::corrupted("Empty bitstream"));
        }

        // Find the sentinel '1' bit in the last byte
        let last_byte = self.data[self.data.len() - 1];
        if last_byte == 0 {
            return Err(Error::corrupted("Invalid bitstream: no sentinel bit"));
        }

        // Find highest set bit (the sentinel)
        let sentinel_pos = 7 - last_byte.leading_zeros() as i8;

        // Total bits available: all bits in previous bytes + bits below sentinel
        let prev_bytes_bits = (self.data.len() - 1) * 8;
        let last_byte_bits = sentinel_pos as usize; // bits 0..(sentinel_pos-1)
        let total_bits = prev_bytes_bits + last_byte_bits;

        // Set up reversed mode starting from just below the sentinel
        self.reversed = true;
        self.rev_byte_idx = self.data.len() - 1;
        self.rev_bit_pos = sentinel_pos - 1; // -1 can be negative if sentinel at bit 0
        self.rev_total_bits = total_bits;

        Ok(())
    }

    /// Initialize for FSE bitstream reading (Zstd sequence bitstream).
    ///
    /// FSE bitstreams use a different bit ordering than Huffman:
    /// - Bytes are loaded into a little-endian container
    /// - The sentinel bit marks the end
    /// - Bits are read from the LSB going UP (bitsConsumed starts at 0)
    ///
    /// This matches zstd's BIT_DStream behavior.
    pub fn init_fse(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Err(Error::corrupted("Empty bitstream"));
        }

        // Load bytes into container (little-endian)
        let mut container: u64 = 0;
        for (i, &byte) in self.data.iter().enumerate() {
            if i >= 8 {
                break; // Only load up to 8 bytes
            }
            container |= (byte as u64) << (i * 8);
        }

        // Find the sentinel (highest set bit)
        if container == 0 {
            return Err(Error::corrupted("Invalid bitstream: no sentinel bit"));
        }
        let sentinel_pos = 63 - container.leading_zeros() as usize;

        // Total data bits = sentinel_pos (bits 0 to sentinel_pos-1)
        let total_bits = sentinel_pos;

        self.fse_mode = true;
        self.fse_container = container;
        self.fse_bits_consumed = 0;
        self.fse_total_bits = total_bits;
        self.fse_stream_bits_consumed = 0;
        self.fse_byte_pos = 0;

        Ok(())
    }

    /// Switch to LSB-first reading for the remaining bits.
    ///
    /// After reading initial states in reversed (MSB-first) mode, call this to
    /// switch to LSB-first mode for reading extra bits. This is because zstd
    /// bitstreams have initial states at the end (near sentinel) and extra bits
    /// at the beginning (read from bit 0 going up).
    pub fn switch_to_lsb_mode(&mut self) -> Result<()> {
        if !self.reversed {
            return Err(Error::corrupted("switch_to_lsb_mode requires reversed mode"));
        }

        // Load the remaining bits into the FSE container
        let remaining_bits = self.rev_total_bits;
        if remaining_bits == 0 {
            self.fse_mode = true;
            self.fse_container = 0;
            self.fse_bits_consumed = 0;
            self.fse_total_bits = 0;
            self.fse_stream_bits_consumed = 0;
            self.fse_byte_pos = 0;
            return Ok(());
        }

        // Read the remaining bytes into a little-endian container
        // The remaining bits are in bytes [0..rev_byte_idx] plus some bits in rev_byte_idx
        let mut container: u64 = 0;
        for (i, &byte) in self.data.iter().enumerate() {
            if i >= 8 {
                break;
            }
            container |= (byte as u64) << (i * 8);
        }

        // The remaining bits to read are the LOWER bits of the stream
        // (from bit 0 up to the current read position)
        self.fse_mode = true;
        self.fse_container = container;
        self.fse_bits_consumed = 0;
        self.fse_total_bits = remaining_bits;
        self.fse_stream_bits_consumed = 0;
        self.fse_byte_pos = 0;
        self.reversed = false;

        Ok(())
    }

    /// Refill the FSE container when we have consumed enough bits.
    fn fse_refill(&mut self) {
        // Only refill if we have consumed at least 32 bits (4 bytes)
        // This matches zstd reference implementation
        if self.fse_bits_consumed < 32 {
            return;
        }

        // Calculate how many whole bytes we have consumed (max 7 to stay safe)
        let bytes_consumed = (self.fse_bits_consumed / 8).min(7);
        if bytes_consumed == 0 {
            return;
        }

        // Shift out the consumed bytes (safe: bytes_consumed <= 7, so shift <= 56)
        let shift_bits = bytes_consumed * 8;
        self.fse_container >>= shift_bits;
        self.fse_bits_consumed -= shift_bits;
        self.fse_byte_pos += bytes_consumed;

        // Load new bytes into the high bits of the container
        for i in 0..bytes_consumed {
            let byte_idx = self.fse_byte_pos + 8 - bytes_consumed + i;
            if byte_idx < self.data.len() {
                let byte = self.data[byte_idx] as u64;
                let shift = (8 - bytes_consumed + i) * 8;
                self.fse_container |= byte << shift;
            }
        }
    }

    /// Read bits in FSE mode (LSB-first from little-endian container).
    fn read_bits_fse(&mut self, n: usize) -> Result<u32> {
        if self.fse_stream_bits_consumed + n > self.fse_total_bits {
            return Err(Error::unexpected_eof(self.fse_stream_bits_consumed + n));
        }

        // Refill container if needed before reading
        self.fse_refill();

        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        let value = ((self.fse_container >> self.fse_bits_consumed) as u32) & mask;
        self.fse_bits_consumed += n;
        self.fse_stream_bits_consumed += n;

        Ok(value)
    }

    /// Read `n` bits from the stream.
    ///
    /// In forward mode: reads LSB first from low to high byte indices.
    /// In reversed mode: reads from high to low bit positions, high to low byte indices.
    /// In FSE mode: reads LSB-first from little-endian container.
    pub fn read_bits(&mut self, n: usize) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(Error::corrupted("Cannot read more than 32 bits at once"));
        }

        if self.fse_mode {
            self.read_bits_fse(n)
        } else if self.reversed {
            self.read_bits_reversed(n)
        } else {
            self.read_bits_forward(n)
        }
    }

    /// Read bits in forward mode (LSB first, low to high bytes).
    fn read_bits_forward(&mut self, n: usize) -> Result<u32> {
        if !self.has_bits(n) {
            return Err(Error::unexpected_eof(self.byte_pos * 8 + self.bit_pos as usize));
        }

        let mut result = 0u32;
        let mut bits_read = 0;

        while bits_read < n {
            let byte = self.data[self.byte_pos];
            let available = 8 - self.bit_pos as usize;
            let to_read = (n - bits_read).min(available);

            // Extract bits from current position
            let mask = ((1u32 << to_read) - 1) as u8;
            let bits = (byte >> self.bit_pos) & mask;

            result |= (bits as u32) << bits_read;
            bits_read += to_read;

            self.bit_pos += to_read as u8;
            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Ok(result)
    }

    /// Read bits in reversed mode (from backward buffer).
    ///
    /// Zstd bitstreams are read from the sentinel position downward.
    /// Bits are extracted from high positions going down, and assembled
    /// with earlier-read bits as higher-order bits in the result.
    fn read_bits_reversed(&mut self, n: usize) -> Result<u32> {
        if self.rev_total_bits < n {
            return Err(Error::unexpected_eof(n));
        }

        let mut result = 0u32;
        let mut bits_read = 0;

        while bits_read < n {
            // If we're at a negative bit position, move to previous byte
            if self.rev_bit_pos < 0 {
                if self.rev_byte_idx == 0 {
                    return Err(Error::unexpected_eof(bits_read));
                }
                self.rev_byte_idx -= 1;
                self.rev_bit_pos = 7;
            }

            let byte = self.data[self.rev_byte_idx];
            let bits_to_read = (n - bits_read).min((self.rev_bit_pos + 1) as usize);

            // Extract bits from high positions going down
            // Example: if rev_bit_pos=5 and we need 3 bits, get bits 5,4,3
            let shift = (self.rev_bit_pos + 1) as usize - bits_to_read;
            let mask = ((1u32 << bits_to_read) - 1) as u8;
            let extracted = (byte >> shift) & mask;

            // Earlier-read bits become higher-order in result
            result = (result << bits_to_read) | (extracted as u32);
            bits_read += bits_to_read;

            self.rev_bit_pos -= bits_to_read as i8;
        }

        self.rev_total_bits -= n;
        Ok(result)
    }

    /// Check if we have at least n bits remaining.
    fn has_bits(&self, n: usize) -> bool {
        if self.fse_mode {
            self.fse_stream_bits_consumed + n <= self.fse_total_bits
        } else if self.reversed {
            self.rev_total_bits >= n
        } else {
            let total_bits = self.data.len() * 8;
            let consumed = self.byte_pos * 8 + self.bit_pos as usize;
            consumed + n <= total_bits
        }
    }

    /// Check if the stream is exhausted.
    pub fn is_empty(&self) -> bool {
        if self.fse_mode {
            self.fse_stream_bits_consumed >= self.fse_total_bits
        } else if self.reversed {
            self.rev_total_bits == 0
        } else {
            self.byte_pos >= self.data.len()
        }
    }

    /// Get the number of bits remaining.
    pub fn bits_remaining(&self) -> usize {
        if self.fse_mode {
            self.fse_total_bits.saturating_sub(self.fse_stream_bits_consumed)
        } else if self.reversed {
            self.rev_total_bits
        } else if self.byte_pos >= self.data.len() {
            0
        } else {
            (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
        }
    }

    /// Read bits without consuming them (peek).
    pub fn peek_bits(&self, n: usize) -> Result<u32> {
        let mut clone = self.clone();
        clone.read_bits(n)
    }

    /// Peek bits with zero padding if fewer than n bits remain.
    ///
    /// This is used for Huffman decoding where implicit zero padding
    /// exists at the front of the stream. Returns available bits
    /// shifted to MSB position, with zeros in lower positions.
    pub fn peek_bits_padded(&self, n: usize) -> Result<u32> {
        if !self.reversed {
            // Forward mode: just use normal peek
            return self.peek_bits(n);
        }

        let available = self.rev_total_bits;
        if available >= n {
            // Enough bits, normal peek
            return self.peek_bits(n);
        }

        if available == 0 {
            // No bits left at all
            return Err(Error::unexpected_eof(0));
        }

        // Read available bits and shift to MSB position
        let mut clone = self.clone();
        let bits = clone.read_bits(available)?;
        // Shift to put these bits in the high positions
        Ok(bits << (n - available))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fse::table::FseTable;

    #[test]
    fn test_bit_reader_empty() {
        let data = [];
        let reader = BitReader::new(&data);
        assert!(reader.is_empty());
        assert_eq!(reader.bits_remaining(), 0);
    }

    #[test]
    fn test_bit_reader_single_byte() {
        let data = [0b10110100]; // Binary: 10110100
        let mut reader = BitReader::new(&data);

        // Read 4 bits at a time (LSB first)
        let low4 = reader.read_bits(4).unwrap();
        let high4 = reader.read_bits(4).unwrap();

        assert_eq!(low4, 0b0100); // Low 4 bits
        assert_eq!(high4, 0b1011); // High 4 bits
    }

    #[test]
    fn test_bit_reader_multiple_bytes() {
        let data = [0xAB, 0xCD]; // Two bytes: AB CD
        let mut reader = BitReader::new(&data);

        // Read 8 bits at a time
        let first = reader.read_bits(8).unwrap();
        let second = reader.read_bits(8).unwrap();

        assert_eq!(first, 0xAB);
        assert_eq!(second, 0xCD);
    }

    #[test]
    fn test_bit_reader_cross_byte() {
        let data = [0xFF, 0x00]; // 11111111 00000000
        let mut reader = BitReader::new(&data);

        // Read 4 bits
        let first = reader.read_bits(4).unwrap();
        assert_eq!(first, 0x0F); // Low 4 bits of 0xFF

        // Read 8 bits (crosses byte boundary)
        let cross = reader.read_bits(8).unwrap();
        assert_eq!(cross, 0x0F); // High 4 of 0xFF + Low 4 of 0x00
    }

    #[test]
    fn test_bit_reader_init_from_end() {
        // Data with sentinel: last byte 0x80 = 0b10000000, sentinel at bit 7
        let data = [0x00, 0x80];
        let mut reader = BitReader::new(&data);
        reader.init_from_end().unwrap();

        // Sentinel at bit 7, so bits below sentinel = 7 (bits 0-6)
        // Previous byte = 8 bits, total = 8 + 7 = 15 bits
        assert_eq!(reader.bits_remaining(), 15);
    }

    #[test]
    fn test_bit_reader_init_from_end_lower_sentinel() {
        // Last byte 0x04 = 0b00000100, sentinel at bit 2
        let data = [0xFF, 0x04];
        let mut reader = BitReader::new(&data);
        reader.init_from_end().unwrap();

        // Sentinel at bit 2, so bits below sentinel = 2 (bits 0-1)
        // Previous byte = 8 bits, total = 8 + 2 = 10 bits
        assert_eq!(reader.bits_remaining(), 10);
    }

    #[test]
    fn test_bit_reader_eof() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);

        // Read all 8 bits
        reader.read_bits(8).unwrap();

        // Next read should fail
        let result = reader.read_bits(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_bit_reader_peek() {
        let data = [0b11110000];
        let reader = BitReader::new(&data);

        let peek1 = reader.peek_bits(4).unwrap();
        let peek2 = reader.peek_bits(4).unwrap();

        // Peek should not consume bits
        assert_eq!(peek1, peek2);
        assert_eq!(peek1, 0b0000); // Low 4 bits
        assert_eq!(reader.bits_remaining(), 8);
    }

    #[test]
    fn test_bit_reader_read_zero() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);

        let zero = reader.read_bits(0).unwrap();
        assert_eq!(zero, 0);
        assert_eq!(reader.bits_remaining(), 8);
    }

    #[test]
    fn test_fse_decoder_creation() {
        let distribution = [4i16, 4];
        let table = FseTable::build(&distribution, 3, 1).unwrap();
        let decoder = FseDecoder::new(&table);

        assert_eq!(decoder.state(), 0);
    }

    #[test]
    fn test_fse_decoder_init_state() {
        let distribution = [4i16, 4];
        let table = FseTable::build(&distribution, 3, 1).unwrap();
        let mut decoder = FseDecoder::new(&table);

        // Create a bitstream with initial state = 5 (binary: 101)
        let data = [0b00000101];
        let mut bits = BitReader::new(&data);

        decoder.init_state(&mut bits).unwrap();
        assert_eq!(decoder.state(), 5);
    }

    #[test]
    fn test_fse_decoder_decode_sequence() {
        // Build a simple table
        let distribution = [4i16, 4]; // Two symbols, equal probability
        let table = FseTable::build(&distribution, 3, 1).unwrap();
        let mut decoder = FseDecoder::new(&table);

        // Set initial state and decode a few symbols
        decoder.set_state(0);
        let sym0 = decoder.peek_symbol();

        // Symbol should be 0 or 1
        assert!(sym0 <= 1);
    }

    #[test]
    fn test_fse_decoder_state_transitions() {
        // Verify state transitions lead to valid states
        let distribution = [6i16, 2]; // Symbol 0 more common
        let table = FseTable::build(&distribution, 3, 1).unwrap();

        // For each starting state, verify transition is valid
        for start_state in 0..8 {
            let _decoder = FseDecoder::new(&table);
            let entry = table.decode(start_state);

            // Verify symbol is valid
            assert!(entry.symbol <= 1);

            // Verify num_bits is reasonable
            assert!(entry.num_bits <= 3);
        }
    }
}
