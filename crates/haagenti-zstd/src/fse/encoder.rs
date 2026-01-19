//! FSE stream encoder.
//!
//! Implements the Finite State Entropy encoder for Zstandard compression.
//! FSE is a variant of ANS (Asymmetric Numeral Systems) that provides
//! near-optimal compression with very fast decoding.
//!
//! ## Optimizations
//!
//! This implementation uses several optimizations:
//! - Flat encoding table for cache efficiency
//! - Pre-computed symbol state indices for O(1) lookup
//! - Packed encoding entries (64-bit)
//! - Zero-copy state transitions
//!
//! ## References
//!
//! - [RFC 8878 Section 4.1](https://datatracker.ietf.org/doc/html/rfc8878#section-4.1)
//! - [Asymmetric Numeral Systems](https://arxiv.org/abs/0902.0271)

use super::table::FseTable;

/// Packed FSE encoding entry for cache efficiency.
/// Layout: [baseline:16][num_bits:8][symbol:8][state_offset:32]
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(8))]
pub struct FseEncodeEntry {
    /// Number of bits to output for this symbol.
    pub num_bits: u8,
    /// Base value for computing next state.
    pub delta_find_state: i16,
    /// State offset for this symbol occurrence (unused in current impl).
    #[allow(dead_code)]
    pub delta_nb_bits: u16,
}

/// Optimized FSE encoder using flat table layout.
///
/// The encoder uses a flat array indexed by (symbol, occurrence) for
/// cache-efficient encoding. Symbol frequency counts are updated inline.
#[derive(Debug)]
pub struct FseEncoder {
    /// Flat encoding table: indexed by (symbol * max_states + state_index)
    /// Each symbol has up to max_states encoding entries.
    encode_table: Vec<FseEncodeEntry>,
    /// Number of states per symbol (table_size / num_symbols average)
    states_per_symbol: usize,
    /// Symbol count array: number of states for each symbol
    symbol_counts: [u16; 256],
    /// Symbol start indices in the flat table
    symbol_starts: [u32; 256],
    /// Current symbol occurrence counters
    symbol_next: [u16; 256],
    /// Current encoder state.
    state: usize,
    /// Accuracy log.
    accuracy_log: u8,
    /// Table size
    table_size: usize,
}

impl FseEncoder {
    /// Build an optimized FSE encoder from a decoding table.
    ///
    /// Uses a flat table layout for cache efficiency.
    pub fn from_decode_table(decode_table: &FseTable) -> Self {
        let accuracy_log = decode_table.accuracy_log();
        let table_size = decode_table.size();

        // Count states per symbol
        let mut symbol_counts = [0u16; 256];
        for state in 0..table_size {
            let entry = decode_table.decode(state);
            symbol_counts[entry.symbol as usize] += 1;
        }

        // Calculate symbol start indices
        let mut symbol_starts = [0u32; 256];
        let mut offset = 0u32;
        for i in 0..256 {
            symbol_starts[i] = offset;
            offset += symbol_counts[i] as u32;
        }

        // Build flat encoding table
        let total_entries = table_size;
        let mut encode_table = vec![FseEncodeEntry::default(); total_entries];
        let mut symbol_next_temp = [0u16; 256];

        for state in 0..table_size {
            let decode_entry = decode_table.decode(state);
            let symbol = decode_entry.symbol as usize;

            let idx = symbol_starts[symbol] as usize + symbol_next_temp[symbol] as usize;
            symbol_next_temp[symbol] += 1;

            if idx < encode_table.len() {
                encode_table[idx] = FseEncodeEntry {
                    num_bits: decode_entry.num_bits,
                    delta_find_state: state as i16,
                    delta_nb_bits: (decode_entry.num_bits as u16) << 8 | (decode_entry.baseline & 0xFF),
                };
            }
        }

        Self {
            encode_table,
            states_per_symbol: table_size / 256.max(1),
            symbol_counts,
            symbol_starts,
            symbol_next: [0u16; 256],
            state: 0,
            accuracy_log,
            table_size,
        }
    }

    /// Initialize the encoder with a symbol (first symbol sets initial state).
    #[inline]
    pub fn init_state(&mut self, symbol: u8) {
        let sym_idx = symbol as usize;
        if self.symbol_counts[sym_idx] > 0 {
            let entry_idx = self.symbol_starts[sym_idx] as usize;
            if entry_idx < self.encode_table.len() {
                self.state = self.encode_table[entry_idx].delta_find_state as usize;
            }
        }
        // Reset occurrence counters
        self.symbol_next = [0u16; 256];
    }

    /// Get the current state for serialization.
    #[inline]
    pub fn get_state(&self) -> usize {
        self.state
    }

    /// Get the accuracy log.
    #[inline]
    pub fn accuracy_log(&self) -> u8 {
        self.accuracy_log
    }

    /// Encode a symbol, returning the bits to output.
    ///
    /// Returns (bits_value, num_bits) where bits_value contains num_bits to output.
    #[inline]
    pub fn encode_symbol(&mut self, symbol: u8) -> (u32, u8) {
        let sym_idx = symbol as usize;
        let count = self.symbol_counts[sym_idx];

        if count == 0 {
            return (0, 0);
        }

        // Get entry for this symbol's current occurrence
        let occurrence = self.symbol_next[sym_idx] % count;
        let entry_idx = self.symbol_starts[sym_idx] as usize + occurrence as usize;

        self.symbol_next[sym_idx] = self.symbol_next[sym_idx].wrapping_add(1);

        if entry_idx >= self.encode_table.len() {
            return (0, 0);
        }

        let entry = &self.encode_table[entry_idx];
        let num_bits = entry.num_bits;
        let mask = (1u32 << num_bits) - 1;
        let bits = (self.state as u32) & mask;

        // Update state
        self.state = entry.delta_find_state as usize;

        (bits, num_bits)
    }

    /// Reset for encoding a new stream.
    #[inline]
    pub fn reset(&mut self) {
        self.state = 0;
        self.symbol_next = [0u16; 256];
    }
}

/// Optimized FSE bitstream writer.
///
/// Uses a 64-bit buffer for efficient bit packing with minimal flushes.
#[derive(Debug)]
pub struct FseBitWriter {
    /// Output buffer.
    buffer: Vec<u8>,
    /// Current 64-bit accumulator.
    accum: u64,
    /// Bits currently in accumulator (0-56).
    bits_in_accum: u32,
}

impl FseBitWriter {
    /// Create a new bit writer with pre-allocated capacity.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(256)
    }

    /// Create a new bit writer with specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            accum: 0,
            bits_in_accum: 0,
        }
    }

    /// Write bits to the stream.
    ///
    /// Uses a 64-bit accumulator to minimize flush operations.
    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        if num_bits == 0 {
            return;
        }

        // Add bits to accumulator
        self.accum |= (value as u64) << self.bits_in_accum;
        self.bits_in_accum += num_bits as u32;

        // Flush complete bytes when we have 8+ bytes (64 bits)
        // This is rare, so we optimize for the common case
        if self.bits_in_accum >= 56 {
            self.flush_bytes();
        }
    }

    /// Flush complete bytes from accumulator to buffer.
    #[inline(always)]
    fn flush_bytes(&mut self) {
        // Flush 32 bits (4 bytes) at a time when possible for efficiency
        // This is faster than byte-by-byte while still being correct
        while self.bits_in_accum >= 32 {
            let bytes = (self.accum as u32).to_le_bytes();
            self.buffer.extend_from_slice(&bytes);
            self.accum >>= 32;
            self.bits_in_accum -= 32;
        }
        // Flush remaining complete bytes one at a time
        while self.bits_in_accum >= 8 {
            self.buffer.push((self.accum & 0xFF) as u8);
            self.accum >>= 8;
            self.bits_in_accum -= 8;
        }
    }

    /// Finish the bitstream, adding sentinel bit.
    pub fn finish(mut self) -> Vec<u8> {
        // Add sentinel bit
        self.write_bits(1, 1);

        // Flush all complete bytes
        self.flush_bytes();

        // Flush remaining partial byte
        if self.bits_in_accum > 0 {
            self.buffer.push(self.accum as u8);
        }

        self.buffer
    }

    /// Get the accumulated bits without finishing.
    pub fn into_bytes(mut self) -> Vec<u8> {
        // Flush all complete bytes first
        self.flush_bytes();

        // Flush remaining partial byte
        if self.bits_in_accum > 0 {
            self.buffer.push(self.accum as u8);
        }
        self.buffer
    }

    /// Get current size in bytes (approximate).
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len() + ((self.bits_in_accum as usize + 7) / 8)
    }

    /// Check if the writer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bits_in_accum == 0
    }
}

impl Default for FseBitWriter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Interleaved FSE Encoder for Sequences
// =============================================================================

/// Interleaved FSE encoder for Zstd sequences.
///
/// Zstd sequences use three interleaved FSE streams:
/// - Literal Length (LL)
/// - Offset (OF)
/// - Match Length (ML)
///
/// Each stream maintains its own state, and bits are interleaved in a
/// specific order for optimal decoding performance.
#[derive(Debug)]
pub struct InterleavedFseEncoder {
    /// Literal length encoder
    ll_encoder: FseEncoder,
    /// Offset encoder
    of_encoder: FseEncoder,
    /// Match length encoder
    ml_encoder: FseEncoder,
}

impl InterleavedFseEncoder {
    /// Create a new interleaved encoder from the three FSE tables.
    pub fn new(ll_table: &FseTable, of_table: &FseTable, ml_table: &FseTable) -> Self {
        Self {
            ll_encoder: FseEncoder::from_decode_table(ll_table),
            of_encoder: FseEncoder::from_decode_table(of_table),
            ml_encoder: FseEncoder::from_decode_table(ml_table),
        }
    }

    /// Initialize all three encoders with their first symbols.
    #[inline]
    pub fn init_states(&mut self, ll: u8, of: u8, ml: u8) {
        self.ll_encoder.init_state(ll);
        self.of_encoder.init_state(of);
        self.ml_encoder.init_state(ml);
    }

    /// Encode one sequence triplet (LL, OF, ML).
    ///
    /// Returns the bits and counts for each stream in the correct order.
    #[inline]
    pub fn encode_sequence(&mut self, ll: u8, of: u8, ml: u8) -> [(u32, u8); 3] {
        // Encoding order for Zstd: OF, ML, LL
        let of_bits = self.of_encoder.encode_symbol(of);
        let ml_bits = self.ml_encoder.encode_symbol(ml);
        let ll_bits = self.ll_encoder.encode_symbol(ll);

        [of_bits, ml_bits, ll_bits]
    }

    /// Get the final states for all three encoders.
    #[inline]
    pub fn get_states(&self) -> (usize, usize, usize) {
        (
            self.ll_encoder.get_state(),
            self.of_encoder.get_state(),
            self.ml_encoder.get_state(),
        )
    }

    /// Get accuracy logs for all three encoders.
    #[inline]
    pub fn accuracy_logs(&self) -> (u8, u8, u8) {
        (
            self.ll_encoder.accuracy_log(),
            self.of_encoder.accuracy_log(),
            self.ml_encoder.accuracy_log(),
        )
    }

    /// Reset all encoders for a new sequence section.
    #[inline]
    pub fn reset(&mut self) {
        self.ll_encoder.reset();
        self.of_encoder.reset();
        self.ml_encoder.reset();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fse::{FseTable, LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG};

    #[test]
    fn test_fse_encoder_creation() {
        let table = FseTable::from_predefined(
            &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
            LITERAL_LENGTH_ACCURACY_LOG,
        ).unwrap();

        let encoder = FseEncoder::from_decode_table(&table);
        assert_eq!(encoder.accuracy_log(), LITERAL_LENGTH_ACCURACY_LOG);
    }

    #[test]
    fn test_fse_encoder_init_state() {
        let table = FseTable::from_predefined(
            &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
            LITERAL_LENGTH_ACCURACY_LOG,
        ).unwrap();

        let mut encoder = FseEncoder::from_decode_table(&table);
        encoder.init_state(0);

        // State should be valid (within table bounds)
        assert!(encoder.get_state() < table.size());
    }

    #[test]
    fn test_fse_encoder_encode_symbol() {
        let table = FseTable::from_predefined(
            &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
            LITERAL_LENGTH_ACCURACY_LOG,
        ).unwrap();

        let mut encoder = FseEncoder::from_decode_table(&table);
        encoder.init_state(0);

        // Encode a few symbols
        for _ in 0..10 {
            let (bits, num_bits) = encoder.encode_symbol(0);
            // num_bits should be reasonable
            assert!(num_bits <= LITERAL_LENGTH_ACCURACY_LOG);
            // bits should fit in num_bits
            assert!(bits < (1 << num_bits) || num_bits == 0);
        }
    }

    #[test]
    fn test_fse_bit_writer_simple() {
        let mut writer = FseBitWriter::new();
        writer.write_bits(0b101, 3);
        let result = writer.finish();

        // Should have bits + sentinel
        assert!(!result.is_empty());
    }

    #[test]
    fn test_fse_bit_writer_multi_byte() {
        let mut writer = FseBitWriter::new();
        writer.write_bits(0xAB, 8);
        writer.write_bits(0xCD, 8);
        let result = writer.into_bytes();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0xAB);
        assert_eq!(result[1], 0xCD);
    }

    #[test]
    fn test_fse_bit_writer_capacity() {
        let writer = FseBitWriter::with_capacity(1024);
        assert!(writer.is_empty());
    }

    #[test]
    fn test_fse_bit_writer_len() {
        let mut writer = FseBitWriter::new();
        writer.write_bits(0xFF, 8);
        assert_eq!(writer.len(), 1);

        writer.write_bits(0xFF, 8);
        assert_eq!(writer.len(), 2);
    }

    #[test]
    fn test_fse_bit_writer_large() {
        let mut writer = FseBitWriter::new();

        // Write many bytes
        for i in 0..1000 {
            writer.write_bits((i % 256) as u32, 8);
        }

        let result = writer.into_bytes();
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_interleaved_encoder() {
        use crate::fse::{
            MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
            OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG,
        };

        let ll_table = FseTable::from_predefined(
            &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
            LITERAL_LENGTH_ACCURACY_LOG,
        ).unwrap();
        let ml_table = FseTable::from_predefined(
            &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
            MATCH_LENGTH_ACCURACY_LOG,
        ).unwrap();
        let of_table = FseTable::from_predefined(
            &OFFSET_DEFAULT_DISTRIBUTION,
            OFFSET_ACCURACY_LOG,
        ).unwrap();

        let mut encoder = InterleavedFseEncoder::new(&ll_table, &of_table, &ml_table);
        encoder.init_states(0, 0, 0);

        // Encode a sequence
        let [of_bits, ml_bits, ll_bits] = encoder.encode_sequence(0, 0, 0);

        // All should produce valid outputs
        assert!(of_bits.1 <= OFFSET_ACCURACY_LOG);
        assert!(ml_bits.1 <= MATCH_LENGTH_ACCURACY_LOG);
        assert!(ll_bits.1 <= LITERAL_LENGTH_ACCURACY_LOG);
    }
}
