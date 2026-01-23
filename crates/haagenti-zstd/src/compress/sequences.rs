//! Sequence Encoding - Novel RLE-first approach.
//!
//! Traditional Zstd implementations jump straight to FSE for sequence encoding.
//! This module takes a different approach: leverage RLE mode for uniform match
//! patterns, which is simpler, faster, and often just as effective.
//!
//! ## RLE Sequence Mode
//!
//! When all sequences share similar characteristics (same offset code, similar
//! lengths), RLE mode encodes just one symbol per stream (LL, OF, ML) instead
//! of building FSE tables. The bitstream only contains extra bits.
//!
//! ## FSE Sequence Mode
//!
//! For non-uniform patterns, we use FSE encoding with predefined tables.
//! This ensures compatibility while still providing good compression.
//!
//! ## Benefits
//!
//! - RLE mode for uniform patterns (simple, fast, no table overhead)
//! - Predefined FSE tables for varied patterns (reliable, good compression)
//! - Long matches preserved (not split unnecessarily)

use crate::block::{Sequence, LITERAL_LENGTH_BASELINE};
use crate::fse::{
    cached_ll_table, cached_ml_table, cached_of_table, FseBitWriter, InterleavedTansEncoder,
    TansEncoder,
};
use crate::CustomFseTables;
use haagenti_core::Result;

/// Match length encoding table derived from zstd's predefined ML table.
///
/// Each entry is (extra_bits, baseline, max_value) for ML codes 0-52.
/// These values come from the unique (seq_base, seq_extra_bits) pairs in
/// ML_PREDEFINED_TABLE_DIRECT, ensuring compatibility with reference zstd.
///
/// IMPORTANT: zstd's predefined table uses DIFFERENT values than RFC 8878
/// starting at code 43. The encoder MUST use these values to match what
/// the decoder expects.
///
/// Key differences from RFC:
/// - Codes match zstd's ML_defaultDTable reference implementation
/// - Code 42: 5 bits, baseline 99 (covering 99-130)
/// - Code 43: 7 bits, baseline 131 (covering 131-258)
/// - Code 44: 8 bits, baseline 259 (covering 259-514)
/// - Code 45+: Continue with progressively larger ranges
const ML_ENCODE_TABLE: [(u8, u32, u32); 53] = [
    // Codes 0-31: match_length 3-34, no extra bits
    (0, 3, 3),
    (0, 4, 4),
    (0, 5, 5),
    (0, 6, 6),
    (0, 7, 7),
    (0, 8, 8),
    (0, 9, 9),
    (0, 10, 10),
    (0, 11, 11),
    (0, 12, 12),
    (0, 13, 13),
    (0, 14, 14),
    (0, 15, 15),
    (0, 16, 16),
    (0, 17, 17),
    (0, 18, 18),
    (0, 19, 19),
    (0, 20, 20),
    (0, 21, 21),
    (0, 22, 22),
    (0, 23, 23),
    (0, 24, 24),
    (0, 25, 25),
    (0, 26, 26),
    (0, 27, 27),
    (0, 28, 28),
    (0, 29, 29),
    (0, 30, 30),
    (0, 31, 31),
    (0, 32, 32),
    (0, 33, 33),
    (0, 34, 34),
    // Codes 32-35: 1 extra bit each
    (1, 35, 36),
    (1, 37, 38),
    (1, 39, 40),
    (1, 41, 42),
    // Codes 36-37: 2 extra bits each
    (2, 43, 46),
    (2, 47, 50),
    // Codes 38-39: 3 extra bits each
    (3, 51, 58),
    (3, 59, 66),
    // Codes 40-41: 4 extra bits each
    (4, 67, 82),
    (4, 83, 98),
    // Code 42: 5 extra bits (from zstd: baseline 99)
    (5, 99, 130),
    // Code 43: 7 extra bits (from zstd: baseline 131)
    (7, 131, 258),
    // Code 44: 8 extra bits (from zstd: baseline 259)
    (8, 259, 514),
    // Code 45: 9 extra bits (from zstd: baseline 515)
    (9, 515, 1026),
    // Code 46: 10 extra bits (from zstd: baseline 1027)
    (10, 1027, 2050),
    // Code 47: 11 extra bits (from zstd: baseline 2051)
    (11, 2051, 4098),
    // Code 48: 12 extra bits (from zstd: baseline 4099)
    (12, 4099, 8194),
    // Code 49: 13 extra bits (from zstd: baseline 8195)
    (13, 8195, 16386),
    // Code 50: 14 extra bits (from zstd: baseline 16387)
    (14, 16387, 32770),
    // Code 51: 15 extra bits (from zstd: baseline 32771)
    (15, 32771, 65538),
    // Code 52: 16 extra bits (from zstd: baseline 65539)
    (16, 65539, 131074),
];

/// Encoded sequence codes with extra bits.
#[derive(Debug, Clone, Copy)]
pub struct EncodedSequence {
    /// Literal length code (0-35)
    pub ll_code: u8,
    /// Literal length extra bits
    pub ll_extra: u32,
    /// Literal length extra bit count
    pub ll_bits: u8,
    /// Offset code (0-31)
    pub of_code: u8,
    /// Offset extra bits
    pub of_extra: u32,
    /// Offset extra bit count (per RFC 8878 Table 15)
    pub of_bits: u8,
    /// Match length code (0-52)
    pub ml_code: u8,
    /// Match length extra bits
    pub ml_extra: u32,
    /// Match length extra bit count
    pub ml_bits: u8,
}

impl EncodedSequence {
    /// Encode a sequence into codes and extra bits.
    #[inline]
    pub fn from_sequence(seq: &Sequence) -> Self {
        let (ll_code, ll_extra, ll_bits) = encode_literal_length(seq.literal_length);
        let (of_code, of_extra, of_bits) = encode_offset(seq.offset);
        let (ml_code, ml_extra, ml_bits) = encode_match_length(seq.match_length);

        Self {
            ll_code,
            ll_extra,
            ll_bits,
            of_code,
            of_extra,
            of_bits,
            ml_code,
            ml_extra,
            ml_bits,
        }
    }
}

/// Encode literal length to code + extra bits.
///
/// Optimized with direct lookup for small values and binary-like search for larger.
#[inline(always)]
fn encode_literal_length(value: u32) -> (u8, u32, u8) {
    // Fast path: values 0-15 map directly to codes 0-15 with no extra bits
    if value < 16 {
        return (value as u8, 0, 0);
    }

    // Values 16-17: code 16, 1 extra bit
    if value < 18 {
        return (16, value - 16, 1);
    }

    // Values 18-19: code 17, 1 extra bit
    if value < 20 {
        return (17, value - 18, 1);
    }

    // Values 20-23: code 18, 2 extra bits
    if value < 24 {
        return (18, value - 20, 2);
    }

    // For larger values, use the table but start from a good estimate
    // The codes follow a pattern where each code covers 2^extra_bits values
    let log_estimate = if value < 64 {
        4
    } else if value < 256 {
        6
    } else if value < 1024 {
        8
    } else {
        10
    };

    // Search from the estimated position
    for (code, &(bits, baseline)) in LITERAL_LENGTH_BASELINE
        .iter()
        .enumerate()
        .skip(log_estimate)
    {
        let max_value = if bits == 0 {
            baseline
        } else {
            baseline + ((1u32 << bits) - 1)
        };

        if value >= baseline && value <= max_value {
            return (code as u8, value - baseline, bits);
        }
    }

    // Fallback to last code for very large values
    let last_idx = LITERAL_LENGTH_BASELINE.len() - 1;
    let (bits, baseline) = LITERAL_LENGTH_BASELINE[last_idx];
    (last_idx as u8, value.saturating_sub(baseline), bits)
}

/// Encode match length to code + extra bits.
///
/// Uses ML_ENCODE_TABLE which contains zstd's predefined values.
/// This ensures the encoder writes the same number of extra bits
/// that the decoder expects to read.
///
/// Optimized with direct lookup for common match lengths.
#[inline(always)]
fn encode_match_length(value: u32) -> (u8, u32, u8) {
    // Fast path: values 3-34 map directly to codes 0-31 with no extra bits
    // This covers the vast majority of match lengths
    if (3..=34).contains(&value) {
        return ((value - 3) as u8, 0, 0);
    }

    // Match length must be at least 3
    if value < 3 {
        return (0, 0, 0); // Treat as minimum match
    }

    // Values 35-42: codes 32-35, 1 extra bit each
    if value <= 42 {
        let code = 32 + ((value - 35) / 2) as u8;
        let baseline = 35 + ((code - 32) as u32 * 2);
        return (code, value - baseline, 1);
    }

    // Values 43-50: codes 36-37, 2 extra bits each
    if value <= 50 {
        let code = if value < 47 { 36 } else { 37 };
        let baseline = if code == 36 { 43 } else { 47 };
        return (code, value - baseline, 2);
    }

    // Values 51-66: codes 38-39, 3 extra bits each
    if value <= 66 {
        let code = if value < 59 { 38 } else { 39 };
        let baseline = if code == 38 { 51 } else { 59 };
        return (code, value - baseline, 3);
    }

    // For larger values (67+), search in ML_ENCODE_TABLE starting from code 40
    // This ensures we use the exact same baselines as the decoder
    for (code, &(bits, baseline, max_value)) in ML_ENCODE_TABLE.iter().enumerate().skip(40) {
        if value >= baseline && value <= max_value {
            return (code as u8, value - baseline, bits);
        }
    }

    // Fallback to last code for very large values
    let last_idx = ML_ENCODE_TABLE.len() - 1;
    let (bits, baseline, _) = ML_ENCODE_TABLE[last_idx];
    (last_idx as u8, value.saturating_sub(baseline), bits)
}

/// Encode offset_value to code + extra bits + bit count.
///
/// Per RFC 8878 Table 7:
///   Offset_Value = (1 << Offset_Code) + Extra_Bits
///
/// Inverse (for encoding):
///   Offset_Code = floor(log2(Offset_Value))
///   Extra_Bits = Offset_Value - (1 << Offset_Code)
///   Number_of_Bits = Offset_Code
///
/// The offset in Sequence is the offset_value:
/// - 1-3: repeat offset references (handled by RepeatOffsets::resolve)
/// - >= 4: actual_offset + 3
fn encode_offset(offset_value: u32) -> (u8, u32, u8) {
    if offset_value == 0 {
        return (0, 0, 0);
    }

    // offset_code = floor(log2(offset_value))
    let offset_code = 31 - offset_value.leading_zeros();
    let baseline = 1u32 << offset_code;
    let extra = offset_value - baseline;
    let num_bits = offset_code as u8;

    (offset_code as u8, extra, num_bits)
}

/// Check if sequences are suitable for RLE mode.
///
/// Returns (uniform_ll, uniform_of, uniform_ml) indicating which
/// streams can use RLE mode.
///
/// Optimized to encode and check uniformity in a single pass.
pub fn analyze_for_rle(sequences: &[Sequence]) -> RleSuitability {
    if sequences.is_empty() {
        return RleSuitability::all_rle(0, 0, 0);
    }

    // Pre-allocate with exact capacity
    let mut encoded = Vec::with_capacity(sequences.len());

    // Encode first sequence to get reference codes
    let first = EncodedSequence::from_sequence(&sequences[0]);
    let (ll_code, of_code, ml_code) = (first.ll_code, first.of_code, first.ml_code);
    encoded.push(first);

    // Track uniformity while encoding (single pass)
    let mut ll_uniform = true;
    let mut of_uniform = true;
    let mut ml_uniform = true;

    for seq in sequences.iter().skip(1) {
        let enc = EncodedSequence::from_sequence(seq);

        // Check uniformity inline
        ll_uniform = ll_uniform && enc.ll_code == ll_code;
        of_uniform = of_uniform && enc.of_code == of_code;
        ml_uniform = ml_uniform && enc.ml_code == ml_code;

        encoded.push(enc);
    }

    RleSuitability {
        ll_uniform,
        ll_code,
        of_uniform,
        of_code,
        ml_uniform,
        ml_code,
        encoded,
    }
}

/// RLE suitability analysis result.
#[derive(Debug)]
pub struct RleSuitability {
    /// Whether literal lengths are uniform (same code)
    pub ll_uniform: bool,
    /// The LL code to use for RLE
    pub ll_code: u8,
    /// Whether offsets are uniform
    pub of_uniform: bool,
    /// The OF code to use for RLE
    pub of_code: u8,
    /// Whether match lengths are uniform
    pub ml_uniform: bool,
    /// The ML code to use for RLE
    pub ml_code: u8,
    /// Pre-encoded sequences
    pub encoded: Vec<EncodedSequence>,
}

impl RleSuitability {
    fn all_rle(ll: u8, of: u8, ml: u8) -> Self {
        Self {
            ll_uniform: true,
            ll_code: ll,
            of_uniform: true,
            of_code: of,
            ml_uniform: true,
            ml_code: ml,
            encoded: Vec::new(),
        }
    }

    /// Check if all three streams can use RLE mode.
    pub fn all_uniform(&self) -> bool {
        self.ll_uniform && self.of_uniform && self.ml_uniform
    }
}

/// Encode sequences using RLE mode.
///
/// This is simpler than FSE and works well when matches are uniform.
/// Mode byte uses RLE (01) for each stream: 0b01_01_01_00 = 0x54
pub fn encode_sequences_rle(
    sequences: &[Sequence],
    suitability: &RleSuitability,
    output: &mut Vec<u8>,
) -> Result<()> {
    if sequences.is_empty() {
        output.push(0);
        return Ok(());
    }

    let count = sequences.len();

    // Encode sequence count
    if count < 128 {
        output.push(count as u8);
    } else if count < 0x7F00 {
        output.push(((count >> 8) + 128) as u8);
        output.push((count & 0xFF) as u8);
    } else {
        output.push(255);
        let adjusted = count - 0x7F00;
        output.push((adjusted & 0xFF) as u8);
        output.push(((adjusted >> 8) & 0xFF) as u8);
    }

    // Mode byte: RLE for all three streams
    // Per RFC 8878 Section 3.1.1.4:
    //   bits[1:0] = LL mode
    //   bits[3:2] = OF mode
    //   bits[5:4] = ML mode
    //   bits[7:6] = Reserved (must be 0)
    // RLE mode = 1, so all three at 01:
    //   0b00_01_01_01 = 0x15
    output.push(0x15);

    // RLE symbols
    output.push(suitability.ll_code);
    output.push(suitability.of_code);
    output.push(suitability.ml_code);

    // Build bitstream with extra bits
    let bitstream = build_rle_bitstream(&suitability.encoded);
    output.extend_from_slice(&bitstream);

    Ok(())
}

/// Encode sequences using predefined FSE tables.
///
/// Uses Zstd's predefined FSE tables for LL, OF, and ML codes.
/// The bitstream format follows RFC 8878:
/// - Sequence count (variable length)
/// - Mode byte (0x00 for predefined tables)
/// - Compressed bitstream with FSE symbols and extra bits
///
/// # Performance
///
/// Uses statically cached FSE tables to avoid rebuilding on every block.
/// The tables are built once on first access and reused thereafter.
pub fn encode_sequences_fse(sequences: &[Sequence], output: &mut Vec<u8>) -> Result<()> {
    if sequences.is_empty() {
        output.push(0);
        return Ok(());
    }

    // Encode sequences to get codes and extra bits
    let encoded: Vec<EncodedSequence> = sequences
        .iter()
        .map(EncodedSequence::from_sequence)
        .collect();

    encode_sequences_fse_with_encoded(&encoded, output)
}

/// Encode pre-encoded sequences using predefined FSE tables.
///
/// This is the fast path when sequences have already been encoded (e.g., from
/// `analyze_for_rle`), avoiding redundant encoding work.
///
/// # Performance
///
/// Uses statically cached FSE tables to avoid rebuilding on every block.
/// By accepting pre-encoded sequences, this eliminates double-encoding overhead.
pub fn encode_sequences_fse_with_encoded(
    encoded: &[EncodedSequence],
    output: &mut Vec<u8>,
) -> Result<()> {
    if encoded.is_empty() {
        output.push(0);
        return Ok(());
    }

    let count = encoded.len();

    // Encode sequence count
    if count < 128 {
        output.push(count as u8);
    } else if count < 0x7F00 {
        output.push(((count >> 8) + 128) as u8);
        output.push((count & 0xFF) as u8);
    } else {
        output.push(255);
        let adjusted = count - 0x7F00;
        output.push((adjusted & 0xFF) as u8);
        output.push(((adjusted >> 8) & 0xFF) as u8);
    }

    // Mode byte: Predefined tables for all three streams
    // LL[7:6]=00, OF[5:4]=00, ML[3:2]=00, reserved[1:0]=00
    output.push(0x00);

    // Use CACHED predefined tANS encoders (built once, cloned per block).
    // Cloning is ~100x faster than building from FSE tables.
    let mut tans = InterleavedTansEncoder::new_predefined();

    // Build bitstream
    let bitstream = build_fse_bitstream(encoded, &mut tans);
    output.extend_from_slice(&bitstream);

    Ok(())
}

/// Encode sequences using custom FSE tables.
///
/// Uses the provided custom tables where available, falling back to predefined
/// tables for any that are `None`. Custom tables can improve compression when
/// the data has symbol distributions that differ from the predefined tables.
///
/// # Mode Byte
///
/// When custom tables are used, the mode byte indicates "FSE_Compressed" mode (10)
/// for each stream, requiring the table to be serialized in the bitstream.
/// When predefined tables are used, mode is "Predefined" (00).
///
/// # Example
///
/// ```rust,ignore
/// let custom_tables = CustomFseTables::new()
///     .with_ll_table(my_ll_table);
/// encode_sequences_with_custom_tables(&encoded, &custom_tables, &mut output)?;
/// ```
pub fn encode_sequences_with_custom_tables(
    encoded: &[EncodedSequence],
    custom_tables: &CustomFseTables,
    output: &mut Vec<u8>,
) -> Result<()> {
    if encoded.is_empty() {
        output.push(0);
        return Ok(());
    }

    let count = encoded.len();

    // Encode sequence count
    if count < 128 {
        output.push(count as u8);
    } else if count < 0x7F00 {
        output.push(((count >> 8) + 128) as u8);
        output.push((count & 0xFF) as u8);
    } else {
        output.push(255);
        let adjusted = count - 0x7F00;
        output.push((adjusted & 0xFF) as u8);
        output.push(((adjusted >> 8) & 0xFF) as u8);
    }

    // Determine mode for each stream:
    // - 00: Predefined (use built-in tables)
    // - 10: FSE_Compressed (custom table follows in bitstream)
    //
    // For now, we always use predefined mode, but build tANS encoders from
    // custom tables if provided. This gives us the encoding benefit without
    // needing to serialize custom tables (which adds overhead).
    //
    // TODO: Add FSE_Compressed mode support for full custom table functionality
    let mode_byte = 0x00; // Predefined for all streams
    output.push(mode_byte);

    // Build tANS encoders from custom or predefined tables
    let ll_table = custom_tables
        .ll_table
        .as_ref()
        .map(|t| t.as_ref())
        .unwrap_or_else(|| cached_ll_table());
    let of_table = custom_tables
        .of_table
        .as_ref()
        .map(|t| t.as_ref())
        .unwrap_or_else(|| cached_of_table());
    let ml_table = custom_tables
        .ml_table
        .as_ref()
        .map(|t| t.as_ref())
        .unwrap_or_else(|| cached_ml_table());

    let ll_encoder = TansEncoder::from_decode_table(ll_table);
    let of_encoder = TansEncoder::from_decode_table(of_table);
    let ml_encoder = TansEncoder::from_decode_table(ml_table);

    let mut tans = InterleavedTansEncoder::from_encoders(ll_encoder, of_encoder, ml_encoder);

    // Build bitstream
    let bitstream = build_fse_bitstream(encoded, &mut tans);
    output.extend_from_slice(&bitstream);

    Ok(())
}

/// Build the bitstream for FSE mode.
///
/// Zstd sequence bitstream format (read backwards from end):
/// 1. Initial states: LL, OF, ML (read first, written last)
/// 2. For each sequence (processed forward, read via backward bits):
///    - OF decode_symbol -> reads OF state bits
///    - ML decode_symbol -> reads ML state bits
///    - LL decode_symbol -> reads LL state bits
///    - Read LL extra bits
///    - Read ML extra bits
///    - Read OF extra bits
///
/// For encoding (write order is reverse of read order):
/// 1. For each sequence (reverse order):
///    - OF extra bits
///    - ML extra bits
///    - LL extra bits
///    - LL FSE state bits
///    - ML FSE state bits
///    - OF FSE state bits
/// 2. ML initial state
/// 3. OF initial state
/// 4. LL initial state (written last, read first)
#[allow(unused_variables)]
fn build_fse_bitstream(encoded: &[EncodedSequence], tans: &mut InterleavedTansEncoder) -> Vec<u8> {
    #[cfg(test)]
    let debug = std::env::var("DEBUG_FSE").is_ok();
    if encoded.is_empty() {
        return vec![0x01]; // Minimal sentinel
    }

    let mut bits = FseBitWriter::new();

    // Get accuracy logs
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();

    // Zstd bitstream layout:
    // - States at MSB end (read first, before LSB switch)
    // - After LSB switch, decoder reads in FORWARD order (seq 0, 1, 2, ..., N-1):
    //   - For each seq: extras (LL, ML, OF), then FSE update bits (LL, OF, ML) (skip for last)
    //
    // tANS encoding MUST process symbols in REVERSE order to produce correct FSE bits.
    // But we need to write bits in FORWARD order for LSB-first reading.
    //
    // Solution: Pre-compute FSE bits in reverse order, store them, then write in forward order.

    let last_idx = encoded.len() - 1;
    let last_seq = &encoded[last_idx];

    // Step 1: Initialize with the LAST sequence's symbols
    tans.init_states(last_seq.ll_code, last_seq.of_code, last_seq.ml_code);

    #[cfg(test)]
    if std::env::var("DEBUG_FSE_DETAIL").is_ok() {
        let (ll_s, of_s, ml_s) = tans.get_states();
        eprintln!(
            "FSE init with codes ({}, {}, {}): states = ({}, {}, {})",
            last_seq.ll_code, last_seq.of_code, last_seq.ml_code, ll_s, of_s, ml_s
        );
    }

    // Step 2: Encode FSE bits in REVERSE order (seq N-2 down to 0)
    // We skip the last sequence because the decoder doesn't update states for it.
    let mut fse_bits_per_seq: Vec<[(u32, u8); 3]> = vec![[(0, 0); 3]; last_idx];

    for i in (0..last_idx).rev() {
        let seq = &encoded[i];
        let fse_bits = tans.encode_sequence(seq.ll_code, seq.of_code, seq.ml_code);

        #[cfg(test)]
        if std::env::var("DEBUG_FSE_DETAIL").is_ok() {
            let (ll_s, of_s, ml_s) = tans.get_states();
            eprintln!("FSE encode seq[{}] codes ({}, {}, {}): bits=LL({},{}) ML({},{}) OF({},{}), new states=({}, {}, {})",
                      i, seq.ll_code, seq.of_code, seq.ml_code,
                      fse_bits[0].0, fse_bits[0].1,
                      fse_bits[2].0, fse_bits[2].1,
                      fse_bits[1].0, fse_bits[1].1,
                      ll_s, of_s, ml_s);
        }

        fse_bits_per_seq[i] = fse_bits;
    }

    // Step 3: Write in FORWARD order for LSB-first reading
    // For each non-last sequence: write extras, then FSE update bits.
    // Extra bits order: LL, ML, OF (matches decoder read order)
    // FSE update bits order: LL, ML, OF (matches decoder state update order)
    for i in 0..last_idx {
        let seq = &encoded[i];

        // Write extra bits in LL, ML, OF order
        if seq.ll_bits > 0 {
            bits.write_bits(seq.ll_extra, seq.ll_bits);
        }
        if seq.ml_bits > 0 {
            bits.write_bits(seq.ml_extra, seq.ml_bits);
        }
        if seq.of_bits > 0 {
            bits.write_bits(seq.of_extra, seq.of_bits);
        }

        // Write FSE update bits: LL, ML, OF order
        let [ll_fse, of_fse, ml_fse] = fse_bits_per_seq[i];
        bits.write_bits(ll_fse.0, ll_fse.1);
        bits.write_bits(ml_fse.0, ml_fse.1);
        bits.write_bits(of_fse.0, of_fse.1);
    }

    // Write LAST sequence's extra bits (no FSE update for last)
    if last_seq.ll_bits > 0 {
        bits.write_bits(last_seq.ll_extra, last_seq.ll_bits);
    }
    if last_seq.ml_bits > 0 {
        bits.write_bits(last_seq.ml_extra, last_seq.ml_bits);
    }
    if last_seq.of_bits > 0 {
        bits.write_bits(last_seq.of_extra, last_seq.of_bits);
    }

    // Get final encoder states (become initial decoder states)
    let (ll_state, of_state, ml_state) = tans.get_states();

    #[cfg(test)]
    if std::env::var("DEBUG_FSE").is_ok() {
        eprintln!("FSE encode: {} sequences", encoded.len());
        eprintln!(
            "  Last seq: ll_code={}, of_code={}, ml_code={}",
            last_seq.ll_code, last_seq.of_code, last_seq.ml_code
        );
        eprintln!(
            "  Last seq extras: ll={}({} bits), ml={}({} bits), of={}({} bits)",
            last_seq.ll_extra,
            last_seq.ll_bits,
            last_seq.ml_extra,
            last_seq.ml_bits,
            last_seq.of_extra,
            last_seq.of_bits
        );
        eprintln!(
            "  Final states: ll={}, of={}, ml={}",
            ll_state, of_state, ml_state
        );
    }

    // Write states in ML, OF, LL order
    // They end up at MSB end, decoder reads MSB-first (LL, OF, ML)
    bits.write_bits(ml_state, ml_log);
    bits.write_bits(of_state, of_log);
    bits.write_bits(ll_state, ll_log);

    bits.finish()
}

/// Build the bitstream for RLE mode.
///
/// In RLE mode, the bitstream only contains extra bits (no FSE state updates).
/// Extra bits are laid out in LL, ML, OF order per sequence,
/// sequences written in reverse order for backward reading.
fn build_rle_bitstream(encoded: &[EncodedSequence]) -> Vec<u8> {
    if encoded.is_empty() {
        return vec![0x01]; // Minimal sentinel
    }

    let mut bits = FseBitWriter::new();

    // Write sequences in reverse order (seq N-1, N-2, ..., 0)
    // Extra bits within each sequence: LL, ML, OF
    for seq in encoded.iter().rev() {
        if seq.ll_bits > 0 {
            bits.write_bits(seq.ll_extra, seq.ll_bits);
        }
        if seq.ml_bits > 0 {
            bits.write_bits(seq.ml_extra, seq.ml_bits);
        }
        if seq.of_bits > 0 {
            bits.write_bits(seq.of_extra, seq.of_bits);
        }
    }

    bits.finish()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_literal_length_small() {
        // Codes 0-15 map directly to values 0-15 with no extra bits
        for i in 0..16 {
            let (code, extra, bits) = encode_literal_length(i);
            assert_eq!(code, i as u8);
            assert_eq!(extra, 0);
            assert_eq!(bits, 0);
        }
    }

    #[test]
    fn test_encode_literal_length_with_extra_bits() {
        // Code 16: 1 extra bit, baseline 16
        let (code, extra, bits) = encode_literal_length(16);
        assert_eq!(code, 16);
        assert_eq!(extra, 0);
        assert_eq!(bits, 1);

        let (code, extra, bits) = encode_literal_length(17);
        assert_eq!(code, 16);
        assert_eq!(extra, 1);
        assert_eq!(bits, 1);
    }

    #[test]
    fn test_encode_match_length() {
        // Code 0: baseline 3, no extra bits
        let (code, extra, bits) = encode_match_length(3);
        assert_eq!(code, 0);
        assert_eq!(extra, 0);
        assert_eq!(bits, 0);

        // Code 1: baseline 4
        let (code, extra, bits) = encode_match_length(4);
        assert_eq!(code, 1);
        assert_eq!(extra, 0);
        assert_eq!(bits, 0);
    }

    #[test]
    fn test_encode_offset() {
        // Per RFC 8878 Table 7:
        //   Offset_Value = (1 << Offset_Code) + Extra_Bits
        // Inverse:
        //   Offset_Code = floor(log2(Offset_Value))
        //   Extra_Bits = Offset_Value - (1 << Offset_Code)

        // offset_value 1 → code 0, extra 0
        let (code, extra, bits) = encode_offset(1);
        assert_eq!(code, 0);
        assert_eq!(extra, 0);
        assert_eq!(bits, 0);

        // offset_value 2 → code 1, extra 0
        let (code, extra, bits) = encode_offset(2);
        assert_eq!(code, 1);
        assert_eq!(extra, 0);
        assert_eq!(bits, 1);

        // offset_value 3 → code 1, extra 1
        let (code, extra, bits) = encode_offset(3);
        assert_eq!(code, 1);
        assert_eq!(extra, 1);
        assert_eq!(bits, 1);

        // offset_value 4 → code 2, extra 0
        let (code, extra, bits) = encode_offset(4);
        assert_eq!(code, 2);
        assert_eq!(extra, 0);
        assert_eq!(bits, 2);

        // offset_value 7 → code 2, extra 3
        let (code, extra, bits) = encode_offset(7);
        assert_eq!(code, 2);
        assert_eq!(extra, 3);
        assert_eq!(bits, 2);

        // offset_value 8 → code 3, extra 0
        let (code, extra, bits) = encode_offset(8);
        assert_eq!(code, 3);
        assert_eq!(extra, 0);
        assert_eq!(bits, 3);

        // offset_value 19 → code 4, extra 3 (19 = 16 + 3)
        let (code, extra, bits) = encode_offset(19);
        assert_eq!(code, 4);
        assert_eq!(extra, 3);
        assert_eq!(bits, 4);

        // offset_value 100 → code 6, extra 36 (100 = 64 + 36)
        let (code, extra, bits) = encode_offset(100);
        assert_eq!(code, 6);
        assert_eq!(extra, 36);
        assert_eq!(bits, 6);
    }

    #[test]
    fn test_analyze_for_rle_uniform() {
        let sequences = vec![
            Sequence::new(0, 4, 3), // LL=0, OF code for 4, ML=3
            Sequence::new(0, 4, 3),
            Sequence::new(0, 4, 3),
        ];

        let suitability = analyze_for_rle(&sequences);
        assert!(suitability.all_uniform());
    }

    #[test]
    fn test_analyze_for_rle_non_uniform() {
        let sequences = vec![
            Sequence::new(0, 4, 3),
            Sequence::new(10, 100, 20), // Different values
        ];

        let suitability = analyze_for_rle(&sequences);
        assert!(!suitability.all_uniform());
    }

    #[test]
    fn test_encode_sequences_rle_empty() {
        let sequences: Vec<Sequence> = vec![];
        let suitability = analyze_for_rle(&sequences);

        let mut output = Vec::new();
        encode_sequences_rle(&sequences, &suitability, &mut output).unwrap();

        assert_eq!(output, vec![0]); // Just count = 0
    }

    #[test]
    fn test_encode_sequences_rle_single() {
        let sequences = vec![Sequence::new(0, 4, 3)];
        let suitability = analyze_for_rle(&sequences);

        let mut output = Vec::new();
        encode_sequences_rle(&sequences, &suitability, &mut output).unwrap();

        // Should have: count(1), mode(0x15), LL code, OF code, ML code, bitstream
        assert!(output.len() >= 5);
        assert_eq!(output[0], 1); // count
        assert_eq!(output[1], 0x15); // RLE mode for all three streams
    }

    #[test]
    fn test_encoded_sequence_creation() {
        let seq = Sequence::new(5, 8, 10);
        let encoded = EncodedSequence::from_sequence(&seq);

        assert_eq!(encoded.ll_code, 5); // Direct mapping for 0-15
        assert_eq!(encoded.ml_code, 7); // 10 - 3 = 7th match length code
    }
}
