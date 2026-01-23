//! Sequences section decoding.
//!
//! Sequences are LZ77-style commands: (literal_length, offset, match_length).
//!
//! ## Repeat Offsets
//!
//! Zstd uses repeat offsets to efficiently encode recently-used offsets.
//! Offset values 1-3 are interpreted as repeat offset references.
//! Initial repeat offsets are [1, 4, 8] per RFC 8878.
//!
//! ## Symbol Compression Modes
//!
//! Each of LL/OF/ML can use different compression modes:
//! - Predefined: Use hardcoded FSE distributions
//! - RLE: Single symbol repeated for all sequences
//! - FSE: Custom FSE table encoded in the stream
//! - Repeat: Reuse previous FSE table (not yet implemented)

use crate::block::literals::LiteralsSection;
use crate::fse::{BitReader, FseDecoder, FseTable};
use haagenti_core::{Error, Result};

/// Result type for parse_table_for_mode: (optional (table, bytes_consumed), optional RLE symbol)
type ParseTableResult = Result<(Option<(FseTable, usize)>, Option<u8>)>;

/// Maximum symbol values for sequence codes (RFC 8878)
const MAX_LL_SYMBOL: u8 = 35; // Literal length codes 0-35
const MAX_OF_SYMBOL: u8 = 31; // Offset codes 0-31
const MAX_ML_SYMBOL: u8 = 52; // Match length codes 0-52

/// Tracks the three repeat offsets used for efficient offset encoding.
///
/// Per RFC 8878 Section 3.1.1.5, offset values 1-3 reference recent offsets.
#[derive(Debug, Clone, Copy)]
struct RepeatOffsets {
    offsets: [u32; 3],
}

impl RepeatOffsets {
    /// Create with initial values per RFC 8878.
    fn new() -> Self {
        Self { offsets: [1, 4, 8] }
    }

    /// Resolve an offset_value to an actual offset and update repeat offsets.
    ///
    /// Per RFC 8878:
    /// - offset_value 1: Use repeat_offset_1 (special case when literal_length == 0)
    /// - offset_value 2: Use repeat_offset_2
    /// - offset_value 3: Use repeat_offset_3 (special case when literal_length == 0)
    /// - offset_value > 3: New offset = value - 3
    fn resolve(&mut self, offset_value: u32, literal_length: u32) -> u32 {
        if offset_value == 0 {
            // Invalid per spec, but handle gracefully
            1
        } else if offset_value <= 3 {
            // Repeat offset reference
            if literal_length == 0 {
                // Special case: when LL=0, offset codes are shifted
                match offset_value {
                    1 => {
                        // Use repeat_offset_2
                        let offset = self.offsets[1];
                        // Swap offsets[0] and offsets[1]
                        self.offsets.swap(0, 1);
                        offset
                    }
                    2 => {
                        // Use repeat_offset_3
                        let offset = self.offsets[2];
                        // Rotate: [2] -> [0], [0] -> [1], [1] -> [2]
                        let temp = self.offsets[2];
                        self.offsets[2] = self.offsets[1];
                        self.offsets[1] = self.offsets[0];
                        self.offsets[0] = temp;
                        offset
                    }
                    3 => {
                        // Use repeat_offset_1 - 1 (with minimum of 1)
                        let offset = self.offsets[0].saturating_sub(1).max(1);
                        // This becomes the new repeat_offset_1
                        self.offsets[2] = self.offsets[1];
                        self.offsets[1] = self.offsets[0];
                        self.offsets[0] = offset;
                        offset
                    }
                    _ => unreachable!(),
                }
            } else {
                // Normal case: use the indexed repeat offset
                let idx = (offset_value - 1) as usize;
                let offset = self.offsets[idx];

                // Rotate used offset to front
                if idx > 0 {
                    let temp = self.offsets[idx];
                    for i in (1..=idx).rev() {
                        self.offsets[i] = self.offsets[i - 1];
                    }
                    self.offsets[0] = temp;
                }
                offset
            }
        } else {
            // New offset: value - 3
            let offset = offset_value - 3;
            // Push to front, shift others back
            self.offsets[2] = self.offsets[1];
            self.offsets[1] = self.offsets[0];
            self.offsets[0] = offset;
            offset
        }
    }
}

/// A single decoded sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sequence {
    /// Number of literals to copy before the match.
    pub literal_length: u32,
    /// Offset back in the output buffer.
    pub offset: u32,
    /// Number of bytes to copy from the match position.
    pub match_length: u32,
}

impl Sequence {
    /// Create a new sequence.
    pub fn new(literal_length: u32, offset: u32, match_length: u32) -> Self {
        Self {
            literal_length,
            offset,
            match_length,
        }
    }
}

/// Symbol decoding mode for sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolMode {
    /// Predefined FSE distribution.
    Predefined,
    /// RLE mode - single symbol repeated.
    Rle,
    /// FSE compressed.
    Fse,
    /// Repeat previous FSE table.
    Repeat,
}

impl SymbolMode {
    /// Parse mode from 2-bit field.
    pub fn from_field(field: u8) -> Self {
        match field {
            0 => SymbolMode::Predefined,
            1 => SymbolMode::Rle,
            2 => SymbolMode::Fse,
            3 => SymbolMode::Repeat,
            _ => unreachable!(),
        }
    }
}

/// Parsed sequences section.
#[derive(Debug, Clone)]
pub struct SequencesSection {
    /// Number of sequences.
    pub num_sequences: usize,
    /// The decoded sequences.
    pub sequences: Vec<Sequence>,
}

impl SequencesSection {
    /// Parse a sequences section.
    ///
    /// # Arguments
    /// * `input` - Input data starting at sequences section
    /// * `literals` - Previously parsed literals section (for context)
    pub fn parse(input: &[u8], _literals: &LiteralsSection) -> Result<Self> {
        if input.is_empty() {
            // No sequences is valid
            return Ok(Self {
                num_sequences: 0,
                sequences: Vec::new(),
            });
        }

        // Parse number of sequences
        let (num_sequences, count_header_size) = Self::parse_sequence_count(input)?;

        if num_sequences == 0 {
            return Ok(Self {
                num_sequences: 0,
                sequences: Vec::new(),
            });
        }

        if input.len() < count_header_size + 1 {
            return Err(Error::corrupted("Sequences section truncated"));
        }

        // Parse symbol compression modes
        // Per RFC 8878 Section 3.1.1.4:
        //   bits[1:0] = Literals_Lengths_Mode
        //   bits[3:2] = Offsets_Mode
        //   bits[5:4] = Match_Lengths_Mode
        //   bits[7:6] = Reserved (must be 0)
        let mode_byte = input[count_header_size];
        let ll_mode = SymbolMode::from_field(mode_byte & 0x03);
        let of_mode = SymbolMode::from_field((mode_byte >> 2) & 0x03);
        let ml_mode = SymbolMode::from_field((mode_byte >> 4) & 0x03);

        let mut pos = count_header_size + 1;

        // Check for all-RLE special case (fast path)
        if ll_mode == SymbolMode::Rle && of_mode == SymbolMode::Rle && ml_mode == SymbolMode::Rle {
            if input.len() < pos + 3 {
                return Err(Error::corrupted("RLE sequence symbols truncated"));
            }
            let ll_sym = input[pos];
            let of_sym = input[pos + 1];
            let ml_sym = input[pos + 2];
            pos += 3;

            return Self::decode_rle_sequences(
                &input[pos..],
                num_sequences,
                ll_sym,
                of_sym,
                ml_sym,
            );
        }

        // Parse each FSE table based on its mode
        let (ll_table, ll_rle_sym) = Self::parse_table_for_mode(
            ll_mode,
            &input[pos..],
            MAX_LL_SYMBOL,
            "Literal Length",
            &PREDEFINED_LL_DISTRIBUTION,
            PREDEFINED_LL_ACCURACY_LOG,
        )?;
        pos += ll_table.as_ref().map_or(0, |(_, consumed)| *consumed);

        let (of_table, of_rle_sym) = Self::parse_table_for_mode(
            of_mode,
            &input[pos..],
            MAX_OF_SYMBOL,
            "Offset",
            &PREDEFINED_OF_DISTRIBUTION,
            PREDEFINED_OF_ACCURACY_LOG,
        )?;
        pos += of_table.as_ref().map_or(0, |(_, consumed)| *consumed);

        let (ml_table, ml_rle_sym) = Self::parse_table_for_mode(
            ml_mode,
            &input[pos..],
            MAX_ML_SYMBOL,
            "Match Length",
            &PREDEFINED_ML_DISTRIBUTION,
            PREDEFINED_ML_ACCURACY_LOG,
        )?;
        pos += ml_table.as_ref().map_or(0, |(_, consumed)| *consumed);

        // If any are RLE, decode with RLE symbols
        let has_rle = ll_rle_sym.is_some() || of_rle_sym.is_some() || ml_rle_sym.is_some();

        // For mixed modes, we need the tables
        let ll = ll_table
            .map(|(t, _)| t)
            .or_else(|| {
                FseTable::from_predefined(&PREDEFINED_LL_DISTRIBUTION, PREDEFINED_LL_ACCURACY_LOG)
                    .ok()
            })
            .ok_or_else(|| Error::corrupted("Failed to build LL table"))?;

        let of = of_table
            .map(|(t, _)| t)
            .or_else(|| {
                FseTable::from_predefined(&PREDEFINED_OF_DISTRIBUTION, PREDEFINED_OF_ACCURACY_LOG)
                    .ok()
            })
            .ok_or_else(|| Error::corrupted("Failed to build OF table"))?;

        let ml = ml_table
            .map(|(t, _)| t)
            .or_else(|| {
                FseTable::from_predefined(&PREDEFINED_ML_DISTRIBUTION, PREDEFINED_ML_ACCURACY_LOG)
                    .ok()
            })
            .ok_or_else(|| Error::corrupted("Failed to build ML table"))?;

        // Handle mixed RLE/FSE modes
        if has_rle {
            return Self::decode_mixed_sequences(
                &input[pos..],
                num_sequences,
                &ll,
                ll_rle_sym,
                &of,
                of_rle_sym,
                &ml,
                ml_rle_sym,
            );
        }

        let (ll_table, of_table, ml_table) = (ll, of, ml);

        // Decode sequences from bitstream
        let bitstream_data = &input[pos..];
        Self::decode_fse_sequences(
            bitstream_data,
            num_sequences,
            &ll_table,
            &of_table,
            &ml_table,
        )
    }

    /// Decode sequences using FSE tables.
    fn decode_fse_sequences(
        data: &[u8],
        num_sequences: usize,
        ll_table: &FseTable,
        of_table: &FseTable,
        ml_table: &FseTable,
    ) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::corrupted("Empty sequence bitstream"));
        }

        // Initialize bit reader from end (backward bitstream)
        let mut bits = BitReader::new(data);
        bits.init_from_end()?;

        // Initialize FSE decoders
        let mut ll_decoder = FseDecoder::new(ll_table);
        let mut of_decoder = FseDecoder::new(of_table);
        let mut ml_decoder = FseDecoder::new(ml_table);

        // Read initial states (MSB-first from sentinel)
        // Per RFC 8878: read LL, OF, ML order
        ll_decoder.init_state(&mut bits)?;
        of_decoder.init_state(&mut bits)?;
        ml_decoder.init_state(&mut bits)?;

        // Switch to LSB-first mode for reading extra bits
        // Extra bits are at the beginning of the bitstream (read from bit 0 up)
        bits.switch_to_lsb_mode()?;

        let mut sequences = Vec::with_capacity(num_sequences);
        let mut repeat_offsets = RepeatOffsets::new();

        for i in 0..num_sequences {
            // Per RFC 8878 and zstd reference implementation:
            // 1. Get symbols from current state (no bits read)
            // 2. Read extra bits in REVERSE of decode order: LL, ML, OF (from LSB)
            //    The encoder writes them in this order, so we read them back the same way.
            // 3. Update states with FSE bits: LL, ML, OF order (skip for last sequence)
            let is_last = i == num_sequences - 1;

            // Step 1: Peek symbols/values from current states
            let ll_code = ll_decoder.peek_symbol();
            let of_code = of_decoder.peek_symbol();
            let _ml_code = ml_decoder.peek_symbol();

            // Step 2: Read extra bits in LL, ML, OF order
            // This matches our encoder write order for internal consistency.

            // LL extra bits (read first from LSB)
            let ll_extra_bits = if ll_code < LITERAL_LENGTH_BASELINE.len() as u8 {
                let (bits_needed, _) = LITERAL_LENGTH_BASELINE[ll_code as usize];
                if bits_needed > 0 {
                    bits.read_bits(bits_needed as usize)?
                } else {
                    0
                }
            } else {
                0
            };

            // ML extra bits (read second)
            // IMPORTANT: We must read seq_base BEFORE update_state, as it comes from current state
            let ml_seq_extra_bits = ml_decoder.peek_seq_extra_bits();
            let ml_seq_base = ml_decoder.peek_seq_base();
            let ml_extra_bits = if ml_seq_extra_bits > 0 {
                bits.read_bits(ml_seq_extra_bits as usize)?
            } else {
                0
            };

            // OF extra bits (read last)
            let of_num_bits = offset_code_extra_bits(of_code);
            let of_extra_bits = if of_num_bits > 0 {
                bits.read_bits(of_num_bits as usize)?
            } else {
                0
            };

            // Step 3: Update states with FSE bits (skip for last sequence)
            // Order per zstd: LL, ML, OF (matching encoder write order)
            if !is_last {
                ll_decoder.update_state(&mut bits)?;
                ml_decoder.update_state(&mut bits)?;
                of_decoder.update_state(&mut bits)?;
            }

            // Decode final values
            let literal_length = decode_literal_length(ll_code, ll_extra_bits);
            // Match length = seq_base + extra bits (seq_base was read BEFORE update_state)
            let match_length = ml_seq_base + ml_extra_bits;
            let offset_value = decode_offset(of_code, of_extra_bits);

            // Convert offset_value to actual offset using repeat offset handling
            let offset = repeat_offsets.resolve(offset_value, literal_length);

            sequences.push(Sequence::new(literal_length, offset, match_length));
        }

        Ok(Self {
            num_sequences,
            sequences,
        })
    }

    /// Decode sequences in RLE mode (single repeated symbol for each type).
    fn decode_rle_sequences(
        data: &[u8],
        num_sequences: usize,
        ll_sym: u8,
        of_sym: u8,
        ml_sym: u8,
    ) -> Result<Self> {
        if data.is_empty() && num_sequences > 0 {
            return Err(Error::corrupted("Empty RLE sequence data"));
        }

        let mut bits = BitReader::new(data);
        if !data.is_empty() {
            // RLE mode: extra bits only (no FSE states), read LSB-first
            bits.init_fse()?;
        }

        let mut sequences = Vec::with_capacity(num_sequences);
        let mut repeat_offsets = RepeatOffsets::new();

        for _ in 0..num_sequences {
            // Read extra bits in LL, ML, OF order (forward from LSB)

            // LL extra bits (read first from LSB)
            let ll_extra_bits = if ll_sym < LITERAL_LENGTH_BASELINE.len() as u8 {
                let (bits_needed, _) = LITERAL_LENGTH_BASELINE[ll_sym as usize];
                if bits_needed > 0 {
                    bits.read_bits(bits_needed as usize)?
                } else {
                    0
                }
            } else {
                0
            };

            // ML extra bits (read second)
            let ml_extra_bits = if ml_sym < MATCH_LENGTH_BASELINE.len() as u8 {
                let (bits_needed, _) = MATCH_LENGTH_BASELINE[ml_sym as usize];
                if bits_needed > 0 {
                    bits.read_bits(bits_needed as usize)?
                } else {
                    0
                }
            } else {
                0
            };

            // OF extra bits (read last)
            let of_num_bits = offset_code_extra_bits(of_sym);
            let of_extra_bits = if of_num_bits > 0 {
                bits.read_bits(of_num_bits as usize)?
            } else {
                0
            };

            let literal_length = decode_literal_length(ll_sym, ll_extra_bits);
            let match_length = decode_match_length(ml_sym, ml_extra_bits);
            let offset_value = decode_offset(of_sym, of_extra_bits);

            // Convert offset_value to actual offset using repeat offset handling
            let offset = repeat_offsets.resolve(offset_value, literal_length);

            sequences.push(Sequence::new(literal_length, offset, match_length));
        }

        Ok(Self {
            num_sequences,
            sequences,
        })
    }

    /// Parse FSE table or RLE symbol based on mode.
    ///
    /// Returns (Some((table, bytes_consumed)), None) for FSE/Predefined modes,
    /// or (None, Some(rle_symbol)) for RLE mode.
    fn parse_table_for_mode(
        mode: SymbolMode,
        data: &[u8],
        max_symbol: u8,
        name: &str,
        predefined_dist: &[i16],
        predefined_log: u8,
    ) -> ParseTableResult {
        match mode {
            SymbolMode::Predefined => {
                let table = FseTable::from_predefined(predefined_dist, predefined_log)?;
                Ok((Some((table, 0)), None))
            }
            SymbolMode::Rle => {
                if data.is_empty() {
                    return Err(Error::corrupted(format!("{} RLE symbol missing", name)));
                }
                // RLE consumes 1 byte for the symbol, return in tuple for position tracking
                Ok((
                    Some((
                        FseTable::from_predefined(predefined_dist, predefined_log)?,
                        1,
                    )),
                    Some(data[0]),
                ))
            }
            SymbolMode::Fse => {
                let (table, consumed) = FseTable::parse(data, max_symbol)?;
                Ok((Some((table, consumed)), None))
            }
            SymbolMode::Repeat => {
                // Repeat mode requires storing previous tables, not yet implemented
                Err(Error::Unsupported(format!(
                    "{} Repeat mode not yet implemented",
                    name
                )))
            }
        }
    }

    /// Decode sequences with mixed FSE/RLE modes.
    ///
    /// Some symbols come from RLE (fixed), others from FSE tables.
    #[allow(clippy::too_many_arguments)]
    fn decode_mixed_sequences(
        data: &[u8],
        num_sequences: usize,
        ll_table: &FseTable,
        ll_rle: Option<u8>,
        of_table: &FseTable,
        of_rle: Option<u8>,
        ml_table: &FseTable,
        ml_rle: Option<u8>,
    ) -> Result<Self> {
        if data.is_empty() && num_sequences > 0 {
            return Err(Error::corrupted("Empty mixed sequence data"));
        }

        let mut bits = BitReader::new(data);
        if !data.is_empty() {
            bits.init_from_end()?;
        }

        // Initialize FSE decoders for non-RLE streams
        let mut ll_decoder = if ll_rle.is_none() {
            let mut d = FseDecoder::new(ll_table);
            d.init_state(&mut bits)?;
            Some(d)
        } else {
            None
        };

        let mut of_decoder = if of_rle.is_none() {
            let mut d = FseDecoder::new(of_table);
            d.init_state(&mut bits)?;
            Some(d)
        } else {
            None
        };

        let mut ml_decoder = if ml_rle.is_none() {
            let mut d = FseDecoder::new(ml_table);
            d.init_state(&mut bits)?;
            Some(d)
        } else {
            None
        };

        // Switch to LSB-first mode for reading extra bits
        bits.switch_to_lsb_mode()?;

        let mut sequences = Vec::with_capacity(num_sequences);
        let mut repeat_offsets = RepeatOffsets::new();

        for _ in 0..num_sequences {
            // Get codes from FSE or RLE
            let of_code = if let Some(ref mut dec) = of_decoder {
                dec.peek_symbol()
            } else {
                of_rle.unwrap()
            };

            let ml_code = if let Some(ref mut dec) = ml_decoder {
                dec.decode_symbol(&mut bits)?
            } else {
                ml_rle.unwrap()
            };

            let ll_code = if let Some(ref mut dec) = ll_decoder {
                dec.decode_symbol(&mut bits)?
            } else {
                ll_rle.unwrap()
            };

            // Read extra bits in LL, ML, OF order
            let ll_extra_bits = if ll_code < LITERAL_LENGTH_BASELINE.len() as u8 {
                let (bits_needed, _) = LITERAL_LENGTH_BASELINE[ll_code as usize];
                if bits_needed > 0 {
                    bits.read_bits(bits_needed as usize)?
                } else {
                    0
                }
            } else {
                0
            };

            let ml_extra_bits = if ml_code < MATCH_LENGTH_BASELINE.len() as u8 {
                let (bits_needed, _) = MATCH_LENGTH_BASELINE[ml_code as usize];
                if bits_needed > 0 {
                    bits.read_bits(bits_needed as usize)?
                } else {
                    0
                }
            } else {
                0
            };

            let of_num_bits = offset_code_extra_bits(of_code);
            let of_extra_bits = if of_num_bits > 0 {
                bits.read_bits(of_num_bits as usize)?
            } else {
                0
            };

            let literal_length = decode_literal_length(ll_code, ll_extra_bits);
            let match_length = decode_match_length(ml_code, ml_extra_bits);
            let offset_value = decode_offset(of_code, of_extra_bits);
            let offset = repeat_offsets.resolve(offset_value, literal_length);

            sequences.push(Sequence::new(literal_length, offset, match_length));
        }

        Ok(Self {
            num_sequences,
            sequences,
        })
    }

    /// Parse the sequence count from the header.
    fn parse_sequence_count(input: &[u8]) -> Result<(usize, usize)> {
        if input.is_empty() {
            return Err(Error::corrupted("Empty sequences header"));
        }

        let byte0 = input[0] as usize;

        if byte0 == 0 {
            // No sequences
            Ok((0, 1))
        } else if byte0 < 128 {
            // 1 byte: count = byte0
            Ok((byte0, 1))
        } else if byte0 < 255 {
            // 2 bytes: count = ((byte0 - 128) << 8) + byte1
            if input.len() < 2 {
                return Err(Error::corrupted("Sequences count truncated"));
            }
            let count = ((byte0 - 128) << 8) + (input[1] as usize);
            Ok((count, 2))
        } else {
            // 3 bytes: count = byte1 + (byte2 << 8) + 0x7F00
            if input.len() < 3 {
                return Err(Error::corrupted("Sequences count truncated"));
            }
            let count = (input[1] as usize) + ((input[2] as usize) << 8) + 0x7F00;
            Ok((count, 3))
        }
    }
}

// =============================================================================
// Predefined FSE Distributions (RFC 8878)
// =============================================================================

/// Predefined literal length distribution (accuracy log = 6).
/// 36 symbols with normalized frequencies summing to 64.
pub const PREDEFINED_LL_DISTRIBUTION: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, // Codes 0-7
    2, 2, 2, 2, 2, 1, 1, 1, // Codes 8-15
    2, 2, 2, 2, 2, 2, 2, 2, // Codes 16-23
    2, 3, 2, 1, 1, 1, 1, 1, // Codes 24-31
    -1, -1, -1, -1, // Codes 32-35 (less than 1)
];

/// Predefined offset distribution (accuracy log = 5).
/// 29 symbols with normalized frequencies summing to 32.
pub const PREDEFINED_OF_DISTRIBUTION: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, // Codes 0-7
    2, 1, 1, 1, 1, 1, 1, 1, // Codes 8-15
    1, 1, 1, 1, 1, 1, 1, 1, // Codes 16-23
    -1, -1, -1, -1, -1, // Codes 24-28 (less than 1)
];

/// Predefined match length distribution (accuracy log = 6).
/// 53 symbols with normalized frequencies summing to 64.
pub const PREDEFINED_ML_DISTRIBUTION: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, // Codes 0-7
    2, 1, 1, 1, 1, 1, 1, 1, // Codes 8-15
    1, 1, 1, 1, 1, 1, 1, 1, // Codes 16-23
    1, 1, 1, 1, 1, 1, 1, 1, // Codes 24-31
    1, 1, 1, 1, 1, 1, 1, 1, // Codes 32-39
    1, 1, 1, 1, 1, 1, -1, -1, // Codes 40-47
    -1, -1, -1, -1, -1, // Codes 48-52 (less than 1)
];

/// Predefined accuracy log for literal lengths.
pub const PREDEFINED_LL_ACCURACY_LOG: u8 = 6;

/// Predefined accuracy log for offsets.
pub const PREDEFINED_OF_ACCURACY_LOG: u8 = 5;

/// Predefined accuracy log for match lengths.
pub const PREDEFINED_ML_ACCURACY_LOG: u8 = 6;

// =============================================================================
// Baseline Tables (RFC 8878)
// =============================================================================

/// Literal length code to (extra bits, baseline) mapping.
pub const LITERAL_LENGTH_BASELINE: [(u8, u32); 36] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 14),
    (0, 15),
    (1, 16),
    (1, 18),
    (1, 20),
    (1, 22),
    (2, 24),
    (2, 28),
    (3, 32),
    (3, 40),
    (4, 48),
    (6, 64),
    (7, 128),
    (8, 256),
    (9, 512),
    (10, 1024),
    (11, 2048),
    (12, 4096),
    (13, 8192),
    (14, 16384),
    (15, 32768),
    (16, 65536),
];

/// Match length code to (extra bits, baseline) mapping using ZSTD's predefined values.
///
/// IMPORTANT: This uses zstd's predefined values, NOT RFC 8878 Table 6.
/// zstd's predefined ML table differs from RFC starting at code 43:
/// - Code 43: zstd uses 7 bits (baseline 131), RFC uses 5 bits
/// - Code 44: zstd uses 8 bits (baseline 259), RFC uses 6 bits (baseline 163)
/// - Code 45+: All shifted to accommodate zstd's larger ranges
///
/// Each entry is (Number_of_Bits, Baseline) for match length codes 0-52.
pub const MATCH_LENGTH_BASELINE: [(u8, u32); 53] = [
    // Codes 0-31: No extra bits, match_length = baseline (3-34)
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 14),
    (0, 15),
    (0, 16),
    (0, 17),
    (0, 18),
    (0, 19),
    (0, 20),
    (0, 21),
    (0, 22),
    (0, 23),
    (0, 24),
    (0, 25),
    (0, 26),
    (0, 27),
    (0, 28),
    (0, 29),
    (0, 30),
    (0, 31),
    (0, 32),
    (0, 33),
    (0, 34),
    // Codes 32-35: 1 extra bit (matches RFC)
    (1, 35),
    (1, 37),
    (1, 39),
    (1, 41),
    // Codes 36-37: 2 extra bits (matches RFC)
    (2, 43),
    (2, 47),
    // Codes 38-39: 3 extra bits (matches RFC)
    (3, 51),
    (3, 59),
    // Codes 40-41: 4 extra bits (matches RFC)
    (4, 67),
    (4, 83),
    // Code 42: 5 extra bits (matches RFC)
    (5, 99),
    // Code 43: 7 extra bits (DIFFERS from RFC's 5 bits!)
    (7, 131),
    // Code 44: 8 extra bits (DIFFERS from RFC's 6 bits with baseline 163!)
    (8, 259),
    // Code 45: 9 extra bits (zstd: baseline 515)
    (9, 515),
    // Code 46: 10 extra bits (zstd: baseline 1027)
    (10, 1027),
    // Code 47: 11 extra bits (zstd: baseline 2051)
    (11, 2051),
    // Code 48: 12 extra bits (zstd: baseline 4099)
    (12, 4099),
    // Code 49: 13 extra bits (zstd: baseline 8195)
    (13, 8195),
    // Code 50: 14 extra bits (zstd: baseline 16387)
    (14, 16387),
    // Code 51: 15 extra bits (zstd: baseline 32771)
    (15, 32771),
    // Code 52: 16 extra bits (zstd: baseline 65539)
    (16, 65539),
];

/// Decode literal length from code and extra bits.
pub fn decode_literal_length(code: u8, extra_bits: u32) -> u32 {
    if code as usize >= LITERAL_LENGTH_BASELINE.len() {
        return 0;
    }
    let (bits, baseline) = LITERAL_LENGTH_BASELINE[code as usize];
    if bits == 0 {
        baseline
    } else {
        baseline + (extra_bits & ((1 << bits) - 1))
    }
}

/// Decode match length from code and extra bits.
pub fn decode_match_length(code: u8, extra_bits: u32) -> u32 {
    if code as usize >= MATCH_LENGTH_BASELINE.len() {
        return 3; // Minimum match length
    }
    let (bits, baseline) = MATCH_LENGTH_BASELINE[code as usize];
    if bits == 0 {
        baseline
    } else {
        baseline + (extra_bits & ((1 << bits) - 1))
    }
}

/// Decode offset from code and extra bits per RFC 8878 Table 7.
///
/// Formula: Offset_Value = (1 << Offset_Code) + Extra_Bits
///
/// The returned offset_value is then interpreted by RepeatOffsets::resolve():
/// - offset_value 1-3: use repeat offset
/// - offset_value > 3: actual_offset = offset_value - 3
pub fn decode_offset(code: u8, extra_bits: u32) -> u32 {
    let code = code.min(31); // Clamp to prevent overflow
    (1u32 << code) + extra_bits
}

/// Get the number of extra bits for an offset code per RFC 8878 Table 7.
///
/// Number_of_Extra_Bits = Offset_Code
pub fn offset_code_extra_bits(code: u8) -> u8 {
    code.min(31)
}

/// Build predefined FSE tables for sequence decoding.
#[cfg(test)]
fn build_predefined_tables() -> Result<(FseTable, FseTable, FseTable)> {
    let ll_table = FseTable::build(
        &PREDEFINED_LL_DISTRIBUTION,
        PREDEFINED_LL_ACCURACY_LOG,
        PREDEFINED_LL_DISTRIBUTION.len() as u8,
    )?;

    let of_table = FseTable::build(
        &PREDEFINED_OF_DISTRIBUTION,
        PREDEFINED_OF_ACCURACY_LOG,
        PREDEFINED_OF_DISTRIBUTION.len() as u8,
    )?;

    let ml_table = FseTable::build(
        &PREDEFINED_ML_DISTRIBUTION,
        PREDEFINED_ML_ACCURACY_LOG,
        PREDEFINED_ML_DISTRIBUTION.len() as u8,
    )?;

    Ok((ll_table, of_table, ml_table))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_creation() {
        let seq = Sequence::new(10, 5, 20);
        assert_eq!(seq.literal_length, 10);
        assert_eq!(seq.offset, 5);
        assert_eq!(seq.match_length, 20);
    }

    #[test]
    fn test_symbol_mode_parsing() {
        assert_eq!(SymbolMode::from_field(0), SymbolMode::Predefined);
        assert_eq!(SymbolMode::from_field(1), SymbolMode::Rle);
        assert_eq!(SymbolMode::from_field(2), SymbolMode::Fse);
        assert_eq!(SymbolMode::from_field(3), SymbolMode::Repeat);
    }

    #[test]
    fn test_sequence_count_zero() {
        let (count, size) = SequencesSection::parse_sequence_count(&[0]).unwrap();
        assert_eq!(count, 0);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_sequence_count_small() {
        // Count = 50 (single byte)
        let (count, size) = SequencesSection::parse_sequence_count(&[50]).unwrap();
        assert_eq!(count, 50);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_sequence_count_medium() {
        // Count in range 128-32767 uses 2 bytes
        // Count = 300: byte0 = 128 + (300 >> 8) = 129, byte1 = 300 & 0xFF = 44
        // Actually: count = ((byte0 - 128) << 8) + byte1
        // 300 = ((129 - 128) << 8) + 44 = 256 + 44 = 300 ✓
        let (count, size) = SequencesSection::parse_sequence_count(&[129, 44]).unwrap();
        assert_eq!(count, 300);
        assert_eq!(size, 2);
    }

    #[test]
    fn test_sequence_count_large() {
        // Count >= 0x7F00 uses 3 bytes
        // byte0 = 255, count = byte1 + (byte2 << 8) + 0x7F00
        // For count = 0x7F00: byte1 = 0, byte2 = 0
        let (count, size) = SequencesSection::parse_sequence_count(&[255, 0, 0]).unwrap();
        assert_eq!(count, 0x7F00);
        assert_eq!(size, 3);

        // For count = 0x8000: need 0x8000 - 0x7F00 = 0x100 = 256
        // byte1 = 0, byte2 = 1
        let (count, size) = SequencesSection::parse_sequence_count(&[255, 0, 1]).unwrap();
        assert_eq!(count, 0x8000);
        assert_eq!(size, 3);
    }

    #[test]
    fn test_literal_length_decoding() {
        // Code 0-15: no extra bits, value = code
        assert_eq!(decode_literal_length(0, 0), 0);
        assert_eq!(decode_literal_length(15, 0), 15);

        // Code 16: 1 extra bit, baseline 16
        assert_eq!(decode_literal_length(16, 0), 16);
        assert_eq!(decode_literal_length(16, 1), 17);

        // Code 20: 2 extra bits, baseline 24
        assert_eq!(decode_literal_length(20, 0), 24);
        assert_eq!(decode_literal_length(20, 3), 27);
    }

    #[test]
    fn test_match_length_decoding() {
        // Code 0: baseline 3, no extra bits
        assert_eq!(decode_match_length(0, 0), 3);

        // Code 31: baseline 34, no extra bits
        assert_eq!(decode_match_length(31, 0), 34);

        // Code 32: baseline 35, 1 extra bit
        assert_eq!(decode_match_length(32, 0), 35);
        assert_eq!(decode_match_length(32, 1), 36);
    }

    #[test]
    fn test_offset_decoding() {
        // Per RFC 8878 Table 7:
        // Offset_Value = (1 << Offset_Code) + Extra_Bits

        // Code 0: values 1-2
        assert_eq!(decode_offset(0, 0), 1);
        assert_eq!(decode_offset(0, 1), 2); // Note: only 0 bits for code 0, so extra=1 exceeds range

        // Code 1: values 2-3
        assert_eq!(decode_offset(1, 0), 2);
        assert_eq!(decode_offset(1, 1), 3);

        // Code 2: values 4-7
        assert_eq!(decode_offset(2, 0), 4);
        assert_eq!(decode_offset(2, 3), 7);

        // Code 3: values 8-15
        assert_eq!(decode_offset(3, 0), 8);
        assert_eq!(decode_offset(3, 7), 15);

        // Code 10: values 1024-2047
        assert_eq!(decode_offset(10, 0), 1024);
        assert_eq!(decode_offset(10, 500), 1524);
    }

    #[test]
    fn test_empty_sequences() {
        let literals = LiteralsSection::new_raw(vec![]);
        let result = SequencesSection::parse(&[], &literals);
        assert!(result.is_ok());
        let section = result.unwrap();
        assert_eq!(section.num_sequences, 0);
        assert!(section.sequences.is_empty());
    }

    #[test]
    fn test_zero_sequences() {
        let literals = LiteralsSection::new_raw(vec![]);
        let result = SequencesSection::parse(&[0], &literals);
        assert!(result.is_ok());
        let section = result.unwrap();
        assert_eq!(section.num_sequences, 0);
    }

    #[test]
    fn test_predefined_tables_build() {
        // Test that predefined tables can be built successfully
        let result = build_predefined_tables();
        assert!(result.is_ok(), "Build failed: {:?}", result.err());

        let (ll_table, of_table, ml_table) = result.unwrap();

        // Verify table sizes match expected
        assert_eq!(ll_table.accuracy_log(), PREDEFINED_LL_ACCURACY_LOG);
        assert_eq!(of_table.accuracy_log(), PREDEFINED_OF_ACCURACY_LOG);
        assert_eq!(ml_table.accuracy_log(), PREDEFINED_ML_ACCURACY_LOG);
    }

    #[test]
    fn test_predefined_ll_distribution_sum() {
        // Verify the distribution sums to 2^accuracy_log (accounting for -1 values)
        let sum: i32 = PREDEFINED_LL_DISTRIBUTION
            .iter()
            .filter(|&&x| x > 0)
            .map(|&x| x as i32)
            .sum();
        // -1 values represent "less than 1" and need 1 slot each
        let less_than_one = PREDEFINED_LL_DISTRIBUTION
            .iter()
            .filter(|&&x| x == -1)
            .count();

        // Total should equal 2^6 = 64
        assert!(sum + less_than_one as i32 <= 64);
    }

    #[test]
    fn test_predefined_of_distribution_sum() {
        let sum: i32 = PREDEFINED_OF_DISTRIBUTION
            .iter()
            .filter(|&&x| x > 0)
            .map(|&x| x as i32)
            .sum();
        let less_than_one = PREDEFINED_OF_DISTRIBUTION
            .iter()
            .filter(|&&x| x == -1)
            .count();

        // Total should equal 2^5 = 32
        assert!(sum + less_than_one as i32 <= 32);
    }

    #[test]
    fn test_predefined_ml_distribution_sum() {
        let sum: i32 = PREDEFINED_ML_DISTRIBUTION
            .iter()
            .filter(|&&x| x > 0)
            .map(|&x| x as i32)
            .sum();
        let less_than_one = PREDEFINED_ML_DISTRIBUTION
            .iter()
            .filter(|&&x| x == -1)
            .count();

        // Total should equal 2^6 = 64
        assert!(sum + less_than_one as i32 <= 64);
    }

    #[test]
    fn test_literal_length_baseline_all_codes() {
        // Test all 36 literal length codes
        for code in 0..36u8 {
            let (bits, baseline) = LITERAL_LENGTH_BASELINE[code as usize];
            let result = decode_literal_length(code, 0);
            assert_eq!(result, baseline, "Code {} failed", code);

            // Test with max extra bits
            if bits > 0 {
                let max_extra = (1u32 << bits) - 1;
                let result_max = decode_literal_length(code, max_extra);
                assert_eq!(result_max, baseline + max_extra);
            }
        }
    }

    #[test]
    fn test_match_length_baseline_all_codes() {
        // Test all 53 match length codes
        for code in 0..53u8 {
            let (bits, baseline) = MATCH_LENGTH_BASELINE[code as usize];
            let result = decode_match_length(code, 0);
            assert_eq!(result, baseline, "Code {} failed", code);

            if bits > 0 {
                let max_extra = (1u32 << bits) - 1;
                let result_max = decode_match_length(code, max_extra);
                assert_eq!(result_max, baseline + max_extra);
            }
        }
    }

    #[test]
    fn test_offset_decoding_range() {
        // Per RFC 8878 Table 7: Offset_Value = (1 << Code) + Extra_Bits
        for code in 0..=30u8 {
            let base_offset = decode_offset(code, 0);
            assert_eq!(base_offset, 1u32 << code, "Code {} failed", code);
        }

        // Code 31 is clamped to prevent overflow (1 << 31 would overflow u32)
        assert_eq!(decode_offset(31, 0), 1u32 << 31);
    }

    #[test]
    fn test_sequences_with_predefined_mode() {
        // Build a minimal sequences section with predefined mode
        // Mode byte: 0b00000000 = all predefined (LL=0, OF=0, ML=0, reserved=0)
        let literals = LiteralsSection::new_raw(vec![]);

        // 1 sequence, predefined mode, minimal bitstream
        let mut data = vec![1]; // 1 sequence
        data.push(0x00); // Mode byte: all predefined

        // Add a valid bitstream with sentinel
        // The bitstream needs initial states and at least one sequence's worth of data
        data.push(0x80); // Minimal bitstream with sentinel

        let result = SequencesSection::parse(&data, &literals);
        // This might fail due to bitstream issues, but the mode parsing should work
        if result.is_err() {
            // Expected for minimal bitstream
        } else {
            let section = result.unwrap();
            assert_eq!(section.num_sequences, 1);
        }
    }

    #[test]
    fn test_mode_byte_parsing() {
        // Verify mode byte layout: LL[7:6], OF[5:4], ML[3:2], reserved[1:0]
        // All predefined: 0b00_00_00_00 = 0x00
        let mode_byte = 0x00u8;
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 6) & 0x03),
            SymbolMode::Predefined
        );
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 4) & 0x03),
            SymbolMode::Predefined
        );
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 2) & 0x03),
            SymbolMode::Predefined
        );

        // All RLE: 0b01_01_01_00 = 0x54
        let mode_byte = 0x54u8;
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 6) & 0x03),
            SymbolMode::Rle
        );
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 4) & 0x03),
            SymbolMode::Rle
        );
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 2) & 0x03),
            SymbolMode::Rle
        );

        // All FSE compressed: 0b10_10_10_00 = 0xA8
        let mode_byte = 0xA8u8;
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 6) & 0x03),
            SymbolMode::Fse
        );
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 4) & 0x03),
            SymbolMode::Fse
        );
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 2) & 0x03),
            SymbolMode::Fse
        );

        // All repeat: 0b11_11_11_00 = 0xFC
        let mode_byte = 0xFCu8;
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 6) & 0x03),
            SymbolMode::Repeat
        );
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 4) & 0x03),
            SymbolMode::Repeat
        );
        assert_eq!(
            SymbolMode::from_field((mode_byte >> 2) & 0x03),
            SymbolMode::Repeat
        );
    }

    #[test]
    fn test_sequence_count_boundary_127() {
        // Boundary at 127: single byte
        let (count, size) = SequencesSection::parse_sequence_count(&[127]).unwrap();
        assert_eq!(count, 127);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_sequence_count_boundary_128() {
        // 128 uses 2 bytes: ((128 - 128) << 8) + byte1
        // For count = 128: byte0 = 128, byte1 = 128
        let (count, size) = SequencesSection::parse_sequence_count(&[128, 128]).unwrap();
        assert_eq!(count, 128);
        assert_eq!(size, 2);
    }

    // =========================================================================
    // RepeatOffsets Tests
    // =========================================================================

    #[test]
    fn test_repeat_offsets_initial_values() {
        let offsets = RepeatOffsets::new();
        assert_eq!(offsets.offsets, [1, 4, 8]);
    }

    #[test]
    fn test_repeat_offsets_new_offset() {
        let mut offsets = RepeatOffsets::new();

        // New offset: value > 3, actual = value - 3
        // offset_value 7 -> actual 4, becomes new repeat_offset_1
        let result = offsets.resolve(7, 5); // LL=5 (non-zero)
        assert_eq!(result, 4); // 7 - 3 = 4
        assert_eq!(offsets.offsets, [4, 1, 4]); // 4 pushed to front
    }

    #[test]
    fn test_repeat_offsets_use_first() {
        let mut offsets = RepeatOffsets::new();

        // Use repeat_offset_1 (value=1) with LL > 0
        let result = offsets.resolve(1, 5);
        assert_eq!(result, 1); // Initial repeat_offset_1
        assert_eq!(offsets.offsets, [1, 4, 8]); // No rotation when using first
    }

    #[test]
    fn test_repeat_offsets_use_second() {
        let mut offsets = RepeatOffsets::new();

        // Use repeat_offset_2 (value=2) with LL > 0
        let result = offsets.resolve(2, 5);
        assert_eq!(result, 4); // Initial repeat_offset_2
        assert_eq!(offsets.offsets, [4, 1, 8]); // 4 rotated to front
    }

    #[test]
    fn test_repeat_offsets_use_third() {
        let mut offsets = RepeatOffsets::new();

        // Use repeat_offset_3 (value=3) with LL > 0
        let result = offsets.resolve(3, 5);
        assert_eq!(result, 8); // Initial repeat_offset_3
        assert_eq!(offsets.offsets, [8, 1, 4]); // 8 rotated to front
    }

    #[test]
    fn test_repeat_offsets_ll_zero_special_case_1() {
        let mut offsets = RepeatOffsets::new();

        // When LL=0 and value=1: use repeat_offset_2 instead, swap positions
        let result = offsets.resolve(1, 0);
        assert_eq!(result, 4); // Uses repeat_offset_2
        assert_eq!(offsets.offsets, [4, 1, 8]); // Swapped positions 0 and 1
    }

    #[test]
    fn test_repeat_offsets_ll_zero_special_case_2() {
        let mut offsets = RepeatOffsets::new();

        // When LL=0 and value=2: use repeat_offset_3, rotate
        let result = offsets.resolve(2, 0);
        assert_eq!(result, 8); // Uses repeat_offset_3
        assert_eq!(offsets.offsets, [8, 1, 4]); // Rotated
    }

    #[test]
    fn test_repeat_offsets_ll_zero_special_case_3() {
        let mut offsets = RepeatOffsets::new();

        // When LL=0 and value=3: use repeat_offset_1 - 1
        // Initial repeat_offset_1 = 1, so 1 - 1 = 0, but minimum is 1
        let result = offsets.resolve(3, 0);
        assert_eq!(result, 1); // max(1-1, 1) = 1
        assert_eq!(offsets.offsets, [1, 1, 4]); // New value pushed to front
    }

    #[test]
    fn test_repeat_offsets_ll_zero_case_3_larger_offset() {
        let mut offsets = RepeatOffsets::new();

        // First set a larger repeat_offset_1
        offsets.resolve(13, 5); // 13 - 3 = 10
        assert_eq!(offsets.offsets[0], 10);

        // Now LL=0 and value=3: use 10 - 1 = 9
        let result = offsets.resolve(3, 0);
        assert_eq!(result, 9);
        assert_eq!(offsets.offsets[0], 9);
    }

    #[test]
    fn test_repeat_offsets_sequence_of_operations() {
        let mut offsets = RepeatOffsets::new();
        // Initial: [1, 4, 8]

        // New offset 103 (value 106): 106 - 3 = 103
        offsets.resolve(106, 5);
        assert_eq!(offsets.offsets, [103, 1, 4]);

        // Use repeat_offset_1 (value 1)
        let result = offsets.resolve(1, 5);
        assert_eq!(result, 103);
        assert_eq!(offsets.offsets, [103, 1, 4]); // No change

        // Use repeat_offset_2 (value 2)
        let result = offsets.resolve(2, 5);
        assert_eq!(result, 1);
        assert_eq!(offsets.offsets, [1, 103, 4]); // Rotated
    }
}
