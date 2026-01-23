//! Finite State Entropy (FSE) coding.
//!
//! FSE is the core entropy coder used throughout Zstandard. It's a variant of
//! ANS (Asymmetric Numeral Systems) that provides near-optimal compression
//! with fast decoding.
//!
//! ## Overview
//!
//! FSE encodes symbols using a state machine. The current state encodes
//! information about previously seen symbols, allowing efficient compression
//! of correlated data.
//!
//! ## Performance
//!
//! The predefined FSE tables are cached using `OnceLock` for zero-overhead
//! access after first use. This eliminates table rebuilding on every block.
//!
//! ## References
//!
//! - [RFC 8878 Section 4.1](https://datatracker.ietf.org/doc/html/rfc8878#section-4.1)
//! - [FSE Educational Decoder](https://github.com/facebook/zstd/blob/dev/doc/educational_decoder.md)

mod decoder;
mod encoder;
mod table;
mod tans_encoder;

use std::sync::OnceLock;

pub use decoder::{BitReader, FseDecoder};
pub use encoder::{FseBitWriter, FseEncoder, InterleavedFseEncoder};
pub use table::{
    FseTable, FseTableEntry, FSE_MAX_ACCURACY_LOG, LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_DEFAULT_DISTRIBUTION,
};
pub use tans_encoder::{InterleavedTansEncoder, TansEncoder, TansSymbolParams};

// =============================================================================
// Static Cached Predefined FSE Tables
// =============================================================================
//
// These tables are built once on first access and cached for the lifetime of
// the program. This eliminates the overhead of rebuilding tables on every block.

static CACHED_LL_TABLE: OnceLock<FseTable> = OnceLock::new();
static CACHED_OF_TABLE: OnceLock<FseTable> = OnceLock::new();
static CACHED_ML_TABLE: OnceLock<FseTable> = OnceLock::new();

/// Get the cached predefined Literal Length FSE table.
///
/// The table is built once on first access and cached for subsequent calls.
/// This provides zero-overhead access to the predefined table.
#[inline]
pub fn cached_ll_table() -> &'static FseTable {
    CACHED_LL_TABLE.get_or_init(|| {
        FseTable::from_hardcoded_ll().expect("LL predefined table construction should never fail")
    })
}

/// Get the cached predefined Offset FSE table.
///
/// The table is built once on first access and cached for subsequent calls.
#[inline]
pub fn cached_of_table() -> &'static FseTable {
    CACHED_OF_TABLE.get_or_init(|| {
        FseTable::from_hardcoded_of().expect("OF predefined table construction should never fail")
    })
}

/// Get the cached predefined Match Length FSE table.
///
/// The table is built once on first access and cached for subsequent calls.
#[inline]
pub fn cached_ml_table() -> &'static FseTable {
    CACHED_ML_TABLE.get_or_init(|| {
        FseTable::from_hardcoded_ml().expect("ML predefined table construction should never fail")
    })
}

// =============================================================================
// Static Cached tANS Encoders
// =============================================================================
//
// These encoders are built once from the cached FSE tables and reused.
// Since encoders have mutable state, we cache "template" encoders that
// can be cloned cheaply for each encoding session.

static CACHED_LL_ENCODER: OnceLock<TansEncoder> = OnceLock::new();
static CACHED_OF_ENCODER: OnceLock<TansEncoder> = OnceLock::new();
static CACHED_ML_ENCODER: OnceLock<TansEncoder> = OnceLock::new();

/// Get a clone of the cached Literal Length tANS encoder.
///
/// Cloning is cheap (just Vec clones) compared to building from scratch.
#[inline]
pub fn cloned_ll_encoder() -> TansEncoder {
    CACHED_LL_ENCODER
        .get_or_init(|| TansEncoder::from_decode_table(cached_ll_table()))
        .clone()
}

/// Get a clone of the cached Offset tANS encoder.
#[inline]
pub fn cloned_of_encoder() -> TansEncoder {
    CACHED_OF_ENCODER
        .get_or_init(|| TansEncoder::from_decode_table(cached_of_table()))
        .clone()
}

/// Get a clone of the cached Match Length tANS encoder.
#[inline]
pub fn cloned_ml_encoder() -> TansEncoder {
    CACHED_ML_ENCODER
        .get_or_init(|| TansEncoder::from_decode_table(cached_ml_table()))
        .clone()
}

/// Default accuracy log for literal length codes.
pub const LITERAL_LENGTH_ACCURACY_LOG: u8 = 6;

/// Default accuracy log for match length codes.
pub const MATCH_LENGTH_ACCURACY_LOG: u8 = 6;

/// Default accuracy log for offset codes.
pub const OFFSET_ACCURACY_LOG: u8 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(FSE_MAX_ACCURACY_LOG <= 15);
        assert!(LITERAL_LENGTH_ACCURACY_LOG <= FSE_MAX_ACCURACY_LOG);
        assert!(MATCH_LENGTH_ACCURACY_LOG <= FSE_MAX_ACCURACY_LOG);
        assert!(OFFSET_ACCURACY_LOG <= FSE_MAX_ACCURACY_LOG);
    }
}
