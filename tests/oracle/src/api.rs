//! A rows: what the Codec surface answers, from Rust's own codec types (spec §2, §3.4); and the
//! names of the T rows, Rust's `#[test]`s.

use crate::tag;
use haagenti_core::{Codec, CompressionLevel as L, Compressor, Decompressor, Error};
use haagenti_deflate::*;

/// "<category> <reason> <recoverable>" (R4: never the message).
fn err(e: &Error) -> String {
    format!("{} {} {}", e.category(), tag(e), e.is_recoverable())
}

fn surface<C: Codec>(tag: &str, rows: &mut Vec<String>) {
    let input = b"Verify roundtrip functionality.".to_vec();
    let c = C::new();
    rows.push(format!("A {tag} algorithm {}", Compressor::algorithm(&c).name()));
    rows.push(format!("A {tag} level {}", c.level().to_level()));
    rows.push(format!("A {tag} level_best {}", C::with_level(L::Best).level().to_level()));
    for n in [0usize, 1, 65535, 65536, 1_000_000] {
        rows.push(format!("A {tag} max_compressed_size {n} {}", c.max_compressed_size(n)));
    }
    rows.push(format!("A {tag} verify_roundtrip {}", c.verify_roundtrip(&input).unwrap()));
    let z = c.compress(&input).unwrap();
    rows.push(format!("A {tag} decompress {}", crate::sum(&c.decompress(&z).unwrap())));
}

fn one<T: Compressor>(tag: &str, c: T, rows: &mut Vec<String>) {
    rows.push(format!("A {tag} algorithm {} level {}", c.algorithm().name(), c.level().to_level()));
}

pub fn rows() -> Vec<String> {
    let mut rows = vec![];
    surface::<DeflateCodec>("DeflateCodec", &mut rows);
    surface::<ZlibCodec>("ZlibCodec", &mut rows);
    surface::<GzipCodec>("GzipCodec", &mut rows);
    one("DeflateCompressor", DeflateCompressor::with_level(L::Fast), &mut rows);
    one("ZlibCompressor", ZlibCompressor::new(), &mut rows);
    one("GzipCompressor", GzipCompressor::with_level(L::Custom(3)), &mut rows);
    for (tag, a) in [("DeflateDecompressor", DeflateDecompressor::new().algorithm()),
                     ("ZlibDecompressor", ZlibDecompressor::new().algorithm()),
                     ("GzipDecompressor", GzipDecompressor::new().algorithm())] {
        rows.push(format!("A {tag} algorithm {}", a.name()));
    }
    // B2: the free makers give what `with_level` gives.
    for (name, l) in [("fast", L::Fast), ("best", L::Best), ("c5", L::Custom(5))] {
        rows.push(format!("A maker deflate_codec {name} {} {}", Compressor::algorithm(&DeflateCodec::with_level(l)).name(), DeflateCodec::with_level(l).level().to_level()));
        rows.push(format!("A maker zlib_codec {name} {} {}", Compressor::algorithm(&ZlibCodec::with_level(l)).name(), ZlibCodec::with_level(l).level().to_level()));
        rows.push(format!("A maker gzip_codec {name} {} {}", Compressor::algorithm(&GzipCodec::with_level(l)).name(), GzipCodec::with_level(l).level().to_level()));
    }
    for n in [-5, 0, 1, 3, 4, 6, 7, 9, 10, 99] {
        rows.push(format!("A level from_level {n} to_level {}", L::from_level(n).to_level()));
    }
    for e in [
        Error::corrupted("x y"), Error::corrupted_at("bad byte", 17), Error::buffer_too_small(9, 4),
        Error::checksum_mismatch(0xDEADBEEF, 1), Error::unexpected_eof(12), Error::algorithm("deflate", "no"),
        Error::InvalidLevel { level: 12, min: 0, max: 9 }, Error::InvalidDictionary("d".into()),
        Error::AllocationFailed { requested_bytes: 64 },
        Error::InvalidState { expected: "open", actual: "closed" }, Error::Unsupported("u".into()),
    ] {
        rows.push(format!("A error {}", err(&e)));
    }
    rows
}

/// Rust's `#[test]`s in haagenti-deflate, by module, the three `#[ignore]`d ones included.
pub fn tests() -> Vec<String> {
    [
        "codec test_deflate_codec_roundtrip", "codec test_zlib_codec_roundtrip", "codec test_gzip_codec_roundtrip",
        "codec test_compression_levels", "codec test_verify_roundtrip",
        "deflate test_deflate_stored_empty", "deflate test_deflate_stored_small", "deflate test_deflate_fast_small",
        "deflate test_deflate_fast_repetitive", "deflate test_deflate_default", "deflate test_roundtrip_random",
        "gzip test_crc32", "gzip test_gzip_roundtrip_empty", "gzip test_gzip_roundtrip_small",
        "gzip test_gzip_roundtrip_large", "gzip test_gzip_interop_decompress", "gzip test_gzip_interop_compress",
        "huffman test_bit_reader_basic", "huffman test_bit_writer_basic", "huffman test_fixed_huffman_decoder",
        "huffman test_reverse_bits",
        "inflate test_inflate_stored", "inflate test_inflate_fixed_literal", "inflate test_inflate_repetitive",
        "inflate test_inflate_mixed",
        "zlib test_adler32", "zlib test_zlib_roundtrip_empty", "zlib test_zlib_roundtrip_small",
        "zlib test_zlib_roundtrip_large", "zlib test_zlib_interop_decompress", "zlib test_zlib_interop_compress",
    ]
    .iter()
    .map(|t| format!("T {t} pass"))
    .collect()
}
