//! LARES-932's oracle (spec §5). From this directory: `cargo run --release --offline` writes
//! ../expected/*.txt and ../consumer/src/gen_*.sg; `-- --check` writes nothing and exits 1 when a
//! committed file differs from what it would write. Every assertion here is a property the rows rely
//! on: flate2 decodes Rust's output, the patched reference agrees with Rust off D1 and with flate2.

mod api;
mod corpus;
mod crafted;
mod emit;
mod port_deflate;
mod port_inflate;
mod port_wrap;

use haagenti_core::{CompressionLevel as L, Error, Result};
use std::io::{Read, Write};

pub const LEVELS: [(&str, L); 7] = [
    ("none", L::None), ("fast", L::Fast), ("default", L::Default), ("best", L::Best),
    ("ultra", L::Ultra), ("c0", L::Custom(0)), ("c5", L::Custom(5)),
];
pub const FORMATS: [&str; 3] = ["deflate", "zlib", "gzip"];

/// Rust's bytes; for deflate, checked equal to the patched compressor's (spec §6 D8).
pub fn compress(fmt: &str, x: &[u8], l: L) -> Vec<u8> {
    if fmt == "deflate" {
        let y = haagenti_deflate::deflate(x, l).unwrap();
        assert!(y == port_deflate::deflate(x, l), "the patched compressor departs from Rust off D8");
        return y;
    }
    match fmt {
        "deflate" => haagenti_deflate::deflate(x, l),
        "zlib" => haagenti_deflate::zlib_compress(x, l),
        _ => haagenti_deflate::gzip_compress(x, l),
    }
    .unwrap()
}

/// The port's answer (the patched reference, output bound `limit`) and Rust's own.
pub fn decode(fmt: &str, x: &[u8], limit: usize) -> (Result<Vec<u8>>, Result<Vec<u8>>) {
    match fmt {
        "deflate" => {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let pa = port_inflate::inflate(x, &mut a, limit).map(|_| a);
            let rb = haagenti_deflate::inflate(x, &mut b).map(|_| b);
            (pa, rb)
        }
        "zlib" => (port_wrap::zlib_decompress(x, limit), haagenti_deflate::zlib_decompress(x)),
        _ => (port_wrap::gzip_decompress(x, limit), haagenti_deflate::gzip_decompress(x)),
    }
}

/// The stable reason tag of an error (spec §3.3, R4), from Rust's message where Rust has only one.
pub fn tag(e: &Error) -> &'static str {
    match e {
        Error::CorruptedData { message, .. } => match message.as_str() {
            "invalid block type 3" | "stored block length mismatch" => "block_type",
            "invalid code count" | "repeat at start" | "repeat overflow" | "zero repeat overflow"
            | "long zero repeat overflow" | "invalid huffman code" | "empty code lengths"
            | "invalid literal/length code" | "invalid code length symbol" => "huffman",
            "invalid distance code" | "distance beyond output" | "invalid distance" => "distance",
            "zlib data too short" | "gzip data too short" | "gzip data truncated" => "truncated",
            "unsupported compression method" | "invalid window size" | "zlib header checksum failed"
            | "preset dictionary not supported" => "header",
            "invalid gzip magic" => "magic",
            "zlib checksum mismatch" => "adler",
            "gzip CRC mismatch" => "crc",
            "gzip size mismatch" => "isize",
            "trailing data after the deflate stream" => "trailing",
            "a second gzip member: members are not supported" => "multi_member",
            m if m == "x y" || m.ends_with(" at offset 17") => "corrupted",
            m => panic!("no reason tag for the message {m:?}: add it to spec §3.3 and here"),
        },
        Error::BufferTooSmall { .. } => "output_limit",
        Error::InvalidLevel { .. } => "level",
        Error::InvalidDictionary(_) => "dictionary",
        Error::ChecksumMismatch { .. } => "checksum",
        Error::UnexpectedEof { .. } => "truncated",
        Error::AllocationFailed { .. } => "allocation",
        Error::Algorithm { .. } => "algorithm",
        Error::InvalidState { .. } => "state",
        Error::Unsupported(_) => "unsupported",
        Error::Io(_) => panic!("Io is not ported (R5)"),
    }
}

fn flate2(fmt: &str, x: &[u8], level: u32) -> Vec<u8> {
    let c = flate2::Compression::new(level);
    let mut out = match fmt {
        "deflate" => { let mut e = flate2::write::DeflateEncoder::new(Vec::new(), c); e.write_all(x).unwrap(); e.finish() }
        "zlib" => { let mut e = flate2::write::ZlibEncoder::new(Vec::new(), c); e.write_all(x).unwrap(); e.finish() }
        _ => { let mut e = flate2::write::GzEncoder::new(Vec::new(), c); e.write_all(x).unwrap(); e.finish() }
    }
    .unwrap();
    out.shrink_to_fit();
    out
}

fn flate2_decodes(fmt: &str, x: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    match fmt {
        "deflate" => flate2::read::DeflateDecoder::new(x).read_to_end(&mut out),
        "zlib" => flate2::read::ZlibDecoder::new(x).read_to_end(&mut out),
        _ => flate2::read::GzDecoder::new(x).read_to_end(&mut out),
    }
    .expect("flate2 decodes it");
    out
}

pub fn sum(x: &[u8]) -> String {
    format!("len={} crc={:08x}", x.len(), port_wrap::crc32(x))
}

/// A compressed output: its hex up to 2 KiB, a digest above.
fn digest(x: &[u8]) -> String {
    if x.len() <= 2048 {
        format!("hex={}", emit::hex(x))
    } else {
        format!("len={} crc={:08x} adler={:08x}", x.len(), port_wrap::crc32(x), port_wrap::adler32(x))
    }
}

/// "ok len crc", or "err <category> <reason>" (R4: never the message).
pub fn outcome(r: &Result<Vec<u8>>) -> String {
    match r {
        Ok(x) => format!("ok {}", sum(x)),
        Err(e) => format!("err {} {}", e.category(), tag(e)),
    }
}

/// The ruling a port answer departs from Rust's under, if one may: D1 (Rust's early end), R3
/// (trailing bytes, a second member) or B4 (the bound).
fn departure(port: &Result<Vec<u8>>, rust: &Result<Vec<u8>>) -> Option<&'static str> {
    if let Err(e) = port {
        match tag(e) {
            "trailing" | "multi_member" => return Some("R3"),
            "output_limit" => return Some("B4"),
            _ => {}
        }
    }
    matches!(rust, Err(Error::UnexpectedEof { .. })).then_some("D1")
}

/// The port's answer, checked against Rust's; each departure is logged with its ruling.
fn judged(name: &str, fmt: &str, x: &[u8], limit: usize, log: &mut Vec<String>) -> String {
    let (port, rust) = decode(fmt, x, limit);
    let (p, r) = (outcome(&port), outcome(&rust));
    if p != r {
        let why = departure(&port, &rust).unwrap_or_else(|| panic!("{name}: the reference departs from Rust unruled: port {p}, rust {r}"));
        log.push(format!("{why} {name} {fmt}: rust {r} | port {p}"));
    }
    p
}

/// The interop tests' bytes: flate2's at its default level (6) for the decode tests, and Rust's at
/// `Default` for the compress tests, which flate2 must decode (checked here, since Sigil has no flate2).
fn tdata() -> Vec<Vec<String>> {
    let mut rows = vec![];
    for (name, fmt, x) in [
        ("test_inflate_fixed_literal", "deflate", &b"A"[..]),
        ("test_inflate_repetitive", "deflate", &b"AAAAAAAAAAAAAAAAAAAA"[..]),
        ("test_inflate_mixed", "deflate", &b"Hello, World! This is a test of DEFLATE compression."[..]),
        ("test_zlib_interop_decompress", "zlib", &b"Testing zlib interoperability with flate2 reference."[..]),
        ("test_gzip_interop_decompress", "gzip", &b"Testing gzip interoperability with flate2 reference."[..]),
    ] {
        rows.push(vec!["T".into(), name.into(), fmt.into(), "6".into(), emit::hex(&flate2(fmt, x, 6))]);
    }
    for (name, fmt, x) in [
        ("test_zlib_interop_compress", "zlib", &b"Testing our zlib compression against flate2."[..]),
        ("test_gzip_interop_compress", "gzip", &b"Testing our gzip compression against flate2."[..]),
    ] {
        let y = compress(fmt, x, L::Default);
        assert_eq!(flate2_decodes(fmt, &y), x);
        rows.push(vec!["T".into(), name.into(), fmt.into(), "default".into(), emit::hex(&y)]);
    }
    rows
}

/// The E rows' inputs (spec §6 D8 = R1). `r1_127`'s last three bytes repeat an earlier trigram with
/// nothing longer, so Rust's lazy step reads past the end; `r1_128` is it plus one byte, where Rust
/// does not panic. Built the same way in tests/consumer/src/corpus.sg.
fn r1_inputs() -> Vec<(&'static str, Vec<u8>)> {
    let mut r127: Vec<u8> = (0u8..124).collect();
    r127.extend([1u8, 2, 3]);
    let mut r128 = r127.clone();
    r128.push(0xff);
    let mut a = vec![255u8];
    a.extend(0u8..97);
    a.extend([0u8, 1, 2]);
    let mut b = corpus::random(997);
    b.extend_from_slice(&b[500..503].to_vec());
    vec![("r1_127", r127), ("r1_128", r128), ("d8_short", a), ("d8_random", b)]
}

/// E rows: the port's bytes where Rust panics (the R1 patch's), and Rust's own where it does not.
fn e_rows(log: &mut Vec<String>) -> Vec<String> {
    let mut e = vec![];
    std::panic::set_hook(Box::new(|_| {})); // R1's panics are expected; each is asserted below
    for (name, x) in r1_inputs() {
        for (ln, l) in [("default", L::Default), ("best", L::Best), ("c0", L::Custom(0))] {
            let rust = std::panic::catch_unwind(|| haagenti_deflate::deflate(&x, l).unwrap());
            let y = port_deflate::deflate(&x, l);
            match rust {
                Ok(z) => assert!(name == "r1_128" && z == y, "{name}: Rust should panic, or match the patch"),
                Err(_) => assert!(name != "r1_128", "Rust's deflate of r1_128 should not panic"),
            }
            assert_eq!(flate2_decodes("deflate", &y), x, "flate2 decodes the port's {name}");
            e.push(format!("E {name} deflate {ln} {}", digest(&y)));
            let p = judged(&format!("E {name}"), "deflate", &y, port_inflate::DEFAULT_LIMIT, log);
            assert_eq!(p, format!("ok {}", sum(&x)), "the port's round trip of {name}");
            e.push(format!("E {name} roundtrip {ln} {}", sum(&x)));
        }
    }
    let _ = std::panic::take_hook();
    e
}

/// C, R and F rows for the named corpus, and the F rows' data.
type Rows = (Vec<String>, Vec<String>, Vec<String>, Vec<Vec<String>>);

fn corpus_rows(packed: &[u8], log: &mut Vec<String>) -> Rows {
    let (mut c, mut r, mut f, mut fdata) = (vec![], vec![], vec![], vec![]);
    for (name, x) in corpus::named(packed) {
        for fmt in FORMATS {
            for (ln, l) in LEVELS {
                let y = compress(fmt, &x, l);
                assert_eq!(flate2_decodes(fmt, &y), x, "flate2 decodes rust {name} {fmt} {ln}");
                c.push(format!("C {name} {fmt} {ln} {}", digest(&y)));
                if matches!(ln, "none" | "fast" | "default") {
                    let p = judged(&format!("R {name} {ln}"), fmt, &y, port_inflate::DEFAULT_LIMIT, log);
                    assert_eq!(p, format!("ok {}", sum(&x)), "the port's round trip of {name}");
                    r.push(format!("R {name} {fmt} {ln} {}", sum(&x)));
                }
            }
            let levels: &[u32] = if fmt == "deflate" { &[0, 1, 6, 9] } else { &[6] };
            for &lv in levels {
                let y = flate2(fmt, &x, lv);
                let p = judged(&format!("F {name} {lv}"), fmt, &y, port_inflate::DEFAULT_LIMIT, log);
                assert_eq!(p, format!("ok {}", sum(&x)), "the port decodes flate2's {name} {fmt} {lv}");
                f.push(format!("F {name} {fmt} {lv} {}", sum(&x)));
                fdata.push(vec!["F".into(), name.to_string(), fmt.to_string(), lv.to_string(), emit::hex(&y)]);
            }
        }
    }
    (c, r, f, fdata)
}

/// V and W rows: the volume inputs.
fn volume_rows(log: &mut Vec<String>) -> Vec<String> {
    let mut v = vec![];
    for (name, x) in corpus::big() {
        for (fmt, ln, l) in [("deflate", "none", L::None), ("deflate", "fast", L::Fast),
                             ("deflate", "default", L::Default), ("gzip", "default", L::Default)] {
            let y = compress(fmt, &x, l);
            assert_eq!(flate2_decodes(fmt, &y), x, "flate2 decodes rust {name} {fmt} {ln}");
            v.push(format!("V {name} {fmt} {ln} {}", digest(&y)));
            let p = judged(&format!("V {name} {ln}"), fmt, &y, port_inflate::DEFAULT_LIMIT, log);
            assert_eq!(p, format!("ok {}", sum(&x)));
            v.push(format!("W {name} {fmt} {ln} {}", sum(&x)));
        }
    }
    v
}

fn main() {
    let check = std::env::args().any(|a| a == "--check");
    let packed = flate2("gzip", &corpus::text(16384), 9);
    let mut out = emit::Out::new(check);
    let mut log = Vec::new();
    let (c, r, f, fdata) = corpus_rows(&packed, &mut log);
    let v = volume_rows(&mut log);
    let e = e_rows(&mut log);
    let (mut m, mut mdata) = (vec![], vec![]);
    for (name, fmt, x, limit) in crafted::named().into_iter().chain(crafted::prefixes()).chain(crafted::flips()) {
        let lim = limit.unwrap_or(port_inflate::DEFAULT_LIMIT);
        m.push(format!("M {name} {fmt} {}", judged(&name, fmt, &x, lim, &mut log)));
        let shown = limit.map_or("-".to_string(), |l| l.to_string());
        mdata.push(vec!["M".into(), name, fmt.to_string(), shown, emit::hex(&x)]);
    }
    out.text("c.txt", "C rows: Rust haagenti's bytes per corpus, format and level (spec §5)", &c);
    out.text("r.txt", "R rows: the port decodes Rust's bytes back to the input", &r);
    out.text("f.txt", "F rows: the port decodes flate2's output to the input", &f);
    out.text("m.txt", "M rows: must-fail and edge inputs, `ok len crc` or `err category reason` (R4)", &m);
    out.text("e.txt", "E rows: R1's inputs; the port's bytes (the R1 patch's where Rust panics) and round trip", &e);
    out.text("v.txt", "V rows: Rust's bytes for the volume inputs; W rows: their round trip", &v);
    out.text("a.txt", "A rows: the Codec surface's answers, from Rust's codec types", &api::rows());
    out.text("t.txt", "T rows: Rust's #[test]s in haagenti-deflate, ported (the 3 ignored ones too)", &api::tests());
    out.text("departures.txt", "Where the port's answer differs from Rust's, and the ruling: D1, R3 or B4", &log);
    out.data("gen_packed", "the `packed` corpus input: flate2's gzip of `text` at level 9",
             &[vec!["P".into(), "packed".into(), "gzip".into(), "9".into(), emit::hex(&packed)]]);
    out.data("gen_f", "flate2's outputs the F rows decode", &fdata);
    out.data("gen_m", "the M rows' inputs; the fourth field is the output bound, `-` for the default", &mdata);
    out.data("gen_t", "the bytes Rust's interop #[test]s take from flate2 (decode) or give to it (compress)", &tdata());
    std::process::exit(out.finish());
}
