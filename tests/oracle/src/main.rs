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

/// The port's answer (the patched reference) and Rust's own.
pub fn decode(fmt: &str, x: &[u8]) -> (Result<Vec<u8>>, Result<Vec<u8>>) {
    match fmt {
        "deflate" => {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let pa = port_inflate::inflate(x, &mut a).map(|_| a);
            let rb = haagenti_deflate::inflate(x, &mut b).map(|_| b);
            (pa, rb)
        }
        "zlib" => (port_wrap::zlib_decompress(x), haagenti_deflate::zlib_decompress(x)),
        _ => (port_wrap::gzip_decompress(x), haagenti_deflate::gzip_decompress(x)),
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

pub fn outcome(r: &Result<Vec<u8>>) -> String {
    match r {
        Ok(x) => format!("ok {}", sum(x)),
        Err(e) => format!("err {} {}", e.category(), e),
    }
}

fn is_eof(r: &Result<Vec<u8>>) -> bool {
    matches!(r, Err(Error::UnexpectedEof { .. }))
}

/// The port's answer, checked against Rust's off D1. Logs every D1 row.
fn judged(name: &str, fmt: &str, x: &[u8], d1: &mut Vec<String>) -> String {
    let (port, rust) = decode(fmt, x);
    let (p, r) = (outcome(&port), outcome(&rust));
    if p != r {
        assert!(is_eof(&rust), "{name}: the reference departs from Rust off D1: port {p}, rust {r}");
        d1.push(format!("{name} {fmt}: rust {r} | port {p}"));
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

/// Inputs whose last three bytes match earlier with nothing longer: Rust's lazy step reads past the
/// end (spec §6 D8). Built the same way in tests/consumer/src/corpus.sg.
fn d8_inputs() -> Vec<(&'static str, Vec<u8>)> {
    let mut a = vec![255u8];
    a.extend(0u8..97);
    a.extend([0u8, 1, 2]);
    let mut b = corpus::random(997);
    b.extend_from_slice(&b[500..503].to_vec());
    vec![("d8_short", a), ("d8_random", b)]
}

fn main() {
    let check = std::env::args().any(|a| a == "--check");
    let text = corpus::text(16384);
    let packed = flate2("gzip", &text, 9);
    let mut out = emit::Out::new(check);
    let mut d1 = Vec::new();
    let (mut c, mut r, mut f, mut m, mut v) = (vec![], vec![], vec![], vec![], vec![]);
    let mut fdata = vec![];
    for (name, x) in corpus::named(&packed) {
        for fmt in FORMATS {
            for (ln, l) in LEVELS {
                let y = compress(fmt, &x, l);
                assert_eq!(flate2_decodes(fmt, &y), x, "flate2 decodes rust {name} {fmt} {ln}");
                c.push(format!("C {name} {fmt} {ln} {}", digest(&y)));
                if matches!(ln, "none" | "fast" | "default") {
                    let p = judged(&format!("R {name} {ln}"), fmt, &y, &mut d1);
                    assert_eq!(p, format!("ok {}", sum(&x)), "the port's round trip of {name}");
                    r.push(format!("R {name} {fmt} {ln} {}", sum(&x)));
                }
            }
            let levels: &[u32] = if fmt == "deflate" { &[0, 1, 6, 9] } else { &[6] };
            for &lv in levels {
                let y = flate2(fmt, &x, lv);
                let p = judged(&format!("F {name} {lv}"), fmt, &y, &mut d1);
                assert_eq!(p, format!("ok {}", sum(&x)), "the port decodes flate2's {name} {fmt} {lv}");
                f.push(format!("F {name} {fmt} {lv} {}", sum(&x)));
                fdata.push(vec!["F".into(), name.to_string(), fmt.to_string(), lv.to_string(), emit::hex(&y)]);
            }
        }
    }
    for (name, x) in corpus::big() {
        for (fmt, ln, l) in [("deflate", "none", L::None), ("deflate", "fast", L::Fast),
                             ("deflate", "default", L::Default), ("gzip", "default", L::Default)] {
            let y = compress(fmt, &x, l);
            assert_eq!(flate2_decodes(fmt, &y), x, "flate2 decodes rust {name} {fmt} {ln}");
            v.push(format!("V {name} {fmt} {ln} {}", digest(&y)));
            let p = judged(&format!("V {name} {ln}"), fmt, &y, &mut d1);
            assert_eq!(p, format!("ok {}", sum(&x)));
            v.push(format!("W {name} {fmt} {ln} {}", sum(&x)));
        }
    }
    let mut e = vec![];
    std::panic::set_hook(Box::new(|_| {})); // D8's panics are expected; each is asserted below
    for (name, x) in d8_inputs() {
        for (ln, l) in [("default", L::Default), ("best", L::Best), ("c0", L::Custom(0))] {
            let rust = std::panic::catch_unwind(|| haagenti_deflate::deflate(&x, l));
            assert!(rust.is_err(), "Rust's deflate of {name} at {ln} should panic (D8)");
            let y = port_deflate::deflate(&x, l);
            assert_eq!(flate2_decodes("deflate", &y), x, "flate2 decodes the port's {name}");
            e.push(format!("E {name} deflate {ln} {}", digest(&y)));
            e.push(format!("E {name} roundtrip {ln} {}", sum(&x)));
        }
    }
    let _ = std::panic::take_hook();
    let mut mdata = vec![];
    for (name, fmt, x) in crafted::named().into_iter().chain(crafted::prefixes()).chain(crafted::flips()) {
        m.push(format!("M {name} {fmt} {}", judged(&name, fmt, &x, &mut d1)));
        mdata.push(vec!["M".into(), name, fmt.to_string(), "-".into(), emit::hex(&x)]);
    }
    out.text("c.txt", "C rows: Rust haagenti's bytes per corpus, format and level (spec §5)", &c);
    out.text("r.txt", "R rows: the port decodes Rust's bytes back to the input", &r);
    out.text("f.txt", "F rows: the port decodes flate2's output to the input", &f);
    out.text("m.txt", "M rows: must-fail and edge inputs, the patched reference's answer", &m);
    out.text("e.txt", "E rows: inputs where Rust's deflate panics (spec §6 D8), the port's bytes and round trip", &e);
    out.text("v.txt", "V rows: Rust's bytes for the volume inputs; W rows: their round trip", &v);
    out.text("a.txt", "A rows: the Codec surface's answers, from Rust's codec types", &api::rows());
    out.text("t.txt", "T rows: Rust's #[test]s in haagenti-deflate, ported (the 3 ignored ones too)", &api::tests());
    out.text("d1.txt", "Where Rust's answer differs from the port's (spec §6 D1): Rust UnexpectedEof, the port decodes", &d1);
    out.data("gen_packed", "the `packed` corpus input: flate2's gzip of `text` at level 9",
             &[vec!["P".into(), "packed".into(), "gzip".into(), "9".into(), emit::hex(&packed)]]);
    out.data("gen_f", "flate2's outputs the F rows decode", &fdata);
    out.data("gen_m", "the M rows' inputs", &mdata);
    out.data("gen_t", "the bytes Rust's interop #[test]s take from flate2 (decode) or give to it (compress)", &tdata());
    std::process::exit(out.finish());
}
