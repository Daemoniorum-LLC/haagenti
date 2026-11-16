//! The must-fail inputs (spec §5, M rows): each named case of spec §3.3, every prefix of three small
//! streams, and seeded single-bit flips. Each is (name, format, bytes).

use crate::corpus::{text, xorshift32};
use haagenti_core::CompressionLevel;
use haagenti_deflate::huffman::{BitWriter, HuffmanEncoder, FIXED_LIT_LENGTHS};
use haagenti_deflate::{deflate, gzip_compress, zlib_compress};

/// (name, format, bytes, output bound: `None` for the default).
pub type Case = (String, &'static str, Vec<u8>, Option<usize>);

fn case(name: &str, fmt: &'static str, bytes: Vec<u8>) -> Case {
    (name.to_string(), fmt, bytes, None)
}

fn bounded(name: &str, fmt: &'static str, bytes: Vec<u8>, limit: usize) -> Case {
    (name.to_string(), fmt, bytes, Some(limit))
}

/// A fixed-Huffman block: BFINAL=1, BTYPE=01, then `f` writes the symbols.
fn fixed(f: impl Fn(&mut BitWriter, &HuffmanEncoder)) -> Vec<u8> {
    let lit = HuffmanEncoder::from_lengths(&FIXED_LIT_LENGTHS);
    let mut w = BitWriter::new();
    w.write_bits(1, 1);
    w.write_bits(1, 2);
    f(&mut w, &lit);
    w.finish()
}

fn sym(w: &mut BitWriter, lit: &HuffmanEncoder, s: usize) {
    let (c, l) = lit.get(s);
    w.write_code(c, l);
}

/// A dynamic header: BFINAL=1, BTYPE=10, HLIT-257, HDIST-1, HCLEN-4 and the code-length code
/// lengths in the RFC's order, then `body` writes the rest.
fn dynamic(hlit: u32, hdist: u32, cl: &[(usize, u32)], body: impl Fn(&mut BitWriter)) -> Vec<u8> {
    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
    let mut lens = [0u32; 19];
    for &(s, l) in cl {
        lens[s] = l;
    }
    let hclen = ORDER.iter().rposition(|&s| lens[s] > 0).map_or(4, |p| (p + 1).max(4));
    let mut w = BitWriter::new();
    w.write_bits(1, 1);
    w.write_bits(2, 2);
    w.write_bits(hlit, 5);
    w.write_bits(hdist, 5);
    w.write_bits(hclen as u32 - 4, 4);
    for &s in ORDER.iter().take(hclen) {
        w.write_bits(lens[s], 3);
    }
    body(&mut w);
    w.finish()
}

/// Write `n` one-bit codes of value `bit` (the two-symbol code-length codes below).
fn bits(w: &mut BitWriter, bit: u32, n: usize) {
    for _ in 0..n {
        w.write_bits(bit, 1);
    }
}

pub fn named() -> Vec<Case> {
    let mut v = vec![
        case("empty", "deflate", vec![]),
        case("btype3", "deflate", vec![0x07]),
        case("stored_mismatch", "deflate", vec![0x01, 5, 0, 0, 0, b'H', b'e', b'l', b'l', b'o']),
        case("stored_truncated", "deflate", vec![0x01, 5, 0, 250, 255, b'H', b'e']),
        case("no_final_block", "deflate", vec![0x00, 1, 0, 254, 255, b'x']),
        case("trailing_after_final", "deflate", vec![0x01, 2, 0, 253, 255, b'H', b'i', 0xde, 0xad]),
        case("distance_beyond_output", "deflate", fixed(|w, l| { sym(w, l, 257); w.write_code(0, 5); })),
        case("distance_far_back", "deflate", fixed(|w, l| {
            sym(w, l, 97);
            sym(w, l, 257);
            w.write_code(29, 5);
            w.write_bits(0, 13);
        })),
        case("lit_286", "deflate", fixed(|w, l| sym(w, l, 286))),
        case("dist_code_30", "deflate", fixed(|w, l| { sym(w, l, 97); sym(w, l, 257); w.write_code(30, 5); })),
        case("code_count_hlit", "deflate", dynamic(31, 0, &[(0, 1), (16, 1)], |_| {})),
        case("code_count_hdist", "deflate", dynamic(0, 31, &[(0, 1), (16, 1)], |_| {})),
        // sym 0 is code 0, the repeat symbol code 1 (canonical: shorter symbol number first)
        case("repeat_at_start", "deflate", dynamic(0, 0, &[(0, 1), (16, 1)], |w| bits(w, 1, 1))),
        case("repeat_overflow", "deflate", dynamic(0, 0, &[(0, 1), (16, 1)], |w| {
            bits(w, 0, 257);
            bits(w, 1, 1);
            w.write_bits(3, 2);
        })),
        case("zero_repeat_overflow", "deflate", dynamic(0, 0, &[(0, 1), (17, 1)], |w| {
            bits(w, 0, 256);
            bits(w, 1, 1);
            w.write_bits(7, 3);
        })),
        case("long_zero_repeat_overflow", "deflate", dynamic(0, 0, &[(0, 1), (18, 1)], |w| {
            bits(w, 1, 1);
            w.write_bits(127, 7);
            bits(w, 1, 1);
            w.write_bits(127, 7);
        })),
        // literal 0 gets length 1, nothing else: slot 1 of its two-slot table is empty
        case("huffman_empty_slot", "deflate", dynamic(0, 0, &[(0, 1), (1, 1)], |w| {
            bits(w, 1, 1);
            bits(w, 0, 256);
            bits(w, 1, 1);
            bits(w, 1, 8);
        })),
        // every literal/length length zero: the all-zero table
        case("huffman_all_zero", "deflate", dynamic(0, 0, &[(0, 1), (1, 1)], |w| {
            bits(w, 0, 257);
            bits(w, 1, 1);
            bits(w, 0, 8);
        })),
    ];
    v.extend(zlib_cases());
    v.extend(gzip_cases());
    v.extend(ruled_cases());
    v
}

/// R3: trailing bytes in each place, and B4: a bound one byte short and exact, in each format.
fn ruled_cases() -> Vec<Case> {
    let t = text(1000);
    let d = deflate(&t, CompressionLevel::Default).unwrap();
    let z = zlib_compress(&t, CompressionLevel::Default).unwrap();
    let g = gzip_compress(&t, CompressionLevel::Default).unwrap();
    let (zn, gn) = (z.len(), g.len());
    vec![
        case("r3_deflate_trailing", "deflate", [d.clone(), vec![0x00]].concat()),
        case("r3_zlib_before_adler", "zlib", [z[..zn - 4].to_vec(), vec![0x00], z[zn - 4..].to_vec()].concat()),
        case("r3_zlib_after_adler", "zlib", [z.clone(), vec![0x00]].concat()),
        case("r3_gzip_before_trailer", "gzip", [g[..gn - 8].to_vec(), vec![0x00], g[gn - 8..].to_vec()].concat()),
        case("r3_gzip_after_trailer", "gzip", [g.clone(), b"xy".to_vec()].concat()),
        case("r3_gzip_after_trailer_1f", "gzip", [g.clone(), vec![0x1f]].concat()),
        bounded("b4_deflate_short", "deflate", d.clone(), 999),
        bounded("b4_deflate_exact", "deflate", d, 1000),
        bounded("b4_zlib_short", "zlib", z.clone(), 999),
        bounded("b4_zlib_exact", "zlib", z, 1000),
        bounded("b4_gzip_short", "gzip", g.clone(), 999),
        bounded("b4_gzip_exact", "gzip", g, 1000),
        bounded("b4_stored_short", "deflate", deflate(&t, CompressionLevel::None).unwrap(), 999),
        bounded("b4_stored_exact", "deflate", deflate(&t, CompressionLevel::None).unwrap(), 1000),
    ]
}

fn fcheck(cmf: u8, flg: u8) -> u8 {
    let r = ((cmf as u16) * 256 + flg as u16) % 31;
    flg | if r == 0 { 0 } else { (31 - r) as u8 }
}

fn zlib_cases() -> Vec<Case> {
    let z = zlib_compress(&text(1000), CompressionLevel::Default).unwrap();
    let with = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut x = z.clone();
        f(&mut x);
        x
    };
    vec![
        case("z_short", "zlib", vec![0x78, 0x9c, 3, 0, 0]),
        case("z_method", "zlib", with(&|x| x[0] = 0x79)),
        case("z_window", "zlib", with(&|x| { x[0] = 0x88; x[1] = fcheck(0x88, 0x80); })),
        case("z_fcheck", "zlib", with(&|x| x[1] ^= 1)),
        case("z_fdict", "zlib", with(&|x| x[1] = fcheck(0x78, 0x20))),
        case("z_adler", "zlib", with(&|x| { let n = x.len(); x[n - 1] ^= 1; })),
        case("z_truncated", "zlib", with(&|x| { let n = x.len(); x.truncate(n - 10); })),
    ]
}

fn gzip_cases() -> Vec<Case> {
    let a = text(1000);
    let g = gzip_compress(&a, CompressionLevel::Default).unwrap();
    let h = gzip_compress(&a[..500], CompressionLevel::Fast).unwrap();
    let with = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut x = g.clone();
        f(&mut x);
        x
    };
    let header = |flags: u8, extra: &[u8]| {
        let mut x = g[..10].to_vec();
        x[3] = flags;
        x.extend_from_slice(extra);
        x.extend_from_slice(&g[10..]);
        x
    };
    vec![
        case("g_short", "gzip", g[..17].to_vec()),
        case("g_magic", "gzip", with(&|x| x[0] = 0x1e)),
        case("g_method", "gzip", with(&|x| x[2] = 7)),
        case("g_crc", "gzip", with(&|x| { let n = x.len(); x[n - 8] ^= 1; })),
        case("g_isize", "gzip", with(&|x| { let n = x.len(); x[n - 1] ^= 1; })),
        case("g_truncated", "gzip", with(&|x| { let n = x.len(); x.truncate(n - 3); })),
        case("g_fextra_long", "gzip", header(4, &[0xff, 0xff])),
        case("g_fextra_ok", "gzip", header(4, &[3, 0, 1, 2, 3])),
        case("g_fname_ok", "gzip", header(8, b"corpus.txt\0")),
        case("g_fname_unterminated", "gzip", { let mut x = g[..10].to_vec(); x[3] = 8; x.extend(vec![b'n'; 40]); x }),
        case("g_fcomment_fhcrc", "gzip", header(16 | 2, b"note\0\xaa\xbb")),
        case("g_two_members_differ", "gzip", [g.clone(), h.clone()].concat()),
        case("g_two_members_same", "gzip", [g.clone(), g.clone()].concat()),
    ]
}

/// Every proper prefix of a small stream in each format.
pub fn prefixes() -> Vec<Case> {
    let s = b"Hello, World! Hello, World!";
    let mut v = Vec::new();
    for (fmt, full) in [
        ("deflate", deflate(s, CompressionLevel::Fast).unwrap()),
        ("zlib", zlib_compress(s, CompressionLevel::Fast).unwrap()),
        ("gzip", gzip_compress(s, CompressionLevel::Fast).unwrap()),
    ] {
        for n in 0..full.len() {
            v.push(case(&format!("prefix_{fmt}_{n}"), fmt, full[..n].to_vec()));
        }
    }
    v
}

/// 64 seeded single-bit flips of a 512-byte text stream at `Default`, per format.
pub fn flips() -> Vec<Case> {
    let t = text(512);
    let mut v = Vec::new();
    let mut x = 0x0BAD_5EEDu32;
    for (fmt, full) in [
        ("deflate", deflate(&t, CompressionLevel::Default).unwrap()),
        ("zlib", zlib_compress(&t, CompressionLevel::Default).unwrap()),
        ("gzip", gzip_compress(&t, CompressionLevel::Default).unwrap()),
    ] {
        for k in 0..64 {
            x = xorshift32(x);
            let bit = (x as usize) % (full.len() * 8);
            let mut m = full.clone();
            m[bit / 8] ^= 1 << (bit % 8);
            v.push(case(&format!("flip_{fmt}_{k}_bit{bit}"), fmt, m));
        }
    }
    v
}
