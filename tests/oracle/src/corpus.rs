//! The corpus (spec §5). The same rules are written in Sigil in tests/consumer/src/corpus.sg; any
//! change here is a change there.

pub const WORDS: [&str; 32] = [
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "as", "was", "with", "be", "by",
    "on", "not", "he", "this", "are", "or", "his", "from", "at", "which", "but", "have", "an",
    "they", "you", "were", "compression",
];

pub fn xorshift32(x: u32) -> u32 {
    let mut x = x;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/// `n` bytes of seeded words: each step draws x, writes WORDS[x % 32], then ".\n" when
/// (x >> 8) % 11 == 0 and " " otherwise; the last word is cut at `n`.
pub fn text(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut x = 2_463_534_242u32;
    while out.len() < n {
        x = xorshift32(x);
        let sep: &[u8] = if (x >> 8) % 11 == 0 { b".\n" } else { b" " };
        for &b in WORDS[(x % 32) as usize].as_bytes().iter().chain(sep) {
            if out.len() < n {
                out.push(b);
            }
        }
    }
    out
}

/// `n` bytes of xorshift32 from 0x9E3779B9, the top byte of each step.
pub fn random(n: usize) -> Vec<u8> {
    let mut x = 0x9E37_79B9u32;
    (0..n)
        .map(|_| {
            x = xorshift32(x);
            (x >> 24) as u8
        })
        .collect()
}

/// The named corpus, in row order. `packed` is flate2's gzip of `text` at level 9, built by main.rs.
pub fn named(packed: &[u8]) -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty", vec![]),
        ("one", vec![0x41]),
        ("all256", (0..=255u8).collect()),
        ("digits", b"123456789".to_vec()), // CRC-32 cbf43926: its top bit set (B5)
        ("run65535", vec![0x61; 65535]),
        ("run65536", vec![0x61; 65536]),
        ("text", text(16384)),
        ("random", random(8192)),
        ("packed", packed.to_vec()),
    ]
}

/// The volume inputs (spec §5): the word generator at 1 MiB and 8 MiB.
pub fn big() -> Vec<(&'static str, Vec<u8>)> {
    vec![("mib1", text(1 << 20)), ("mib8", text(8 << 20))]
}
