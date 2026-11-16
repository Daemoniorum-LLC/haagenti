//! The patched compressor reference (spec §6 D8): haagenti-deflate's `deflate_default` at 28eb2cb,
//! copied, with ONE change: the lazy step is skipped when fewer than three bytes follow `pos + 1`.
//! Rust hashes `pos + 1` there and indexes one past the end: a panic. main.rs checks this copy against
//! Rust's `deflate` on every C and V row, where Rust does not panic.

use haagenti_core::CompressionLevel;
use haagenti_deflate::huffman::{
    BitWriter, HuffmanEncoder, DISTANCE_BASE, DISTANCE_EXTRA_BITS, FIXED_DIST_LENGTHS,
    FIXED_LIT_LENGTHS, LENGTH_BASE, LENGTH_EXTRA_BITS,
};

const MAX_MATCH: usize = 258;
const WINDOW: usize = 32768;

fn hash3(d: &[u8], p: usize) -> usize {
    (((d[p] as usize) << 10) ^ ((d[p + 1] as usize) << 5) ^ d[p + 2] as usize) & (WINDOW - 1)
}

fn emit(w: &mut BitWriter, lit: &HuffmanEncoder, dist: &HuffmanEncoder, len: usize, d: usize) {
    let lc = (0..29).rev().find(|&i| len >= LENGTH_BASE[i] as usize).unwrap_or(0);
    let (c, l) = lit.get(257 + lc);
    w.write_code(c, l);
    if LENGTH_EXTRA_BITS[lc] > 0 {
        w.write_bits((len - LENGTH_BASE[lc] as usize) as u32, LENGTH_EXTRA_BITS[lc]);
    }
    let dc = (0..30).rev().find(|&i| d >= DISTANCE_BASE[i] as usize).unwrap_or(0);
    let (c, l) = dist.get(dc);
    w.write_code(c, l);
    if DISTANCE_EXTRA_BITS[dc] > 0 {
        w.write_bits((d - DISTANCE_BASE[dc] as usize) as u32, DISTANCE_EXTRA_BITS[dc]);
    }
}

fn run(input: &[u8], start: usize, max: usize, at: usize) -> usize {
    let mut len = 0;
    while len < max && input[start + len] == input[at + len] {
        len += 1;
    }
    len
}

/// What the port gives for `deflate(input, level)`.
pub fn deflate(input: &[u8], level: CompressionLevel) -> Vec<u8> {
    match level {
        CompressionLevel::None | CompressionLevel::Fast => haagenti_deflate::deflate(input, level).unwrap(),
        _ if input.len() < 100 => haagenti_deflate::deflate(input, CompressionLevel::Fast).unwrap(),
        _ => chained(input),
    }
}

fn chained(input: &[u8]) -> Vec<u8> {
    let lit = HuffmanEncoder::from_lengths(&FIXED_LIT_LENGTHS);
    let dist = HuffmanEncoder::from_lengths(&FIXED_DIST_LENGTHS);
    let mut w = BitWriter::with_capacity(input.len());
    w.write_bits(1, 1);
    w.write_bits(1, 2);
    let (mut head, mut prev) = (vec![0u32; WINDOW], vec![0u32; WINDOW]);
    let n = input.len();
    let mut pos = 0;
    while pos < n {
        let (mut best, mut best_d) = (0, 0);
        if pos + 3 <= n {
            let h = hash3(input, pos);
            let (mut cand, mut walked) = (head[h] as usize, 0);
            while cand > 0 && pos >= cand && pos - cand <= WINDOW && walked < 128 {
                let len = run(input, cand, (n - pos).min(MAX_MATCH), pos);
                if len > best {
                    best = len;
                    best_d = pos - cand;
                    if len == MAX_MATCH {
                        break;
                    }
                }
                cand = prev[cand % WINDOW] as usize;
                walked += 1;
            }
            prev[pos % WINDOW] = head[h];
            head[h] = pos as u32;
        }
        if best >= 3 {
            // THE CHANGE: Rust's guard is `pos + 1 < n`, which reads input[n] when pos + 3 == n.
            if pos + 4 <= n && best < MAX_MATCH {
                let cand = head[hash3(input, pos + 1)] as usize;
                if cand > 0 && pos + 1 >= cand && pos + 1 - cand <= WINDOW
                    && run(input, cand, (n - pos - 1).min(MAX_MATCH), pos + 1) > best + 1
                {
                    let (c, l) = lit.get(input[pos] as usize);
                    w.write_code(c, l);
                    pos += 1;
                    continue;
                }
            }
            emit(&mut w, &lit, &dist, best, best_d);
            for i in 1..best {
                if pos + i + 3 <= n {
                    let h = hash3(input, pos + i);
                    prev[(pos + i) % WINDOW] = head[h];
                    head[h] = (pos + i) as u32;
                }
            }
            pos += best;
        } else {
            let (c, l) = lit.get(input[pos] as usize);
            w.write_code(c, l);
            pos += 1;
        }
    }
    let (c, l) = lit.get(256);
    w.write_code(c, l);
    w.finish()
}
