//! The patched reference: what the Sigil port must answer when decoding (spec §5, §6 D1).
//!
//! A copy of haagenti-deflate's `inflate`, `zlib_decompress` and `gzip_decompress` at develop
//! 28eb2cb, with ONE change: `Decoder::decode`. Rust's decoder peeks the table's full width and
//! refuses with UnexpectedEof when fewer bits remain, even when the code there is complete. This
//! one decodes a complete code from what remains and otherwise refuses exactly as Rust does. main.rs
//! checks this copy against Rust's `inflate` on every input whose Rust answer is not UnexpectedEof,
//! and against flate2 on every valid stream.

use haagenti_core::{Error, Result};
use haagenti_deflate::huffman::{
    CL_CODE_ORDER, DISTANCE_BASE, DISTANCE_EXTRA_BITS, FIXED_DIST_LENGTHS, FIXED_LIT_LENGTHS,
    LENGTH_BASE, LENGTH_EXTRA_BITS, MAX_BITS,
};

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    count: u8,
}

impl<'a> Reader<'a> {
    fn fill(&mut self, n: usize) -> Result<()> {
        while self.count < n as u8 {
            if self.pos >= self.data.len() {
                return Err(Error::unexpected_eof(self.pos));
            }
            self.buf |= (self.data[self.pos] as u64) << self.count;
            self.pos += 1;
            self.count += 8;
        }
        Ok(())
    }
    /// Fill toward `n` bits without failing; gives the bits held.
    fn fill_upto(&mut self, n: usize) -> usize {
        while (self.count as usize) < n && self.pos < self.data.len() {
            self.buf |= (self.data[self.pos] as u64) << self.count;
            self.pos += 1;
            self.count += 8;
        }
        self.count as usize
    }
    fn consume(&mut self, n: usize) {
        self.buf >>= n;
        self.count -= n as u8;
    }
    fn read_bits(&mut self, n: usize) -> Result<u32> {
        self.fill(n)?;
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        Ok(v)
    }
    fn align(&mut self) {
        let d = self.count % 8;
        if d > 0 {
            self.consume(d as usize);
        }
    }
    fn read_byte(&mut self) -> Result<u8> {
        self.align();
        if self.count >= 8 {
            let b = (self.buf & 0xFF) as u8;
            self.consume(8);
            Ok(b)
        } else if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            Ok(b)
        } else {
            Err(Error::unexpected_eof(self.pos))
        }
    }
    fn read_u16(&mut self) -> Result<u16> {
        let lo = self.read_byte()? as u16;
        let hi = self.read_byte()? as u16;
        Ok(lo | (hi << 8))
    }
}

struct Decoder {
    table: Vec<u16>,
    bits: u8,
}

fn reverse(code: u32, n: u8) -> u32 {
    let (mut r, mut c) = (0, code);
    for _ in 0..n {
        r = (r << 1) | (c & 1);
        c >>= 1;
    }
    r
}

impl Decoder {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        if lengths.is_empty() {
            return Err(Error::corrupted("empty code lengths"));
        }
        let mut bl_count = [0u32; MAX_BITS + 1];
        let mut max_bits = 0u8;
        for &l in lengths {
            if l > 0 {
                bl_count[l as usize] += 1;
                max_bits = max_bits.max(l);
            }
        }
        if max_bits == 0 {
            return Ok(Self { table: vec![0; 2], bits: 1 });
        }
        let mut next = [0u32; MAX_BITS + 1];
        let mut code = 0u32;
        for b in 1..=max_bits as usize {
            code = (code + bl_count[b - 1]) << 1;
            next[b] = code;
        }
        let size = 1usize << max_bits;
        let mut table = vec![0u16; size];
        for (sym, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let c = next[l as usize];
            next[l as usize] += 1;
            let r = reverse(c, l);
            let entry = ((sym as u16) << 4) | (l as u16);
            for i in 0..(1usize << (max_bits - l)) {
                let idx = (r as usize) | (i << l);
                if idx < size {
                    table[idx] = entry;
                }
            }
        }
        Ok(Self { table, bits: max_bits })
    }

    /// THE CHANGE (spec §6 D1). With the table's width in hand, as Rust. With fewer bits left, the
    /// zero-padded slot is taken only when its code is complete within the bits there.
    fn decode(&self, r: &mut Reader) -> Result<u16> {
        let width = self.bits as usize;
        let have = r.fill_upto(width);
        let entry = self.table[(r.buf & ((1u64 << width) - 1)) as usize];
        let len = (entry & 0xF) as usize;
        if have < width && (len == 0 || len > have) {
            return Err(Error::unexpected_eof(r.pos));
        }
        if len == 0 {
            return Err(Error::corrupted("invalid huffman code"));
        }
        r.consume(len);
        Ok(entry >> 4)
    }
}

/// Inflate `input`, appending to `output`.
pub fn inflate(input: &[u8], output: &mut Vec<u8>) -> Result<()> {
    let mut r = Reader { data: input, pos: 0, buf: 0, count: 0 };
    loop {
        let fin = r.read_bits(1)? == 1;
        match r.read_bits(2)? {
            0 => stored(&mut r, output)?,
            1 => {
                let l = Decoder::from_lengths(&FIXED_LIT_LENGTHS)?;
                let d = Decoder::from_lengths(&FIXED_DIST_LENGTHS)?;
                block(&mut r, output, &l, &d)?
            }
            2 => dynamic(&mut r, output)?,
            _ => return Err(Error::corrupted("invalid block type 3")),
        }
        if fin {
            return Ok(());
        }
    }
}

fn stored(r: &mut Reader, out: &mut Vec<u8>) -> Result<()> {
    r.align();
    let len = r.read_u16()?;
    let nlen = r.read_u16()?;
    if len != !nlen {
        return Err(Error::corrupted("stored block length mismatch"));
    }
    for _ in 0..len {
        out.push(r.read_byte()?);
    }
    Ok(())
}

fn dynamic(r: &mut Reader, out: &mut Vec<u8>) -> Result<()> {
    let hlit = r.read_bits(5)? as usize + 257;
    let hdist = r.read_bits(5)? as usize + 1;
    let hclen = r.read_bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(Error::corrupted("invalid code count"));
    }
    let mut cl = [0u8; 19];
    for &o in CL_CODE_ORDER.iter().take(hclen) {
        cl[o] = r.read_bits(3)? as u8;
    }
    let cld = Decoder::from_lengths(&cl)?;
    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < lengths.len() {
        let sym = cld.decode(r)?;
        let (count, value, msg) = match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
                continue;
            }
            16 => {
                if i == 0 {
                    return Err(Error::corrupted("repeat at start"));
                }
                (r.read_bits(2)? as usize + 3, lengths[i - 1], "repeat overflow")
            }
            17 => (r.read_bits(3)? as usize + 3, 0, "zero repeat overflow"),
            18 => (r.read_bits(7)? as usize + 11, 0, "long zero repeat overflow"),
            _ => return Err(Error::corrupted("invalid code length symbol")),
        };
        for _ in 0..count {
            if i >= lengths.len() {
                return Err(Error::corrupted(msg));
            }
            lengths[i] = value;
            i += 1;
        }
    }
    let l = Decoder::from_lengths(&lengths[..hlit])?;
    let d = Decoder::from_lengths(&lengths[hlit..])?;
    block(r, out, &l, &d)
}

fn block(r: &mut Reader, out: &mut Vec<u8>, l: &Decoder, d: &Decoder) -> Result<()> {
    loop {
        let sym = l.decode(r)?;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else if sym <= 285 {
            let c = (sym - 257) as usize;
            let e = LENGTH_EXTRA_BITS[c];
            let length = LENGTH_BASE[c] as usize + if e > 0 { r.read_bits(e as usize)? as usize } else { 0 };
            let ds = d.decode(r)? as usize;
            if ds >= DISTANCE_BASE.len() {
                return Err(Error::corrupted("invalid distance code"));
            }
            let de = DISTANCE_EXTRA_BITS[ds];
            let dist = DISTANCE_BASE[ds] as usize + if de > 0 { r.read_bits(de as usize)? as usize } else { 0 };
            if dist == 0 || dist > 32768 {
                return Err(Error::corrupted("invalid distance"));
            }
            if dist > out.len() {
                return Err(Error::corrupted("distance beyond output"));
            }
            let start = out.len() - dist;
            for k in 0..length {
                let b = out[start + k];
                out.push(b);
            }
        } else {
            return Err(Error::corrupted("invalid literal/length code"));
        }
    }
}
