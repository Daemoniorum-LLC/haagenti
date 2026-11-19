# haagenti in Sigil, slice 1: the Codec surface, DEFLATE, zlib, gzip (LARES-932)

Status: the spec for slice 1, revised to contract 0.1.1 §6 (the planner's rulings R1–R5 and B1–B8,
PR #65 comment 6922707269). The Rust crates `haagenti-core` and `haagenti-deflate` at develop
`28eb2cbd9dfe59d9b4ca273b4f1312ebb426420a` are the oracle; RFCs 1950, 1951 and 1952 are the formats.
Where the two disagree, §6 (in `SLICE-1-DEPARTURES.md`) says which one the port follows and why.
Every name is the port's. Behaviour (the bytes produced, the errors refused) is Rust's unless §6 says
otherwise. Refusals are compared by kind and reason tag, never by message (R4).

## 1. Layout

- One manifest: `Sigil.toml` at the repo root, `[project] name = "haagenti"`. The Rust workspace
  (`Cargo.toml`, `crates/`) is untouched and stays the oracle until the migration completes.
- `src/lib.sg` starts `tome haagenti;` and declares two public scrolls:
  - `core` (`src/core/`): `error`, `level`, `aspects`.
  - `deflate` (`src/deflate/`): `tables`, `bits`, `huffman`, `inflate`, `lz77`, `deflate`,
    `checksum`, `zlib`, `gzip`, and the codec types in `codec_deflate`, `codec_zlib`, `codec_gzip`.
- A consumer writes `invoke haagenti·core·{Error, CompressionLevel, Algorithm, Compressor,
  Decompressor, Codec};` and `invoke haagenti·deflate·{GzipCodec, gzip_compress, …};`.
- Private names carry the prefix `hz_`, since a tome's names share one namespace.
- 300 lines per file and 80 per function. Pure Sigil: no host primitive, no `haagenti-simd` path.

## 2. `haagenti·core`

Only what `haagenti-deflate` uses is ported.

### 2.1 `Error`

`Result<T, Error>` is written out everywhere; there is no alias. `☉ ᛈ Error` keeps ten of Rust's
eleven variants, each with named fields. `Io` and `CorruptedData`'s boxed `source` are dropped (R5).

| Variant | Fields | `category()` | `reason()` |
|---|---|---|---|
| `CorruptedData` | `message: String, reason: String` | `corrupted_data` | its `reason` (§3.3) |
| `BufferTooSmall` | `required: usize, provided: usize` | `buffer_too_small` | `output_limit` |
| `InvalidLevel` | `level: i32, min: i32, max: i32` | `invalid_level` | `level` |
| `InvalidDictionary` | `message: String` | `invalid_dictionary` | `dictionary` |
| `ChecksumMismatch` | `expected: u32, actual: u32` | `checksum_mismatch` | `checksum` |
| `UnexpectedEof` | `bytes_read: usize` | `unexpected_eof` | `truncated` |
| `AllocationFailed` | `requested_bytes: usize` | `allocation_failed` | `allocation` |
| `Algorithm` | `algorithm: String, message: String` | `algorithm_error` | `algorithm` |
| `InvalidState` | `expected: String, actual: String` | `invalid_state` | `state` |
| `Unsupported` | `message: String` | `unsupported` | `unsupported` |

- **Makers.**
  - `Error·refused(reason, message)` is the tagged `CorruptedData` that every refusal in §3.3 uses.
  - `Error·corrupted(m)` keeps Rust's maker, with the reason `corrupted`.
  - `Error·corrupted_at(m, offset)` makes the message `"{m} at offset {offset}"`.
  - The others are `Error·buffer_too_small(required, provided)`,
    `Error·checksum_mismatch(expected, actual)`, `Error·unexpected_eof(bytes_read)` and
    `Error·algorithm(a, m)`.
- **`message()`** is Rust's `Display` text, kept for people. It is never what a test compares.
- **`is_recoverable()`** is `yea` for `UnexpectedEof` and `BufferTooSmall` only.
- **Fields.** Rust's `&'static str` fields become `String`.

### 2.2 `CompressionLevel` and `Algorithm`

- `☉ ᛈ CompressionLevel { None, Fast, Default, Best, Ultra, Custom(i32) }`.
  - `to_level()`: 0, 1, 6, 9, 12, or the custom number.
  - `CompressionLevel·from_level(n)`: 0 → `None`; 1–3 → `Fast`; 4–6 → `Default`; 7–9 → `Best`;
    10 and above → `Ultra`; a negative `n` → `Custom(n)`.
  - Rust's `#[default]` is `Default`, which is what every codec's `new()` takes.
- `☉ ᛈ Algorithm { Lz4, Lz4Hc, Zstd, Brotli, Deflate, Gzip, Zlib, Snappy, Lzma }` with `name()`
  ("lz4", "lz4hc", "zstd", "brotli", "deflate", "gzip", "zlib", "snappy", "lzma"). The other
  `Algorithm` rites (`typical_ratio`, `typical_speed_mbs`, `supports_dictionary`) are not used by
  deflate and are not ported.

### 2.3 The aspects

```sigil
☉ aspect Compressor {
    rite algorithm(&this) → Algorithm;
    rite level(&this) → CompressionLevel;
    rite compress(&this, input: &Vec<u8>) → Result<Vec<u8>, Error>;
    rite max_compressed_size(&this, input_len: usize) → usize;
}
☉ aspect Decompressor {
    rite algorithm(&this) → Algorithm;
    rite decompress(&this, input: &Vec<u8>) → Result<Vec<u8>, Error>;
}
☉ aspect Codec: Compressor + Decompressor {
    rite new() → This;
    rite with_level(level: CompressionLevel) → This;
    rite verify_roundtrip(&this, data: &Vec<u8>) → Result<bool, Error>;
}
```

- **Out of slice 1.** `compress_to`, `decompress_to` and `decompress_with_size` (B1), and `stats()`,
  `measure_ratio`, `CompressionStats` and `CompressionRatio` (B3). Every Rust `stats()` gives `None`.
- **`decompress`** refuses past the default output bound (§3.3, B4).
- **`verify_roundtrip(data)`** compresses, decompresses and gives `Ok(decompressed == data)`, passing
  either step's error through. Rust gives it a default body. The port declares that body in the
  aspect, and each codec also implements it itself, because AOT and wasm do not lower a dependency
  tome's default rite that returns a `Result` (§7 K1).
- **`new` and `with_level` have no subject (B2).**
  - Probed: `DeflateCodec·with_level(l)` and `GzipCodec·new()` check, run, compile and run on wasm
    (`tests/consumer`'s A rows).
  - Called through a generic, `C·with_level(l)` checks and runs, but AOT and wasm do not lower it
    (§7 K9).
  - So `haagenti·deflate` also gives the free makers `deflate_codec(level)`, `zlib_codec(level)` and
    `gzip_codec(level)`, and GAPS.md has the row.
- **Out of this slice too:** the streaming, dictionary, parallel and SIMD aspects and `dct.rs`, since
  deflate implements none of them.

## 3. `haagenti·deflate`

### 3.1 The free rites

| Rite | Gives |
|---|---|
| `deflate(input: &Vec<u8>, level: CompressionLevel)` | `Result<Vec<u8>, Error>`: raw RFC 1951 |
| `inflate(input: &Vec<u8>, output: &Δ Vec<u8>)`, `inflate_limited(input, output, limit)` | `Result<(), Error>`: appends to `output` |
| `zlib_compress(input, level)`, `zlib_decompress(input)`, `zlib_decompress_limited(input, limit)` | RFC 1950 |
| `gzip_compress(input, level)`, `gzip_decompress(input)`, `gzip_decompress_limited(input, limit)` | RFC 1952 |
| `deflate_codec(level)`, `zlib_codec(level)`, `gzip_codec(level)` | the codecs (B2) |
| `crc32(data: &Vec<u8>) → u32`, `crc32_range(data, start, end)`, `adler32(data) → u32` | the checksums |

- **The output bound (B4).** `inflate`, `zlib_decompress` and `gzip_decompress` stop at
  `DEFAULT_OUTPUT_LIMIT` (16 MiB, 16,777,216 bytes). The `*_limited` rites take the bound, which
  counts the bytes this call produces.
- **No `*_to` rites (B1).** `inflate_to`, `zlib_decompress_to` and `gzip_decompress_to` are out of
  slice 1.

`crc32` and `adler32` are private in Rust. They are public here because a gzip or zlib framer, such as
http's `Encoder`, needs them, and because the tests check them against known vectors.

The pieces are public as in Rust (`huffman`): `BitWriter`, `HuffmanEncoder` (`code(s)`, `len(s)`),
`HuffmanDecoder` (`from_lengths`, `table_bits`, `decode`), `reverse_bits`, and the tables in
`haagenti·deflate·tables`. `BitReader` is a cursor, not an owner: a `sigil` cannot hold a borrow, so
each read is given the bytes (`BitReader·over(data)`, `r.read_bits(data, n)`, `r.read_byte(data)`).
Every public rite gives a `Result` where Rust's does. The decoder's inner loops use tome-private
`hz_` rites instead, which give a number and mark an early end on the reader (§4, SIGIL-2153).

### 3.2 Compression: the bytes

`deflate` picks one of three encoders by level. Every encoder is a faithful port, so the output is
Rust's, byte for byte.

- **`None` → stored.** The input is cut into blocks of at most 65,535 bytes, each written as
  `BFINAL|00`, `LEN` and `NLEN` (little endian) and the bytes. `BFINAL` is set on the block that
  reaches the end of the input. Empty input gives the one block `01 00 00 ff ff`.
- **`Fast` → fixed Huffman, greedy.** Empty input is written as stored. Otherwise the output is one
  block, `BFINAL=1` and `BTYPE=01`. A 32,768-entry hash table maps
  `hash3 = ((b0 << 10) ^ (b1 << 5) ^ b2) & 32767` to the last position. A candidate at `prev` is tried
  only when `prev > 0` and `pos - prev ≤ 32768`: position 0 is never a match source. A match is kept
  when its length is at least 3 (at most 258). After a match, the hash of every skipped position that
  has three bytes ahead is updated. Finally the end-of-block code is written.
- **Everything else (`Default`, `Best`, `Ultra`, `Custom(n)` for any `n`) → fixed Huffman with hash
  chains and one-step lazy matching.**
  - Input under 100 bytes goes to `Fast`.
  - The table is as in `Fast`, plus a 32,768-entry chain indexed by `pos % 32768`. At most 128
    candidates are walked, the walk stops at the first match of 258, and a later candidate wins only
    if it is strictly longer.
  - The lazy check probes the head of the chain for `pos + 1` only. If that match is longer than
    `best + 1`, the byte at `pos` is written as a literal. It is skipped when fewer than three bytes
    follow `pos + 1`, where Rust panics (§6 D8).
  - The chain update and the skipped-position update are as in the Rust code.
- The bit writer packs least-significant bit first. Huffman codes are written bit-reversed, and the
  last partial byte is padded with zeros.
- **zlib.** `CMF = 0x78`, then `FLG = FLEVEL << 6 | FCHECK`, then the DEFLATE bytes, then Adler-32
  big endian. `FLEVEL` is 0, 1, 2 and 3 for `None`, `Fast`, `Default`, and `Best` or `Ultra`.
  `Custom(n)` gives 0 when `n ≤ 1`, 1 when `n ≤ 4`, 2 when `n ≤ 6`, and 3 otherwise.
- **gzip.** The header is `1f 8b 08 00`, then `MTIME = 0`, then `XFL`, then `OS = 3`. `XFL` is 2 for
  `Best` or `Ultra`, 4 for `Fast` and 0 otherwise. The DEFLATE bytes follow, then CRC-32 and ISIZE
  (`len mod 2^32`), both little endian.
- `max_compressed_size(n) = n + (n / 65535 + 1) * 5 + k`, where `k` is 10 for deflate, 16 for zlib
  and 28 for gzip.

### 3.3 Decompression: what is accepted, what is refused

`inflate` reads blocks until one has `BFINAL` set. Every refusal is
`CorruptedData { message: Rust's message, reason: tag }` unless the table says otherwise. Tests
compare `category()` and `reason()` (R4).

| Condition | Rust's message | `reason()` |
|---|---|---|
| input ends inside a header, length, extra bits, stored block or Huffman code (§6 D1) | `UnexpectedEof { bytes_read }`: the inflate range read, all of it | `truncated` |
| `BTYPE = 3` | `invalid block type 3` | `block_type` |
| stored `LEN ≠ !NLEN` | `stored block length mismatch` | `block_type` |
| `HLIT > 286` or `HDIST > 30` | `invalid code count` | `huffman` |
| code-length symbol 16 first; a 16, 17 or 18 run past `HLIT + HDIST` | `repeat at start`; `repeat overflow`, `zero repeat overflow`, `long zero repeat overflow` | `huffman` |
| an empty table slot; an empty length list | `invalid huffman code`; `empty code lengths` | `huffman` |
| literal/length symbol 286 or 287 | `invalid literal/length code` | `huffman` |
| distance symbol 30 or 31; distance beyond the output | `invalid distance code`; `distance beyond output` | `distance` |
| bytes after the final block's last byte (R3) | `trailing data after the deflate stream` | `trailing` |
| output past the bound (B4) | `BufferTooSmall { required: limit + 1, provided: limit }` | `output_limit` |

- **Tables.** All-zero lengths build two empty slots. Over-subscribed and incomplete sets are built as
  Rust builds them (§6 D3).
- **Where the stream ends.** The stream ends at the byte holding its last bit; the bits after it in
  that byte are padding.

**zlib** (`zlib_decompress`), in order:
1. Under 6 bytes: `zlib data too short`, reason `truncated`.
2. `CM ≠ 8`: `unsupported compression method`, reason `header`.
3. `CINFO > 7`: `invalid window size`, reason `header`.
4. `(CMF·256 + FLG) mod 31 ≠ 0`: `zlib header checksum failed`, reason `header`.
5. `FDICT` set: `preset dictionary not supported`, reason `header`.
6. Bytes `2 .. len-4` are inflated. If the stream ends before `len-4`, the bytes between it and the
   Adler-32 are refused, reason `trailing` (R3).
7. The Adler-32 (big endian) must match, or `zlib checksum mismatch`, reason `adler`.

**gzip** (`gzip_decompress`), in order:
1. Under 18 bytes: `gzip data too short`, reason `truncated`.
2. Not `1f 8b`: `invalid gzip magic`, reason `magic`.
3. `CM ≠ 8`: `unsupported compression method`, reason `header`.
4. The optional header fields are skipped:
   - `FEXTRA`: `2 + XLEN` bytes, or `UnexpectedEof { bytes_read: 10 }`;
   - `FNAME` and `FCOMMENT`: to the NUL, inclusive;
   - `FHCRC`: 2 bytes.
5. Under 8 bytes left: `gzip data truncated`, reason `truncated`.
6. The bytes up to `len-8` are inflated. The trailer must follow the stream at once (R3):
   - more after a trailer that starts `1f 8b` is another member: `multi_member`;
   - any other bytes there are `trailing`.
7. CRC-32 is checked, then ISIZE (little endian): `gzip CRC mismatch`, reason `crc`; then
   `gzip size mismatch`, reason `isize`.

Nothing panics. Every index is checked before it is read, and §5's must-fail rows cover each line.

### 3.4 The codec types

`DeflateCodec`, `DeflateCompressor` and `DeflateDecompressor`; the same three for `Zlib` and for `Gzip`.

- The codecs implement all three aspects; `deflate_codec(level)` and its siblings make them too.
- The compressors implement `Compressor`, with `new()` (level `Default`) and `with_level(l)`.
- The decompressors implement `Decompressor`, with `new()`.
- `algorithm()` is `Algorithm·Deflate`, `Algorithm·Zlib` or `Algorithm·Gzip`.

## 4. The wasm rule and memory

- No String or Vec is grown and then reset in a loop.
- Every output `Vec<u8>` is made once and pushed once per byte. The pinned compiler does not lower
  `Vec·with_capacity` on wasm (§7 K2), so the port does not use it.
- The hash tables are made once per call with `vec![0u32; 32768]`. They are not module-level, and
  there is no module-level lock.
- Large inputs are never copied to slice them. zlib and gzip inflate a range of the input in place:
  the decoder takes the input with a start and an end.
- There is no generic `≔ r: R` binding (SIGIL-1231 B2).
- No `Result` or `Option` is made per byte or per symbol. On wasm each one is an allocation that is
  never freed (§7 K8, **SIGIL-2153**): `inflate` makes one per block, not per symbol. Only the
  private inner loops avoid them; the public API keeps `Result`. GAPS.md G3 names the rites to
  revert once SIGIL-2153 is fixed. No String is built a char at a
  time either. The test consumer's hex is built as bytes and made a String once (http's GAPS D8).
- The volume rows (§5) measure the wasm heap at 1 MiB and 8 MiB. The README gives node's peak
  resident size per program.

## 5. The oracle and the rows

In `SLICE-1-ROWS.md`: the oracle (`tests/oracle/`, its two named patches), the corpus, every row kind
with its count, and `tests/run.sh`.

## 6–7. Departures and compiler limits

In `SLICE-1-DEPARTURES.md`. §6 says where Rust and the RFCs disagree, and which the port follows (the
planner's rulings R1–R5 among them). §7 lists the compiler limits met. GAPS.md, at the repo root,
keeps one row for each open defect, with the key of the repo whose owner must fix it.
