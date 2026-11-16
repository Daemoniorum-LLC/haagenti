# haagenti in Sigil, slice 1: the Codec surface, DEFLATE, zlib, gzip (LARES-932)

Status: the spec for slice 1, written before the code. The Rust crates `haagenti-core` and
`haagenti-deflate` at develop `28eb2cbd9dfe59d9b4ca273b4f1312ebb426420a` are the oracle; RFCs 1950,
1951 and 1952 are the formats. Where the two disagree, §6 says which one the port follows and why.
Every name is the port's; behaviour (the bytes produced, the errors refused) is Rust's unless §6 says
otherwise.

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

`Result<T, Error>` is written out everywhere; there is no alias. `☉ ᛈ Error` has Rust's eleven
variants, each with named fields:

| Variant | Fields | `category()` | `message()` (Rust's `Display`) |
|---|---|---|---|
| `CorruptedData` | `message: String` | `corrupted_data` | `corrupted data: {message}` |
| `BufferTooSmall` | `required: usize, provided: usize` | `buffer_too_small` | `buffer too small: need {required} bytes, got {provided}` |
| `InvalidLevel` | `level: i32, min: i32, max: i32` | `invalid_level` | `invalid compression level {level}: must be in range [{min}, {max}]` |
| `InvalidDictionary` | `message: String` | `invalid_dictionary` | `invalid dictionary: {message}` |
| `ChecksumMismatch` | `expected: u32, actual: u32` | `checksum_mismatch` | `checksum mismatch: expected 0x{expected:08x}, got 0x{actual:08x}` |
| `UnexpectedEof` | `bytes_read: usize` | `unexpected_eof` | `unexpected EOF after {bytes_read} bytes` |
| `Io` | `message: String` | `io_error` | `I/O error: {message}` |
| `AllocationFailed` | `requested_bytes: usize` | `allocation_failed` | `allocation failed: could not allocate {requested_bytes} bytes` |
| `Algorithm` | `algorithm: String, message: String` | `algorithm_error` | `{algorithm} error: {message}` |
| `InvalidState` | `expected: String, actual: String` | `invalid_state` | `invalid state: expected {expected}, got {actual}` |
| `Unsupported` | `message: String` | `unsupported` | `unsupported: {message}` |

The makers are `Error·corrupted(m)`, `Error·corrupted_at(m, offset)` (message `"{m} at offset
{offset}"`), `Error·buffer_too_small(required, provided)`, `Error·checksum_mismatch(expected,
actual)`, `Error·unexpected_eof(bytes_read)`, `Error·io(m)` and `Error·algorithm(a, m)`.
`is_recoverable()` is `yea` for `UnexpectedEof` and `BufferTooSmall` only.

Two departures from Rust's type, neither of them a behaviour: `Io` holds the message, not a
`std::io::Error`, and `CorruptedData` has no boxed `source`. Rust's `&'static str` fields become
`String`.

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
    rite compress_to(&this, input: &Vec<u8>, output: &Δ Vec<u8>) → Result<usize, Error>;
    rite max_compressed_size(&this, input_len: usize) → usize;
}
☉ aspect Decompressor {
    rite algorithm(&this) → Algorithm;
    rite decompress(&this, input: &Vec<u8>) → Result<Vec<u8>, Error>;
    rite decompress_to(&this, input: &Vec<u8>, output: &Δ Vec<u8>) → Result<usize, Error>;
    rite decompress_with_size(&this, input: &Vec<u8>, output_size: usize) → Result<Vec<u8>, Error>;
}
☉ aspect Codec: Compressor + Decompressor {
    rite new() → This;
    rite with_level(level: CompressionLevel) → This;
    rite verify_roundtrip(&this, data: &Vec<u8>) → Result<bool, Error>;
}
```

- `compress_to` and `decompress_to` treat `output` as Rust treats `&mut [u8]`: its length is the
  buffer's size. They never grow or shrink it. They write the result into `output[0..n]` and give
  `n`. When the result is longer than `output.len()` they refuse with
  `BufferTooSmall { required: n, provided: output.len() }` and leave `output` unchanged.
- `decompress_with_size(input, s)`: Rust makes a zeroed buffer of `s`, calls `decompress_to` and
  truncates. The port gives the same answer: the decompressed bytes when there are at most `s`, and
  `BufferTooSmall { required, provided: s }` otherwise.
- `verify_roundtrip(data)` compresses, decompresses and gives `Ok(decompressed == data)`, passing
  either step's error through.
- Rust gives `decompress_with_size` and `verify_roundtrip` default bodies. The port declares them in
  the aspect with the same bodies, and every type of §3.4 also implements them itself. The reason is a
  compiler defect (§7 K1): AOT and wasm do not lower a dependency tome's default rite that returns a
  `Result`.
- `stats()`, `measure_ratio`, `CompressionStats` and `CompressionRatio` are not ported. Deflate fills
  no stats: every `stats()` in the crate gives `None`.
- The streaming, dictionary, parallel and SIMD aspects and `dct.rs` are out of this slice, since
  deflate implements none of them.

## 3. `haagenti·deflate`

### 3.1 The free rites

| Rite | Gives |
|---|---|
| `deflate(input: &Vec<u8>, level: CompressionLevel)` | `Result<Vec<u8>, Error>`: raw RFC 1951 |
| `inflate(input: &Vec<u8>, output: &Δ Vec<u8>)` | `Result<(), Error>`: appends to `output` |
| `inflate_to(input: &Vec<u8>, output: &Δ Vec<u8>)` | `Result<usize, Error>`, with `compress_to`'s buffer rule |
| `zlib_compress(input, level)`, `zlib_decompress(input)`, `zlib_decompress_to(input, output)` | RFC 1950 |
| `gzip_compress(input, level)`, `gzip_decompress(input)`, `gzip_decompress_to(input, output)` | RFC 1952 |
| `crc32(data: &Vec<u8>) → u32`, `crc32_range(data, start, end)`, `adler32(data) → u32` | the checksums |

`crc32` and `adler32` are private in Rust. They are public here because a gzip or zlib framer, such as
http's `Encoder`, needs them, and because the tests check them against known vectors.

The pieces are public as in Rust (`huffman`): `BitWriter`, `HuffmanEncoder` (`code(s)`, `len(s)`),
`HuffmanDecoder` (`from_lengths`, `table_bits`, `decode`), `reverse_bits`, and the tables in
`haagenti·deflate·tables`. `BitReader` is a cursor, not an owner: a `sigil` cannot hold a borrow, so
each read is given the bytes (`BitReader·over(data)`, `r.read_bits(data, n)`). Beside the
`Result`-giving reads, the hot path has `take`, `byte` and `HuffmanDecoder·next`. These give a number
and mark an early end on the reader, which the caller tests at once (`is_short()`); `next` gives
`HUFFMAN_INVALID` for an empty slot. They exist for wasm (§4, §7 K8).

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

`inflate` reads blocks until one has `BFINAL` set. Bytes after that block are ignored. The refusals,
with their exact messages:

| Condition | Error |
|---|---|
| input ends inside a header, a length or extra bits, a stored block, or a Huffman code (§6 D1) | `UnexpectedEof { bytes_read }`, where `bytes_read` is the inflate input's length |
| `BTYPE = 3` | `corrupted("invalid block type 3")` |
| stored `LEN ≠ !NLEN` | `corrupted("stored block length mismatch")` |
| `HLIT > 286` or `HDIST > 30` | `corrupted("invalid code count")` |
| code-length symbol 16 first | `corrupted("repeat at start")` |
| a 16, 17 or 18 run past `HLIT + HDIST` | `corrupted("repeat overflow")`, `("zero repeat overflow")`, `("long zero repeat overflow")` |
| a lookup that hits an empty table slot | `corrupted("invalid huffman code")` |
| literal/length symbol 286 or 287 | `corrupted("invalid literal/length code")` |
| distance symbol 30 or 31 | `corrupted("invalid distance code")` |
| distance greater than the output so far | `corrupted("distance beyond output")` |

`HuffmanDecoder·from_lengths` refuses only an empty length list (`"empty code lengths"`). All zero
lengths build a two-slot table of empty slots, and any lookup in it is `"invalid huffman code"`.
Over-subscribed and incomplete code sets are built as Rust builds them (§6 D3).

**zlib** (`zlib_decompress`), in order:
1. Fewer than 6 bytes is `"zlib data too short"`.
2. `CM ≠ 8` is `"unsupported compression method"`.
3. `CINFO > 7` is `"invalid window size"`.
4. `(CMF·256 + FLG) mod 31 ≠ 0` is `"zlib header checksum failed"`.
5. `FDICT` set is `"preset dictionary not supported"`.
6. The bytes `2 .. len-4` are inflated; an error there is passed through.
7. The last four bytes, big endian, must equal the Adler-32 of the output, or it is
   `"zlib checksum mismatch"`.

**gzip** (`gzip_decompress`), in order:
1. Fewer than 18 bytes is `"gzip data too short"`.
2. A first two bytes other than `1f 8b` is `"invalid gzip magic"`.
3. `CM ≠ 8` is `"unsupported compression method"`.
4. The optional header fields are skipped:
   - `FEXTRA`: `UnexpectedEof { bytes_read: 10 }` if its length runs off the end; otherwise
     `2 + XLEN` bytes;
   - `FNAME` and `FCOMMENT`: to the NUL, inclusive;
   - `FHCRC`: 2 bytes.
5. If fewer than 8 bytes remain, it is `"gzip data truncated"`.
6. The bytes from there to `len-8` are inflated.
7. CRC-32, then ISIZE, are checked against the **last 8 bytes of the input**:
   `"gzip CRC mismatch"`, then `"gzip size mismatch"`.

Every refusal is a `CorruptedData` unless the table says `UnexpectedEof`. Nothing panics. Every index
is checked before it is read, and §5's must-fails cover each row.

### 3.4 The codec types

`DeflateCodec`, `DeflateCompressor` and `DeflateDecompressor`; the same three for `Zlib` and for `Gzip`.

- The codecs implement all three aspects.
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
  never freed (§7 K8): `inflate` makes one per block, not per symbol. No String is built a char at a
  time either. The test consumer's hex is built as bytes and made a String once (http's GAPS D8).
- The volume rows (§5) measure the wasm heap at 1 MiB and 8 MiB. The README gives node's peak
  resident size per program.

## 5. The oracle and the rows

- **The generator.** `tests/oracle/` is a pinned, offline Rust program (its own `Cargo.lock`, flate2
  `=1.1.11`, the workspace's `haagenti-deflate` by path). It writes `tests/expected/`, which is
  committed, so no Rust runs at test time. The corpus is generated by the same seeded rules in Rust
  and in Sigil (`tests/consumer/src/corpus.sg`):
  - `empty`, `one` (`41`), and `all256`;
  - `run65535` and `run65536` (byte `61`), the stored-block boundary;
  - `text` (16 KiB of seeded words) and `random` (8 KiB of xorshift32);
  - `packed`, flate2's gzip at level 9 of `text`, committed as hex;
  - `mib1` and `mib8`, the word generator at 1 MiB and 8 MiB.
- **C rows.** For each corpus input, each format (deflate, zlib, gzip) and each level (`None`, `Fast`,
  `Default`, `Best`, `Ultra`, `Custom(0)`, `Custom(5)`), Rust haagenti's bytes. The row is the full
  hex up to 2 KiB, and `len crc32 adler32` above that. The oracle also checks that flate2 decodes
  every one of them to the input. The port's output must equal the row, so flate2 decodes the port's
  output too.
- **F rows.** flate2's output at levels 0, 1, 6 and 9 (deflate), and at 6 (zlib and gzip), committed
  as hex. The port must decode each to the input. The `mib*` inputs are left out because their flate2
  bytes are too large to commit.
- **E rows.** Two inputs on which Rust's `deflate` panics at every chained level (§6 D8). The port's
  bytes come from `tests/oracle/src/port_deflate.rs`, Rust's encoder with that one guard. flate2
  decodes them, and the round trip is checked too. The oracle checks the copy against Rust's
  `deflate` on every C and V row.
- **V and W rows.** `mib1` and `mib8` at deflate `None`, `Fast` and `Default` and at gzip `Default`:
  Rust's bytes (V) and the port's round trip (W).
- **A rows.** The Codec surface, taken from Rust's codec types:
  - `algorithm`, `level` and `max_compressed_size`;
  - `verify_roundtrip`;
  - `compress_to`, `decompress_to` and `decompress_with_size` on buffers too small, exact and large;
  - `from_level`;
  - each `Error`'s category, recoverability and text.
- **M rows: must-fail.**
  - The named cases of §3.3: truncation, `BTYPE 3`, bad Huffman tables, distance too far back, bad
    stored length, bad zlib CM, CINFO, FCHECK, FDICT and Adler, bad gzip magic, CM, CRC, ISIZE,
    FEXTRA and truncation, and gzip with two members.
  - Every prefix of three small streams (one per format).
  - 192 seeded single-bit flips.
  - The expected answer is the category and message, or `ok len crc32`.
- **T rows.** Rust's 31 `#[test]`s in `haagenti-deflate`, the 3 ignored ones included, with the same
  inputs and checks. `haagenti-core` has no `#[test]` outside `dct.rs`.
- **The patched reference.** The expected answers to decoding come from
  `tests/oracle/src/port_inflate.rs`. That file is Rust's `inflate`, `zlib_decompress` and
  `gzip_decompress` with the one change of §6 D1. The oracle checks it against Rust's `inflate` on
  every M and F row whose Rust answer is not `UnexpectedEof`, and against flate2 on every valid
  stream.
- **Counts.** T 31, A 82, M 311, F 48, C 168, R 72, E 12, V and W 16, all in `tests/expected/`.
  `tests/expected/d1.txt` lists the 12 R rows where Rust's own `inflate` refuses Rust's own bytes
  (§6 D1).
- `tests/run.sh` runs the rows on `sigil run`, AOT and wasm (qliphoth's runtime at
  `tests/wasm-runtime.pin`). It exits 0 for PASS, 1 for FAIL and 2 for COULD-NOT-LOOK. It also runs
  the wasm negative controls. `cargo run --release --offline -- --check` in `tests/oracle/` exits 1
  when a committed file is not what the oracle writes.

## 6–7. Departures and compiler limits

In `SLICE-1-DEPARTURES.md`: §6 says where Rust and the RFCs disagree and which the port follows; §7 lists the compiler limits met.
