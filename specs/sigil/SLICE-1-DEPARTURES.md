# haagenti in Sigil, slice 1: departures and compiler limits (LARES-932)

The continuation of `SLICE-1-CODEC-DEFLATE.md` (§1–§5), split off to keep each file under 300 lines.

## 6. Where Rust and the RFCs disagree, and what the port does

- **D1 (follows the RFC): a Huffman code at the end of the input.**
  - Rust's `HuffmanDecoder·decode` peeks the table's full width (`max_bits`, up to 15) and refuses
    with `UnexpectedEof` when fewer bits remain. That happens even when the code that is there is
    shorter and complete. So Rust refuses valid streams whose final code ends near the last byte.
    This is the cause of the crate's three `#[ignore]`d tests (`test_inflate_repetitive`,
    `test_deflate_fast_repetitive` and `test_roundtrip_random`). flate2 decodes each of those
    streams.
  - RFC 1951 defines a stream by its codes, not by look-ahead, and §5 requires decoding flate2's
    output. So the port peeks what is there and pads the rest with zeros. When fewer than the table's
    width remain and the slot's code is longer than the bits there, or the slot is empty, it refuses
    with `UnexpectedEof`, as Rust does.
  - The port differs from Rust only where Rust says `UnexpectedEof`.
- **D2 (follows Rust): never matching position 0, and the single fixed-Huffman block.**
  - Both are valid DEFLATE (the RFC fixes no encoder). They are kept because §5 requires Rust's
    bytes.
  - Neither `Best` nor `Ultra` uses dynamic Huffman, and every `Custom(n)`, `Custom(0)` included, is
    compressed by the `Default` path. Only the zlib `FLEVEL` and gzip `XFL` differ.
- **D3 (follows Rust): code sets are not checked.**
  - RFC 1951 implies complete, prefix-free code sets, and zlib refuses over-subscribed and incomplete
    ones. Rust builds them anyway: a later code overwrites the table slots of an earlier one, and an
    empty slot is refused only when it is looked up.
  - A missing end-of-block code is not refused when the tables are built.
  - The port builds the same tables, so it accepts and refuses exactly the streams Rust does.
- **D4 (follows Rust): multi-member gzip.**
  - RFC 1952 §2.2 says a gzip file is a series of members.
  - Rust inflates only the first member's DEFLATE stream and checks the CRC and ISIZE in the last 8
    bytes of the whole input. Two members with different contents are therefore refused with
    `"gzip CRC mismatch"`, and two identical members decode to the first member alone.
  - The port does the same. Concatenating members is a later slice's question.
- **D5 (follows Rust): gzip header checks.**
  - `FHCRC` is skipped, not verified. The reserved `FLG` bits are not refused (RFC 1952 says a
    decoder must refuse them). `XFL`, `OS` and `MTIME` are ignored.
  - An unterminated `FNAME` or `FCOMMENT` runs to the end and is then refused as `"gzip data
    truncated"`.
- **D6 (follows Rust): trailers are read from the end of the input.** zlib takes Adler-32 from the
  last 4 bytes and gzip takes CRC-32 and ISIZE from the last 8, not from just after the DEFLATE
  stream. The bytes between the end of the stream and the trailer are ignored.
- **D7 (follows Rust): the zlib `FLEVEL` bits** are written as §3.2 says and ignored when read.

- **D8 (departs from Rust: Rust panics): the lazy step at the end of the input.**
  - Rust's `deflate_default` guards its lazy step with `pos + 1 < len` and then hashes the three
    bytes at `pos + 1`. When a match of at least 3 ends exactly at the input's end (`pos + 3 == len`),
    it reads `input[len]` and panics ("index out of bounds"). This happens at every chained level
    (`Default`, `Best`, `Ultra`, `Custom(n)`) on inputs of 100 bytes or more.
  - The repro is the E row `d8_short`: `ff`, the bytes 0–96, then `00 01 02`.
  - A port must not panic (contract §3), and Rust gives no bytes to match. So the port skips the lazy
    step when fewer than three bytes follow `pos + 1`. On every input where Rust does not panic, that
    is Rust's step exactly, and the oracle checks it on every C and V row.
  - The E rows give the port's bytes from the patched copy. flate2 decodes them.

## 7. Compiler limits met (reported, not worked around in the compiler)

These were met with sigil-lang develop `e5b8890` and qliphoth `52efb65`. Each was rechecked at
develop `7a493ec`, where each still reproduces. Each is the compiler's or the runtime's to fix. The
port's or the consumer's way around each is named, and none hides a row.

- **K1.** AOT and wasm do not lower an aspect's default rite that returns `Result<…>` when the aspect
  is in a dependency tome. `sigil run` runs it, and a single-file program lowers it.
  - The error is "cannot lower method `twice`: not a known function" on AOT, and an unresolved call
    on wasm.
  - Repro: a tome whose `aspect Dec { rite dec(&this, x: i64) → i64; rite twice(&this, x: i64) →
    Result<i64, String> { Ok(this.dec(x) * 2) } }` and its impl are called from a consumer as
    `a.twice(4)`.
  - Every type of §3.4 implements `decompress_with_size` and `verify_roundtrip` itself (§2.3).
- **K2.** `Vec·with_capacity` is unresolved on wasm ("1 call resolved to nothing"). The port uses
  `Vec·new()` and single pushes (§4).
- **K3.** `sigil run` interprets about 2×10^5 loop iterations a second. Unless `HAAGENTI_FULL=1`, the
  run leg leaves out:
  - `random` above level none;
  - the 64 KiB inputs, `text` and `packed` above `default`;
  - the volume rows.

  `tests/run.sh` reports them as COULD-NOT-LOOK on run, with their count, never as passing. AOT and
  wasm run every row.
- **K4.** On wasm, a `◆` constant reached by its full path in a nested scroll
  (`tome·gen·gen_f_0·GEN_F_0_PARTS`) is unresolved. The generated data gives its counts as rites.
- **K5.** AOT segfaults on `≔ c = ⌥ f() { Ok(x) => x, Err(_) => ⤺ nay };` followed by a tail
  `⌥ g(&c, &Δ out) { Ok(_) => …, Err(_) => nay }` in the same rite. `sigil run` gives the right
  answer. The test consumer binds `f()`'s `Result` and tests `is_err()` instead.
- **K6.** On wasm, inside a rite generic over `C: Codec`, the qualified call
  `Compressor·algorithm(c)` takes the first instantiation's impl for every type: `deflate deflate`
  for a `DeflateCodec` and a `GzipCodec`. The method form `c.algorithm()` is right.
- **K7.** On AOT, inside a rite generic over `C: Codec`, `c.algorithm().name()` bound to a local and
  printed gives an address, or an empty string when the local is annotated `String`. Run and wasm
  are right. The consumer's A rows reach the three codecs through an enum, not a generic rite.
- **K8.** On wasm (qliphoth `52efb65`), every `Result` or `Option` a call gives is an allocation that
  is never freed.
  - The heap is capped at 16 MiB, so a program that makes a few million of them traps ("memory
    access out of bounds" or "heap exhausted").
  - Repro: 400 calls of a rite that pushes 65,535 bytes, each from a rite giving `Ok(97u8)`.
  - A `String` grown a char at a time keeps every old buffer the same way (http's GAPS D8).
  - The port's hot path gives numbers and marks an early end on the reader (§3.1, §4), so `inflate`
    makes a `Result` per block, not per symbol.
