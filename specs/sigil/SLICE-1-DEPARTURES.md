# haagenti in Sigil, slice 1: departures and compiler limits (LARES-932)

The continuation of `SLICE-1-CODEC-DEFLATE.md` (§1–§4) and `SLICE-1-ROWS.md` (§5), split off to keep
each file under 300 lines. Contract 0.1.1 §6's rulings are named where they apply (R1–R5, B1–B8).

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
- **D4 = R3 (departs from Rust): multi-member gzip is refused, every time.**
  - RFC 1952 §2.2 says a gzip file is a series of members.
  - Rust inflates the first member's DEFLATE stream and checks the CRC and ISIZE in the last 8 bytes
    of the whole input. So two different members are refused as a CRC mismatch, and two identical
    members decode to the first member alone.
  - The port refuses both as `multi_member`. A tail that follows the first member's trailer and
    starts `1f 8b` is another member. Supporting members is a later slice.
- **D5 (follows Rust): gzip header checks.**
  - `FHCRC` is skipped, not verified. The reserved `FLG` bits are not refused (RFC 1952 says a
    decoder must refuse them). `XFL`, `OS` and `MTIME` are ignored.
  - An unterminated `FNAME` or `FCOMMENT` runs to the end and is refused as truncated.
- **D6 = R3 (departs from Rust): trailing bytes are refused, with the reason `trailing`.**
  - Rust ignores the bytes after a raw stream's final block. It also ignores the bytes between a zlib
    or gzip stream's end and the trailer it reads from the input's last 4 or 8 bytes.
  - The port refuses each, "never a silent truncation". The trailer must follow the stream at once,
    which is where Rust reads it for every stream it does not depart on.
- **D7 (follows Rust): the zlib `FLEVEL` bits** are written as §3.2 says and ignored when read.
- **D8 = R1 (departs from Rust: Rust panics): the lazy step at the end of the input.**
  - Rust's `deflate_default` guards its lazy step with `pos + 1 < len`, then hashes the three bytes at
    `pos + 1` (`deflate.rs:341`). When a match of 3 or more ends exactly at the input's end
    (`pos + 3 == len`), it reads `input[len]` and panics ("index out of bounds"). This happens at every
    chained level on inputs of 100 bytes or more.
  - The repros are the E rows `r1_127`, `d8_short` and `d8_random`.
  - The port guards the read: it skips the lazy step when fewer than three bytes follow `pos + 1`.
    Wherever Rust does not panic, that is Rust's step exactly. The oracle checks it on every C, V
    and E row, `r1_128` included.
  - The expected bytes come from `tests/oracle/patches/R1-lazy-guard.patch`. flate2 decodes them.
    GAPS.md records the defect for Rust haagenti's owner; the Rust crate is not edited here.
- **D9 = R4 (a test rule): refusals compare by kind and tag.** Rust tells a bad CRC, ISIZE, Adler,
  magic and header apart only by message text. The port keeps Rust's messages and adds the stable
  `reason()` tag of §3.3, which the tests compare.
- **D10 = R5, B1, B3, B4 (the surface):**
  - `Io` and `source` are dropped.
  - The `*_to` rites, `decompress_with_size`, `stats()` and `measure_ratio` are out of slice 1.
  - Decoding takes an output bound (16 MiB by default) that Rust lacks, refused as `output_limit`.
- **B7, measured.** Rust's `#[ignore]`d `test_inflate_repetitive` decodes flate2's dynamic block
  for 20 `A`s. At 28eb2cb it fails in Rust with `UnexpectedEof` (D1). `test_deflate_fast_repetitive`
  and `test_roundtrip_random` fail the same way, on Rust's own output. All three pass in the port
  (T rows).

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
  - `random` above level none (its chained levels take over half an hour);
  - the volume rows (1 MiB at `Default` takes 26 minutes to compress on run).

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
  - Filed as **SIGIL-2153** (P1; linked to SIGIL-2089). The planner reproduced it at sigil
    `bdec61de3`, with a trap at 1–2 million calls. GAPS.md G3 lists the rites to revert when it is
    fixed.
- **K9.** A subject-less aspect rite called through a generic (`C·with_level(l)` in a rite generic
  over `C: Codec`) checks and runs, but is not lowered on AOT ("cannot lower call through
  expression") or wasm (unresolved). By type path (`GzipCodec·with_level(l)`) it is lowered
  everywhere. The free makers `deflate_codec(level)` and its siblings are the generic-free way (B2).
