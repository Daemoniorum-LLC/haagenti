# haagenti in Sigil, slice 1: the oracle and the rows (LARES-932, spec §5)

The continuation of `SLICE-1-CODEC-DEFLATE.md`. Each ruling of contract 0.1.1 §6 has a row on run,
AOT and wasm (B8). Where a row cannot be run, `tests/run.sh` says COULD-NOT-LOOK.

## 5.1 The oracle

- **The generator.** `tests/oracle/` is a pinned, offline Rust program: its own `Cargo.lock`, flate2
  `=1.1.11`, and the workspace's `haagenti-deflate` by path. It writes `tests/expected/` and the
  consumer's generated data (`tests/consumer/src/gen/`). Both are committed, so no Rust runs at test
  time. `cargo run --release --offline -- --check` exits 1 when a committed file is stale.
- **Two named patches.** Each is Rust's code with one change, kept as a diff under
  `tests/oracle/patches/` and applied in `tests/oracle/src/`. The oracle checks each against Rust
  wherever Rust's answer is not the one patched, and against flate2 on every valid stream.

| Patch | File | Change | Rows |
|---|---|---|---|
| `D1-huffman-peek.patch` | `port_inflate.rs` | decode a complete code at the end of the input | R, F, M |
| `R1-lazy-guard.patch` | `port_deflate.rs` | guard the lazy step's `hash3(input, pos+1)` read | E |

- **R3, R4 and B4 live in the reference too.**
  - `port_inflate.rs` and `port_wrap.rs` refuse trailing bytes and a second member.
  - Every refusal carries the reason tag of spec §3.3, and the output bound is taken.
  - The oracle checks this against Rust's answer, kind and tag mapped from Rust's message. They may
    differ only on a D1, R3 or B4 row; each such row is listed in `tests/expected/departures.txt`.

## 5.2 The corpus

The same seeded rules are written in Rust (`tests/oracle/src/corpus.rs`) and in Sigil
(`tests/consumer/src/corpus.sg`).

- `empty`, `one` (`41`), `all256`;
- `digits`: `123456789`, whose CRC-32 `cbf43926` has its top bit set (B5);
- `run65535`, `run65536` (byte `61`): the stored-block boundary;
- `text` (16 KiB of seeded words), `random` (8 KiB of xorshift32);
- `packed`: flate2's gzip at level 9 of `text`, committed as hex;
- `mib1`, `mib8`: the word generator at 1 MiB and 8 MiB.

## 5.3 The rows

| Tag | What | Expected from |
|---|---|---|
| C | each corpus input × deflate, zlib, gzip × `None`, `Fast`, `Default`, `Best`, `Ultra`, `Custom(0)`, `Custom(5)`: the bytes, as hex up to 2 KiB, as `len crc32 adler32` above | Rust's bytes; the oracle checks that flate2 decodes each |
| R | C's `None`, `Fast` and `Default` bytes decoded back | the input |
| F | flate2 at levels 0, 1, 6, 9 (deflate) and 6 (zlib, gzip), decoded; the `mib*` inputs are left out, too large to commit | the input |
| E | R1: `r1_127` (bytes 0–123, then `01 02 03`), and `r1_128`, the same plus `ff`, at `Default`, `Best`, `Custom(0)`; also `d8_short` and `d8_random`; each with its round trip | Rust panics on all but `r1_128`: the R1 patch gives their bytes; `r1_128` is Rust's own |
| V, W | `mib1` and `mib8` at deflate `None`, `Fast`, `Default` and gzip `Default`; their round trip | Rust's bytes; the input |
| M | must-fail and edge inputs, with `ok len crc` or `err <category> <reason>` | the patched reference |
| A | the Codec surface: `algorithm`, `level`, `max_compressed_size`, `verify_roundtrip`, `from_level`, the free makers, each `Error`'s category, reason and recoverability | Rust's codec types |
| T | Rust's 31 `#[test]`s in `haagenti-deflate`, the 3 ignored ones included (B7: they fail in Rust, §6 D1) | each passes |

**The M rows** are:
- **§3.3's named cases:**
  - truncation, `BTYPE 3`, a bad stored length, bad Huffman tables, distances too far back;
  - zlib: CM, CINFO, FCHECK, FDICT, Adler;
  - gzip: magic, CM, CRC, ISIZE, FEXTRA, FNAME;
  - R3: trailing bytes after a raw stream, between a zlib stream and its Adler-32, between a gzip
    stream and its trailer, and after a gzip trailer; two identical gzip members and two different
    ones;
  - B4: a stream decoded under a bound one byte too small (`output_limit`) and under the exact
    bound (ok), in each format;
- every prefix of three small streams;
- 192 seeded single-bit flips.

## 5.4 `tests/run.sh`

- It runs the rows on `sigil run`, AOT and wasm (qliphoth's runtime at `tests/wasm-runtime.pin`, under
  node). It exits 0 for PASS, 1 for FAIL and 2 for COULD-NOT-LOOK.
- The wasm leg runs negative controls: the runner's status pass-through, a must-equal checksum
  program, a wrong expectation that must not match, and a must-trap program.
- `sigil run` is too slow for the C rows of `random` above level none and for the volume rows. They
  run only with `HAAGENTI_FULL=1`, and otherwise are COULD-NOT-LOOK on run (§7 K3).
