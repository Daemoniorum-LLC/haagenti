# GAPS: haagenti in Sigil (LARES-932)

What the Sigil port could not do as it should, or what it found wrong elsewhere. Each row names who
must fix it. The spec is `specs/sigil/SLICE-1-CODEC-DEFLATE.md`; §6 and §7 are in
`SLICE-1-DEPARTURES.md`.

| Row | What | Owner | The port meanwhile | Revert when |
|---|---|---|---|---|
| G1 | **Rust haagenti's `deflate` panics** at every chained level when a match of 3 or more ends exactly at the input's end: `deflate.rs:341` hashes `pos + 1` after checking only `pos + 1 < len`. Repro: E row `r1_127` (bytes 0–123, then `01 02 03`) at `Default` (§6 D8 = R1). | Rust haagenti's owner | guards the read; expected bytes from `tests/oracle/patches/R1-lazy-guard.patch` | Rust's guard lands; then the patch goes |
| G2 | **Rust haagenti's `inflate` refuses valid streams** whose last Huffman code ends near the last byte: `HuffmanDecoder::decode` peeks the full table width. That is why 3 `#[test]`s are ignored, and Rust cannot round-trip its own `Fast`/`Default` output of a 64 KiB run (`tests/expected/departures.txt`) (§6 D1). | Rust haagenti's owner | decodes the complete code; `tests/oracle/patches/D1-huffman-peek.patch` | Rust's decoder is fixed |
| G3 | **SIGIL-2153: on wasm every `Result`/`Option` a call gives is never freed.** A rite giving `Result<u8, _>` or `Option<u8>`, called in a loop, traps after 1–2 million calls ("memory access out of bounds", or the 16 MiB heap exhausted). | sigil-lang (SIGIL-2153, P1) | `inflate`'s inner loops call `BitReader·hz_take`, `hz_byte`, `hz_need` and `HuffmanDecoder·hz_next`, which give numbers and mark an early end (`hz_is_short`), instead of `read_bits`, `read_byte` and `decode`. The rites are `hz_stored` and `hz_block` in `src/deflate/inflate.sg`. | SIGIL-2153 is fixed: call the `Result` rites again |
| G4 | **K1:** AOT and wasm do not lower a dependency tome's aspect default rite that returns a `Result`. | sigil-lang | each codec implements `verify_roundtrip` itself | fixed |
| G5 | **K9 (B2):** a subject-less aspect rite called through a generic, `C·with_level(l)`, is not lowered on AOT or wasm. | sigil-lang | the free makers `deflate_codec(level)`, `zlib_codec(level)`, `gzip_codec(level)` | fixed |
| G6 | **K2:** `Vec·with_capacity` is unresolved on wasm. | sigil-lang | `Vec·new()`, each byte pushed once | fixed |
| G7 | **K4–K7:** a full-path `◆` in a nested scroll unresolved on wasm; an AOT segfault on a `⤺` arm in a `≔`-bound match followed by a tail match; a generic's qualified `Compressor·algorithm(c)` taking the first instantiation on wasm; a generic's `c.algorithm().name()` printing an address on AOT. Repros in spec §7. | sigil-lang | the test consumer avoids each; the tome does not need them | fixed |
| G8 | **wasm at 8 MiB:** each volume row passes on wasm, but all four in one program exhaust node's default 2 GB JavaScript heap: qliphoth keeps vector contents on the JavaScript side. | qliphoth | `tests/run.sh` runs `mib8` one level per program | the runtime keeps vectors in linear memory |
| G9 | **Multi-member gzip** is refused as `multi_member` (R3); RFC 1952 allows members. | this repo, a later slice | refused, every time | members are supported |
