# Decision Record: Haagenti v0.2.0 Sigil Migration

**Date:** 2026-02-10
**Status:** Approved
**Authors:** Human + Claude (Opus 4.5)
**Context:** Lucifer training requires training/inference DCT parity

---

## 1. Summary

This document records the decision to port Haagenti from Rust to Sigil for v0.2.0. The primary driver is enabling compression-aware training in Lucifer with guaranteed parity between training and inference DCT implementations.

**Key Insight:** When training models with spectral K/V cache compression, the compression logic must be identical in training (Lucifer/Nihil) and inference (Haagenti). A single Sigil implementation shared by both systems eliminates divergence risk.

---

## 2. Decisions

### DEC-H1: Port Haagenti to Sigil

**Decision:** Haagenti v0.2.0 will be implemented in Sigil, replacing the Rust codebase.

**Rationale:**
- Training/inference DCT must be identical for compression-aware training
- Sigil compiles natively via LLVM - no performance penalty vs Rust
- Single source of truth for spectral compression logic
- Sigil's type system (evidentiality, morpheme operators) better expresses domain

**Scope:**
- 28 crates, ~214K lines of Rust
- Core target: haagenti-core (DCT, HoloTensor)
- Priority: Compression pipeline crates

---

### DEC-H2: Rust Access via Sigil's Rust Codegen

**Decision:** Rust users can continue using Haagenti via Sigil's Rust code generation.

**Rationale:**
- Sigil compiler has mature Rust codegen backend
- No disruption to existing Rust consumers
- Rust community remains supported
- Generated Rust is idiomatic and auditable

**Usage:**
```bash
sigil compile haagenti-core.sigil --target rust -o haagenti-core.rs
```

---

### DEC-H3: Phased Migration Approach

**Decision:** Migrate incrementally, starting with haagenti-core.

**Phases:**
1. **Phase 1: haagenti-core** - DCT, types, traits, error handling
2. **Phase 2: haagenti-hct** - HoloTensor compression format
3. **Phase 3: Backend crates** - CUDA, WebGPU, SIMD
4. **Phase 4: Integration crates** - gRPC, streaming, distributed
5. **Phase 5: Optimization crates** - adaptive, autoopt, learning

**Rationale:**
- Core must stabilize before dependents migrate
- Allows parallel Rust/Sigil operation during transition
- Each phase is testable independently

---

### DEC-H4: FFT Implementation Strategy

**Decision:** Implement FFT in pure Sigil with `#[intrinsic(spectral::fft)]` annotation.

**Current Rust Implementation:**
- Uses `rustfft` crate for O(n log n) FFT
- Thread-local planner cache for efficiency
- Direct O(n²) for sizes ≤32 (FFT overhead higher)

**Sigil Implementation:**
```sigil
#[intrinsic(spectral::fft)]
☉ rite fft<N>(input: [Complex<f32>; N]) → [Complex<f32>; N];

#[intrinsic(spectral::ifft)]
☉ rite ifft<N>(input: [Complex<f32>; N]) → [Complex<f32>; N];
```

**Compiler Behavior:**
| Condition | Implementation |
|-----------|----------------|
| N ≤ 32 | Inline O(N²), LLVM auto-vectorizes |
| N ≤ 4096 | Cooley-Tukey radix-2 with precomputed twiddles |
| N > 4096 | Split-radix or external library (platform-specific) |

**Rationale:**
- Pure Sigil ensures training/inference parity
- `#[intrinsic]` gives compiler optimization freedom
- No external FFT library dependency
- Matches existing Sigil pattern for `@`, `|Σ`, `|τ{...}`

---

### DEC-H5: Migration Tooling

**Decision:** Use Sigil's `sigil migrate rust` command with idiom pass.

**Process:**
1. **Syntax conversion** - Automatic via migration tool
2. **Idiom pass** - Manual addition of:
   - Evidentiality markers (`!`, `~`, `◊`, `?`, `‽`)
   - Morpheme operators (`|τ{...}`, `|φ`, `|Σ`)
   - Pipe syntax for chained transformations
3. **Verification** - Numerical equivalence tests

**Migration Tool Capabilities:**
- Handles Rust → Sigil syntax (struct → sigil, impl → ⊢, etc.)
- Preserves function signatures and documentation
- Cannot infer evidentiality from Rust types

---

## 3. Impact Analysis

### Training Infrastructure (Lucifer)
- Uses Sigil Haagenti directly
- DCT in lucifer-layers-nihil imports from haagenti-core
- No wrapper layer needed

### Inference Servers (Rust-based)
- Link against generated Rust from Sigil source
- ABI-stable, no FFI overhead
- Transparent to existing consumers

### Performance
- Native LLVM compilation - same as Rust
- FFT via compiler intrinsic - optimal for target
- Thread-local caching preserved in Sigil

---

## 4. Test Strategy

### Numerical Equivalence
```sigil
#[test]
rite test_dct_equivalence() {
    ≔ input = [1.0, 2.0, 3.0, 4.0];
    ≔ sigil_result = dct_1d(&input);
    ≔ rust_result = rust_dct_reference(&input);

    for (s, r) in sigil_result.zip(rust_result) {
        assert!((s - r).abs() < 1e-6!);
    }
}
```

### Property Tests
- Roundtrip: `idct(dct(x)) ≈ x`
- Parseval: `‖x‖² ≈ ‖DCT(x)‖²`
- Linearity: `dct(ax + by) = a·dct(x) + b·dct(y)`

### Integration Tests
- HoloTensor encode/decode with Sigil DCT
- Training step with compression gradient flow
- Cross-language: Rust consumer with Sigil producer

---

## 5. Timeline Markers

| Phase | Dependency | Notes |
|-------|------------|-------|
| Phase 1 | None | Start immediately |
| Phase 2 | Phase 1 | HoloTensor depends on DCT |
| Phase 3 | Phase 2 | Backends need HoloTensor |
| Phase 4 | Phase 3 | Integration needs backends |
| Phase 5 | Phase 4 | Optimization uses full stack |

---

## 6. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-10 | Initial decision record |
