# TDD Roadmap: Haagenti Sigil Migration

**Version:** 1.0.0
**Date:** 2026-02-10
**Methodology:** Agent-TDD (tests as crystallized understanding)
**Related:** DEC-2026-02-10-sigil-migration.md

---

## Overview

This roadmap defines the test-driven migration of Haagenti from Rust to Sigil. Each phase follows the Agent-TDD cycle: UNDERSTAND → SPECIFY → IMPLEMENT → VERIFY.

**Key Principle:** Tests must verify numerical equivalence with the Rust reference implementation. The Sigil version must produce bit-identical results (within floating-point tolerance) for all operations.

---

## Phase 1: haagenti-core Foundation

### P1.1: FFT/IFFT Implementation

**Goal:** Pure Sigil FFT with `#[intrinsic]` annotation

**RED Tests (Write First):**

```sigil
// tests/t1_fft_core.sigil

#[test]
rite test_fft_roundtrip_power_of_2() {
    // Property: IFFT(FFT(x)) = x
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        ≔ input = random_complex_vector(n);
        ≔ freq = fft(&input);
        ≔ recovered = ifft(&freq);
        assert_complex_vectors_close(&input, &recovered, epsilon: 1e-6!);
    }
}

#[test]
rite test_fft_parseval() {
    // Property: ‖x‖² = ‖FFT(x)‖² / N (energy preservation)
    ≔ input = [Complex·new(1.0, 0.0), Complex·new(2.0, 0.0),
               Complex·new(3.0, 0.0), Complex·new(4.0, 0.0)];
    ≔ freq = fft(&input);

    ≔ input_energy = input|τ{|z| z.norm_sqr()}|Σ;
    ≔ freq_energy = freq|τ{|z| z.norm_sqr()}|Σ / 4.0;

    assert!((input_energy - freq_energy).abs() < 1e-6!);
}

#[test]
rite test_fft_linearity() {
    // Property: FFT(ax + by) = a·FFT(x) + b·FFT(y)
    ≔ x = random_complex_vector(16);
    ≔ y = random_complex_vector(16);
    ≔ a = Complex·new(2.0, 1.0);
    ≔ b = Complex·new(-1.0, 0.5);

    ≔ combined = x.zip(&y)|τ{|(xi, yi)| a * xi + b * yi}|collect;
    ≔ fft_combined = fft(&combined);

    ≔ fft_x = fft(&x);
    ≔ fft_y = fft(&y);
    ≔ expected = fft_x.zip(&fft_y)|τ{|(xi, yi)| a * xi + b * yi}|collect;

    assert_complex_vectors_close(&fft_combined, &expected, epsilon: 1e-5!);
}

#[test]
rite test_fft_vs_rust_reference() {
    // Numerical equivalence with Rust rustfft output
    ≔ input = [1.0 + 0.0i, 2.0 + 0.0i, 3.0 + 0.0i, 4.0 + 0.0i];
    ≔ sigil_fft = fft(&input);

    // Known rustfft output for this input
    ≔ rust_reference = [
        Complex·new(10.0, 0.0),
        Complex·new(-2.0, 2.0),
        Complex·new(-2.0, 0.0),
        Complex·new(-2.0, -2.0),
    ]!;

    assert_complex_vectors_close(&sigil_fft, &rust_reference, epsilon: 1e-6!);
}
```

**Test Count:** 8 tests
- Roundtrip for power-of-2 sizes (1 parametric)
- Roundtrip for non-power-of-2 sizes (1 parametric)
- Parseval's theorem (1)
- Linearity (1)
- Convolution theorem (1)
- DC component (1)
- Nyquist component (1)
- Rust reference comparison (1)

---

### P1.2: DCT/IDCT Implementation

**Goal:** DCT-II and DCT-III (IDCT) via FFT

**RED Tests:**

```sigil
// tests/t1_dct_core.sigil

#[test]
rite test_dct_roundtrip() {
    for n in [4, 8, 16, 32, 64, 128, 256] {
        ≔ input = random_f32_vector(n);
        ≔ freq = dct_1d(&input);
        ≔ recovered = idct_1d(&freq);
        assert_vectors_close(&input, &recovered, epsilon: 1e-5!);
    }
}

#[test]
rite test_dct_parseval() {
    // Orthonormal DCT preserves energy
    ≔ input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    ≔ freq = dct_1d(&input);

    ≔ input_energy = input|τ{|x| x * x}|Σ;
    ≔ freq_energy = freq|τ{|x| x * x}|Σ;

    assert!((input_energy - freq_energy).abs() < 1e-5!);
}

#[test]
rite test_dct_dc_component() {
    // DC component is scaled mean
    ≔ input = [1.0, 2.0, 3.0, 4.0];
    ≔ freq = dct_1d(&input);
    ≔ expected_dc = input|Σ / (input.len() as f32).sqrt();
    assert!((freq[0] - expected_dc).abs() < 1e-5!);
}

#[test]
rite test_dct_vs_rust_reference() {
    // Numerical equivalence with Rust haagenti-core output
    ≔ input = [1.0f32, 2.0, 3.0, 4.0];
    ≔ sigil_dct = dct_1d(&input);

    // Known haagenti-core dct_1d output for this input
    ≔ rust_reference = [5.0f32, -2.23044, 0.0, -0.15851]!;

    assert_vectors_close(&sigil_dct, &rust_reference, epsilon: 1e-4!);
}

#[test]
rite test_dct_2d_roundtrip() {
    ≔ input = random_f32_matrix(8, 8);
    ≔ freq = dct_2d(&input, 8, 8);
    ≔ recovered = idct_2d(&freq, 8, 8);
    assert_matrices_close(&input, &recovered, epsilon: 1e-4!);
}
```

**Test Count:** 12 tests
- 1D roundtrip (1 parametric)
- 1D Parseval (1)
- 1D DC component (1)
- 1D linearity (1)
- 1D edge cases (3: zeros, single element, large)
- Rust reference comparison (1)
- 2D roundtrip (1)
- 2D separability verification (1)
- Direct vs FFT equivalence (1)
- f64 precision tests (1)

---

### P1.3: Core Types

**Goal:** Port `types.rs`, `traits.rs`, `error.rs`

**RED Tests:**

```sigil
// tests/t1_types.sigil

#[test]
rite test_compression_stats_default() {
    ≔ stats = CompressionStats·default();
    assert_eq!(stats.original_size, 0!);
    assert_eq!(stats.compressed_size, 0!);
    assert!(stats.compression_ratio().is_nan()!);
}

#[test]
rite test_compression_stats_ratio() {
    ≔ stats = CompressionStats {
        original_size: 1000!,
        compressed_size: 250!,
        ...CompressionStats·default()
    };
    assert!((stats.compression_ratio() - 4.0).abs() < 1e-6!);
}

#[test]
rite test_compression_error_display() {
    ≔ err = CompressionError·InvalidDimension { expected: 128!, got: 64! };
    assert!(err.to_string().contains("128")!);
    assert!(err.to_string().contains("64")!);
}
```

**Test Count:** 8 tests
- CompressionStats creation and methods (3)
- Error types and Display impl (3)
- Trait implementations (2)

---

### P1.4: Stream Processing

**Goal:** Port `stream.rs` with Sigil idioms

**RED Tests:**

```sigil
// tests/t1_stream.sigil

#[test]
rite test_stream_chunk_iterator() {
    ≔ data = [1, 2, 3, 4, 5, 6, 7, 8];
    ≔ chunks = data|chunks(3)|collect;
    assert_eq!(chunks.len(), 3!);
    assert_eq!(chunks[0], [1, 2, 3]!);
    assert_eq!(chunks[1], [4, 5, 6]!);
    assert_eq!(chunks[2], [7, 8]!);
}

#[test]
rite test_stream_windowed() {
    ≔ data = [1, 2, 3, 4, 5];
    ≔ windows = data|windows(3)|collect;
    assert_eq!(windows.len(), 3!);
    assert_eq!(windows[0], [1, 2, 3]!);
    assert_eq!(windows[1], [2, 3, 4]!);
    assert_eq!(windows[2], [3, 4, 5]!);
}
```

**Test Count:** 6 tests
- Chunk iteration (2)
- Windowed iteration (2)
- Edge cases (2)

---

## Phase 1 Summary

| Module | Test Count | Priority |
|--------|------------|----------|
| P1.1 FFT | 8 | Critical |
| P1.2 DCT | 12 | Critical |
| P1.3 Types | 8 | High |
| P1.4 Stream | 6 | Medium |
| **Total** | **34** | |

**Exit Criteria:**
1. All 34 tests pass
2. Numerical equivalence with Rust within ε = 1e-5
3. No external FFT library dependency
4. Clean `sigil compile` with no warnings

---

## Phase 2: haagenti-hct (HoloTensor)

### P2.1: HoloTensor Format

**Goal:** Port holographic tensor compression format

**RED Tests:**

```sigil
// tests/t2_holotensor.sigil

#[test]
rite test_holotensor_spectral_roundtrip() {
    ≔ tensor = random_tensor([4, 128, 128]);
    ≔ encoded = HoloTensor·encode_spectral(&tensor, retain_ratio: 0.5);
    ≔ decoded = encoded|decode;

    // Lossy compression - verify error bounded
    ≔ mse = (&tensor - &decoded)|τ{|x| x * x}|mean;
    assert!(mse < 0.01~);  // ~reported, depends on data
}

#[test]
rite test_holotensor_lrdf_roundtrip() {
    ≔ tensor = random_tensor([4, 128, 128]);
    ≔ encoded = HoloTensor·encode_lrdf(&tensor, rank: 16);
    ≔ decoded = encoded|decode;

    ≔ mse = (&tensor - &decoded)|τ{|x| x * x}|mean;
    assert!(mse < 0.05~);
}
```

**Test Count:** 18 tests
- Spectral encoding/decoding (4)
- LRDF encoding/decoding (4)
- Random projection encoding/decoding (4)
- Format detection (2)
- Serialization (2)
- Error handling (2)

---

### P2.2: Compression Pipeline

**Goal:** End-to-end compression with gradient support

**RED Tests:**

```sigil
// tests/t2_pipeline.sigil

#[test]
rite test_compression_gradient_flow() {
    ≔ tensor = random_tensor([4, 64]).requires_grad();
    ≔ compressed = compress_spectral(&tensor, retain: 0.8);
    ≔ decompressed = decompress(&compressed);
    ≔ loss = decompressed|mean;

    ≔ grads! = loss|∇;
    assert!(grads!.get(&tensor).is_some()!);
    assert!(grads!.get(&tensor).unwrap()|all{|g| g.is_finite()}!);
}
```

**Test Count:** 12 tests

---

## Phase 2 Summary

| Module | Test Count | Priority |
|--------|------------|----------|
| P2.1 HoloTensor | 18 | Critical |
| P2.2 Pipeline | 12 | Critical |
| **Total** | **30** | |

---

## Phase 3: Backend Crates

### P3.1: SIMD Backend

**Test Count:** 10 tests
- F32x16 operations
- Alignment verification
- Fallback behavior

### P3.2: CUDA Backend

**Test Count:** 15 tests
- GPU DCT
- Memory management
- Stream synchronization

### P3.3: WebGPU Backend

**Test Count:** 10 tests
- Compute shader compilation
- Buffer management

---

## Phase 3 Summary

| Module | Test Count | Priority |
|--------|------------|----------|
| P3.1 SIMD | 10 | High |
| P3.2 CUDA | 15 | High |
| P3.3 WebGPU | 10 | Medium |
| **Total** | **35** | |

---

## Complete Test Summary

| Phase | Tests | Description |
|-------|-------|-------------|
| Phase 1 | 34 | Core (FFT, DCT, types) |
| Phase 2 | 30 | HoloTensor format |
| Phase 3 | 35 | Backends |
| Phase 4 | 20 | Integration (gRPC, streaming) |
| Phase 5 | 15 | Optimization (adaptive, autoopt) |
| **Total** | **134** | |

---

## Verification Strategy

### Rust Reference Oracle

For each Sigil function, maintain a Rust reference test:

```bash
# Generate reference outputs
cargo test --package haagenti-core -- --nocapture > reference_outputs.json

# Verify Sigil matches
sigil test tests/ --compare-with reference_outputs.json
```

### Continuous Integration

```yaml
# .github/workflows/sigil-migration.yml
- name: Run Sigil tests
  run: sigil test tests/

- name: Verify Rust equivalence
  run: |
    cargo build --release
    sigil test tests/equivalence/
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-10 | Initial TDD roadmap |
