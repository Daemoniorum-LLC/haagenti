# Haagenti Sigil Migration - Agent TDD Roadmap

## Current Status

**Pass Rate: 242/265 files (91.3%)**

This roadmap follows the Agent-TDD methodology: tests crystallize understanding, not coverage.

---

## Phase 1: Parse Error Fixes (9 files)

### P1.1: Attribute Syntax Errors

**Files:**
- `haagenti-grpc/src/tls.sg` - `//@ rune: from` in wrong context
- `haagenti-latent-cache/src/error.sg` - `//@ rune: from` enum variant field
- `haagenti-neural/src/error.sg` - `//@ rune: from` enum variant field

**Root Cause:** The `#[from]` attribute on enum variant fields was converted to `//@ rune: from` but placed inside the variant definition, not above it.

**Test Case:**
```sigil
//@ rune: test
rite test_enum_from_attribute() {
    // This should parse without error
    //@ rune: derive(Debug, Error)
    ᛈ TestError {
        //@ rune: error("IO error: {0}")
        //@ rune: from
        IoError(std·io·Error),
    }
}
```

**Fix Strategy:**
1. Move `//@ rune: from` above the variant, not inside parentheses
2. Or remove `#[from]` attributes entirely (Sigil may not need them)

---

### P1.2: Shift Operator in Expressions

**Files:**
- `haagenti-zstd/src/fse/encoder.sg` - `expected RParen, found Shl`
- `haagenti-zstd/src/fse/tans_encoder.sg` - `expected RParen, found Shl`

**Root Cause:** The `<<` shift operator in certain contexts confuses the parser.

**Test Case:**
```sigil
//@ rune: test
rite test_shift_in_expression() {
    ≔ x = 1 << 4;
    assert_eq!(x, 16);

    // Complex expression with shift
    ≔ y = (1 << 4) | (2 << 8);
    assert_eq!(y, 528);
}
```

**Fix Strategy:**
1. Identify specific pattern that fails
2. Wrap shifts in parentheses if needed
3. Or fix parser to handle `<<` in all contexts

---

### P1.3: Pattern Matching Issues

**Files:**
- `haagenti-zstd/src/block/sequences.sg` - `expected pattern, found ElementOf`

**Root Cause:** The `of` keyword (converted to `∈`) appears in a pattern context where it's not expected.

**Test Case:**
```sigil
//@ rune: test
rite test_for_pattern_with_tuple() {
    ≔ items = vec![(1, 2), (3, 4)];
    ∀ (a, b) ∈ items.iter() {
        println("{} {}", a, b);
    }
}
```

**Fix Strategy:**
1. Find the specific `∈` usage that fails
2. May need variable rename if `of` is used as identifier

---

### P1.4: Item Expected Errors

**Files:**
- `haagenti/src/lib.sg` - `expected item, found LineComment`
- `haagenti-zstd/src/compress/match_finder.sg` - parse error
- `haagenti-zstd/src/compress/speculative.sg` - `expected identifier`
- `haagenti-streaming/src/adaptive.sg` - parse error

**Root Cause:** Various comment/attribute placement issues.

**Fix Strategy:**
1. Check for remaining `// cfg(...)` patterns with different indentation
2. Remove any orphaned comments that break item parsing

---

## Phase 2: Type Error Fixes (6 files)

### P2.1: Type Mismatch Errors

**Files:**
- `haagenti-serverless/src/fragment_pool.sg` - `type mismatch in argument`
- `haagenti-mobile/src/quantization.sg` - `type mismatch`
- `haagenti-adaptive/tests/t7_adaptive.sg` - `type mismatch`

**Root Cause:** Type inference differs between Rust and Sigil, or explicit type annotations needed.

**Test Case:**
```sigil
//@ rune: test
rite test_vec_type_inference() {
    // Ensure Vec type is inferred correctly
    ≔ v: Vec<u8> = Vec·new();
    v.push(1u8);
    assert_eq!(v.len(), 1);
}
```

**Fix Strategy:**
1. Add explicit type annotations where inference fails
2. Check for numeric literal type issues (e.g., `1` vs `1u8`)

---

### P2.2: Index Type Errors

**Files:**
- `haagenti-zstd/src/block/literals.sg` - `index must be integer`
- `haagenti-zstd/src/huffman/encoder.sg` - `index must be integer`

**Root Cause:** Loop variable or index expression type not inferred as integer.

**Test Case:**
```sigil
//@ rune: test
rite test_array_indexing() {
    ≔ arr = [1, 2, 3, 4, 5];
    ∀ i ∈ 0..arr.len() {
        ≔ val = arr[i];
        assert!(val > 0);
    }
}
```

**Fix Strategy:**
1. Cast loop variables to `usize` if needed
2. Check `.len()` return type handling

---

### P2.3: Function Argument Errors

**Files:**
- `haagenti/src/hct_test_vectors.sg` - `expected at least N arguments`

**Root Cause:** Function call has wrong number of arguments.

**Fix Strategy:**
1. Check function signature vs call site
2. May be missing default arguments or incorrect conversion

---

## Phase 3: Runtime Error Fixes (4 files)

### P3.1: Missing Field/Method Errors

**Files:**
- `haagenti-grpc/src/main.sg` - `no field 'log_level' on struct`
- `haagenti-hct/tests/t05_random_projection.sg` - runtime error
- `haagenti-zstd/src/frame/checksum.sg` - runtime error

**Root Cause:** These files parse correctly but reference fields/methods that don't exist at runtime.

**Test Case:**
```sigil
//@ rune: test
rite test_struct_field_access() {
    Σ Config { log_level: String }
    ≔ c = Config { log_level: "debug".to_string() };
    assert_eq!(c.log_level, "debug");
}
```

**Fix Strategy:**
1. Verify struct definitions match usage
2. Check for missing imports or incorrect module paths

---

## Phase 4: CUDA FFI Reimplementation (4 files)

### P4.1: Pure Sigil CUDA Interface

**Files to rewrite from scratch:**
- `haagenti-cuda/src/cufft_ffi.sg` - cuFFT bindings
- `haagenti-cuda/src/neural_gpu.sg` - Neural network GPU ops
- `haagenti-cuda/src/stream.sg` - CUDA streams
- `haagenti-cuda/src/zstd_gpu.sg` - GPU Zstd compression

**Target Architecture:**
Use Sigil's native CUDA support (`Cuda·init()`, `Cuda·compile_kernel()`, etc.)

**Test Case:**
```sigil
//@ rune: test
rite test_cuda_basic() {
    ⎇ Cuda·init() {
        ≔ devices = Cuda·device_count();
        assert!(devices > 0);
        Cuda·cleanup();
    }
}
```

**Implementation Notes:**
- Sigil compiles directly to LLVM, bypassing C
- Use `Cuda·compile_kernel(source, name)` for JIT kernels
- Use `Cuda·malloc/free` for device memory
- Use `Cuda·memcpy_h2d/d2h` for transfers

---

## Execution Order

| Priority | Phase | Files | Effort |
|----------|-------|-------|--------|
| P0 | 1.1 Attribute Syntax | 3 | Low |
| P0 | 1.4 Item Expected | 4 | Low |
| P1 | 1.2 Shift Operator | 2 | Medium |
| P1 | 1.3 Pattern Issues | 1 | Low |
| P1 | 2.1 Type Mismatch | 3 | Medium |
| P1 | 2.2 Index Types | 2 | Low |
| P2 | 2.3 Arg Count | 1 | Low |
| P2 | 3.1 Runtime Errors | 3 | Medium |
| P3 | 4.1 CUDA Rewrite | 4 | High |

---

## Success Criteria

- [ ] 100% parse success (256/265 files, excluding 4 CUDA rewrites)
- [ ] 100% type check success on parsing files
- [ ] Runtime tests pass where applicable
- [ ] CUDA functionality preserved with native Sigil implementation

---

## Notes

- Tests are specifications, not coverage
- When a fix reveals deeper issues, update this roadmap
- Each phase should be independently committable
