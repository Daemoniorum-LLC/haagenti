# Stash Recovery TODO - haagenti

**Date**: 2026-02-08
**Backup Location**: `/home/crook/stash-backup/`

## Applied Changes

### @{3} - haagenti-cuda - GPU Auto-Detection (APPLIED 2026-02-08)

**Applied from stash:**
- `crates/haagenti-cuda/src/dct_gpu.rs`:
  - Added `compute_capability_to_arch()` function (sm_50 through sm_100/Blackwell)
  - Added `get_compute_capability()` and `get_device_name()` methods to GpuDctContext
  - Updated `ensure_kernels_loaded()` to auto-detect GPU architecture
  - Logs detected compute capability and GPU name on kernel compilation

**Benefit**: CUDA kernels now compile with optimal architecture flags for the detected GPU instead of hardcoded sm_52 fallback.

### @{3} - Turbo Pipeline OutputFormat (APPLIED 2026-02-08)

**Applied from stash:**
- `crates/haagenti/src/pipeline/turbo.rs`:
  - Added `OutputFormat` enum (Safetensors, HctDirectory)
  - Added `output_format` field to `TurboConfig`
  - Increased `max_tensor_size` default to 300M elements (for 70B+ models)
  - Added `write_hct_directory()` function for individual .hct file output
  - Creates `manifest.json` with tensor metadata

- `crates/haagenti/src/pipeline/mod.rs`:
  - Exported `OutputFormat`

- `crates/haagenti/examples/compress_turbo.rs`:
  - Added `--output-format` / `-f` CLI option (safetensors, hct-dir)
  - Added `--max-size` CLI option
  - Updated status display and help text

**Benefit**: Enables outputting compressed models as individual .hct files with manifest, compatible with TieredHoloLoader for progressive/tiered loading.

## Remaining Changes from @{3}

The following changes were NOT applied and should be reviewed:

### Retention Ratio (Lossy Compression) - COMPLEX

The stash adds `retention_ratio` to SpectralEncoder for lossy compression. This requires:
- Sorting coefficients by magnitude (importance order)
- Truncating to keep only top N%
- Distributing retained coefficients across fragments

**Issue**: Current V3 format uses raster order, not importance order. Full integration requires rethinking the encoding format or maintaining two code paths.

**Files affected:**
- `crates/haagenti-hct/src/holotensor.rs` - SpectralEncoder, HoloTensorEncoder

**Workaround**: The turbo pipeline uses `CompressiveSpectralEncoder` which already has retention via its constructor parameter.

### Other Changes (Low Priority)
- `CLAUDE.md` - Documentation updates
- `crates/haagenti-cuda/Cargo.toml` - CUDA 12.x backward compatibility note
- `crates/haagenti/Cargo.toml` - Feature flags
- `crates/haagenti-zstd/src/fse/encoder.rs` - FSE encoder changes (experimental)
- `crates/haagenti-zstd/examples/trace_encoder.rs` - Debug tracing

## Other Stashes

### @{4} - haagenti-fft

Patch: `stash-4-haagenti-fft.patch`

Changes:
- `crates/haagenti-zstd/examples/trace_decode.rs`
- `crates/haagenti-zstd/examples/trace_sequences.rs`

**Status**: Debug/trace examples, low priority.

### @{7} - GEMV kernels

Patch: `stash-7-gemv-kernels.patch`

Changes:
- `Cargo.toml`

**Status**: Likely workspace config, review if needed.

## Summary of Applied Work

| Feature | Status | Benefit |
|---------|--------|---------|
| GPU compute capability auto-detection | DONE | Optimal NVRTC compilation for any GPU |
| OutputFormat enum | DONE | API for format selection |
| HctDirectory output | DONE | Individual .hct files + manifest |
| max_tensor_size increase | DONE | 300M elements for 70B+ models |
| compress_turbo CLI options | DONE | --output-format, --max-size flags |
| retention_ratio in SpectralEncoder | DEFERRED | Complex encoding changes needed |
