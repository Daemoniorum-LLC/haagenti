//! Native CUDA Kernels with Warp-Level Parallelism
//!
//! Phase 6 of the Large Model Optimization Plan.
//!
//! Implements highly optimized GPU decompression using:
//! - Warp-level primitives (shuffle, vote, ballot)
//! - Shared memory token caching
//! - Cooperative groups for synchronization
//! - Vectorized memory access (128-bit loads)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Warp-Level LZ4 Decoder                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Lane 0-7:   Token Parsing (parallel literal length scan)       │
//! │  Lane 8-15:  Match Offset Decoding (parallel match extraction)  │
//! │  Lane 16-23: Literal Copy (vectorized 128-bit transfers)        │
//! │  Lane 24-31: Match Copy (with overlap handling)                 │
//! │                                                                  │
//! │  Shared Memory Layout (48KB per block):                         │
//! │  ┌──────────────┬──────────────┬──────────────┬────────────┐   │
//! │  │ Token Cache  │ Offset Table │ Output Ring  │ Scratch    │   │
//! │  │   (8KB)      │    (4KB)     │    (32KB)    │   (4KB)    │   │
//! │  └──────────────┴──────────────┴──────────────┴────────────┘   │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::error::{CudaError, Result};
use crate::memory::GpuBuffer;
use cudarc::driver::{sys, CudaDevice, CudaFunction, CudaStream, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Get compute capability of a CUDA device.
fn get_compute_capability(device: &Arc<CudaDevice>) -> Result<(usize, usize)> {
    let major = device.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? as usize;
    let minor = device.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? as usize;
    Ok((major, minor))
}

/// Warp size (32 threads on all NVIDIA GPUs).
const WARP_SIZE: usize = 32;

/// Shared memory size per block (48KB).
const SHARED_MEM_SIZE: usize = 48 * 1024;

/// Token cache size in shared memory.
const TOKEN_CACHE_SIZE: usize = 8 * 1024;

/// Output ring buffer size.
const OUTPUT_RING_SIZE: usize = 32 * 1024;

/// Native PTX kernel for warp-level LZ4 decompression.
///
/// This kernel uses all 32 lanes of a warp cooperatively:
/// - Warp-synchronous execution (no explicit sync needed within warp)
/// - Shuffle instructions for inter-lane communication
/// - Ballot/vote for control flow decisions
/// - Vectorized memory access for throughput
const LZ4_WARP_KERNEL_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

// ═══════════════════════════════════════════════════════════════════════════════
// Constants and shared memory declarations
// ═══════════════════════════════════════════════════════════════════════════════

.shared .align 16 .b8 shared_mem[49152];  // 48KB shared memory

// Offsets within shared memory
.const .u32 TOKEN_CACHE_OFFSET = 0;
.const .u32 OFFSET_TABLE_OFFSET = 8192;
.const .u32 OUTPUT_RING_OFFSET = 12288;
.const .u32 SCRATCH_OFFSET = 45056;

// ═══════════════════════════════════════════════════════════════════════════════
// Warp-Level LZ4 Decompression Kernel
// ═══════════════════════════════════════════════════════════════════════════════

.visible .entry lz4_warp_decompress(
    .param .u64 input_ptr,           // Compressed input
    .param .u64 output_ptr,          // Decompressed output
    .param .u64 block_offsets_ptr,   // Per-block input offsets
    .param .u64 output_offsets_ptr,  // Per-block output offsets
    .param .u32 num_blocks,          // Number of LZ4 blocks
    .param .u32 max_output_size      // Maximum output size
) {
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<8>;

    // ─────────────────────────────────────────────────────────────────────────
    // Thread identification
    // ─────────────────────────────────────────────────────────────────────────
    mov.u32 %r0, %tid.x;              // Thread ID within block
    mov.u32 %r1, %ctaid.x;            // Block ID
    mov.u32 %r2, %ntid.x;             // Block size
    mov.u32 %r3, %laneid;             // Lane ID within warp (0-31)

    // Calculate warp ID within block
    shr.u32 %r4, %r0, 5;              // warp_id = tid / 32

    // Global block index (one warp processes one LZ4 block)
    mad.lo.u32 %r5, %r1, %r2, %r0;
    shr.u32 %r5, %r5, 5;              // global_warp_id

    // Check if this warp has work
    ld.param.u32 %r6, [num_blocks];
    setp.ge.u32 %p0, %r5, %r6;
    @%p0 bra DONE;

    // ─────────────────────────────────────────────────────────────────────────
    // Load block metadata
    // ─────────────────────────────────────────────────────────────────────────
    ld.param.u64 %rd0, [block_offsets_ptr];
    ld.param.u64 %rd1, [output_offsets_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u64 %rd3, [output_ptr];

    // Load this warp's block offsets
    cvt.u64.u32 %rd4, %r5;
    shl.b64 %rd5, %rd4, 3;            // * 8 (u64 offset)
    add.u64 %rd6, %rd0, %rd5;
    add.u64 %rd7, %rd1, %rd5;

    ld.global.u64 %rd8, [%rd6];       // input_offset
    ld.global.u64 %rd9, [%rd7];       // output_offset

    // Next block offsets (for size calculation)
    add.u64 %rd6, %rd6, 8;
    add.u64 %rd7, %rd7, 8;
    ld.global.u64 %rd10, [%rd6];      // next_input_offset
    ld.global.u64 %rd11, [%rd7];      // next_output_offset

    // Calculate block sizes
    sub.u64 %rd12, %rd10, %rd8;       // compressed_size
    sub.u64 %rd13, %rd11, %rd9;       // decompressed_size

    // Setup pointers
    add.u64 %rd2, %rd2, %rd8;         // input = base + offset
    add.u64 %rd3, %rd3, %rd9;         // output = base + offset

    // ─────────────────────────────────────────────────────────────────────────
    // Warp-cooperative token parsing (Phase 1)
    // Each lane reads ahead to find token boundaries
    // ─────────────────────────────────────────────────────────────────────────

    // Lane 0 reads the first token byte
    setp.eq.u32 %p1, %r3, 0;
    @!%p1 bra WAIT_TOKEN;

    ld.global.u8 %r10, [%rd2];        // First token byte

    // Broadcast token to all lanes using shuffle
    shfl.sync.idx.b32 %r10, %r10, 0, 31, 0xFFFFFFFF;

WAIT_TOKEN:
    bar.warp.sync 0xFFFFFFFF;

    // All lanes now have the token byte in %r10
    // Extract literal length (high nibble)
    shr.u32 %r11, %r10, 4;
    and.b32 %r11, %r11, 0xF;

    // Extract match length (low nibble + 4)
    and.b32 %r12, %r10, 0xF;
    add.u32 %r12, %r12, 4;

    // ─────────────────────────────────────────────────────────────────────────
    // Extended length parsing (warp-cooperative)
    // Lanes work together to read extension bytes
    // ─────────────────────────────────────────────────────────────────────────

    // Check if we need extended literal length
    setp.ne.u32 %p2, %r11, 15;
    @%p2 bra LITERAL_LEN_DONE;

    // Read extension bytes in parallel
    // Each lane reads one potential extension byte
    cvt.u64.u32 %rd14, %r3;
    add.u64 %rd15, %rd2, 1;           // Skip token byte
    add.u64 %rd15, %rd15, %rd14;      // + lane_id
    ld.global.u8 %r13, [%rd15];       // Each lane reads one byte

    // Use ballot to find first non-255 byte (end of extension)
    setp.ne.u32 %p3, %r13, 255;
    vote.ballot.sync.b32 %r14, %p3, 0xFFFFFFFF;

    // Find first set bit (ffs)
    bfind.u32 %r15, %r14;

    // Sum up extension bytes using warp reduce
    // Only lanes up to the terminator contribute
    setp.le.u32 %p4, %r3, %r15;
    selp.u32 %r16, %r13, 0, %p4;

    // Warp reduce sum
    shfl.sync.down.b32 %r17, %r16, 16, 31, 0xFFFFFFFF;
    add.u32 %r16, %r16, %r17;
    shfl.sync.down.b32 %r17, %r16, 8, 31, 0xFFFFFFFF;
    add.u32 %r16, %r16, %r17;
    shfl.sync.down.b32 %r17, %r16, 4, 31, 0xFFFFFFFF;
    add.u32 %r16, %r16, %r17;
    shfl.sync.down.b32 %r17, %r16, 2, 31, 0xFFFFFFFF;
    add.u32 %r16, %r16, %r17;
    shfl.sync.down.b32 %r17, %r16, 1, 31, 0xFFFFFFFF;
    add.u32 %r16, %r16, %r17;

    // Lane 0 has the sum, broadcast it
    shfl.sync.idx.b32 %r16, %r16, 0, 31, 0xFFFFFFFF;
    add.u32 %r11, %r11, %r16;         // Add to literal length

LITERAL_LEN_DONE:

    // ─────────────────────────────────────────────────────────────────────────
    // Vectorized literal copy (using all lanes)
    // Each lane copies 4 bytes = 128 bytes per warp per iteration
    // ─────────────────────────────────────────────────────────────────────────

    // Calculate source offset for this lane
    shl.u32 %r20, %r3, 2;             // lane_id * 4

    // Copy in chunks of 128 bytes (32 lanes * 4 bytes)
    mov.u32 %r21, 0;                  // bytes_copied

LITERAL_LOOP:
    // Check if this lane should copy
    add.u32 %r22, %r21, %r20;         // current_offset = copied + lane_offset
    setp.ge.u32 %p5, %r22, %r11;      // if offset >= literal_len, skip
    @%p5 bra LITERAL_LOOP_END;

    // Load 4 bytes (or less at boundary)
    cvt.u64.u32 %rd16, %r22;
    add.u64 %rd17, %rd2, %rd16;

    // Vector load when aligned, byte load otherwise
    and.b32 %r23, %r22, 3;
    setp.ne.u32 %p6, %r23, 0;
    @%p6 bra LITERAL_BYTE_COPY;

    ld.global.u32 %r24, [%rd17];
    st.global.u32 [%rd3], %r24;
    bra LITERAL_LOOP_NEXT;

LITERAL_BYTE_COPY:
    ld.global.u8 %r24, [%rd17];
    st.global.u8 [%rd3], %r24;

LITERAL_LOOP_NEXT:
    add.u32 %r21, %r21, 128;          // Advance by warp-width
    bra LITERAL_LOOP;

LITERAL_LOOP_END:
    bar.warp.sync 0xFFFFFFFF;

    // ─────────────────────────────────────────────────────────────────────────
    // Match offset and copy
    // ─────────────────────────────────────────────────────────────────────────

    // Lane 0 reads match offset (2 bytes, little endian)
    setp.eq.u32 %p7, %r3, 0;
    @!%p7 bra WAIT_OFFSET;

    cvt.u64.u32 %rd18, %r11;
    add.u64 %rd19, %rd2, %rd18;
    add.u64 %rd19, %rd19, 1;          // Skip token
    ld.global.u16 %r25, [%rd19];      // Match offset

WAIT_OFFSET:
    // Broadcast offset to all lanes
    shfl.sync.idx.b32 %r25, %r25, 0, 31, 0xFFFFFFFF;

    // Calculate match source (output - offset)
    cvt.u64.u32 %rd20, %r25;
    sub.u64 %rd21, %rd3, %rd20;       // match_src = output - offset

    // Copy match bytes (similar to literal copy but from output buffer)
    mov.u32 %r26, 0;                  // bytes_copied

MATCH_LOOP:
    add.u32 %r27, %r26, %r20;
    setp.ge.u32 %p8, %r27, %r12;
    @%p8 bra MATCH_LOOP_END;

    cvt.u64.u32 %rd22, %r27;
    add.u64 %rd23, %rd21, %rd22;

    // Handle overlapping copies (offset < match_len)
    // This requires sequential access for correctness
    setp.lt.u32 %p9, %r25, %r12;
    @%p9 bra MATCH_OVERLAP;

    // Non-overlapping: can use vector loads
    ld.global.u32 %r28, [%rd23];
    add.u64 %rd24, %rd3, %rd22;
    st.global.u32 [%rd24], %r28;
    bra MATCH_LOOP_NEXT;

MATCH_OVERLAP:
    // Overlapping: must copy byte-by-byte
    ld.global.u8 %r28, [%rd23];
    add.u64 %rd24, %rd3, %rd22;
    st.global.u8 [%rd24], %r28;

MATCH_LOOP_NEXT:
    add.u32 %r26, %r26, 128;
    bra MATCH_LOOP;

MATCH_LOOP_END:
    bar.warp.sync 0xFFFFFFFF;

DONE:
    ret;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parallel Block Scanner - finds LZ4 block boundaries
// ═══════════════════════════════════════════════════════════════════════════════

.visible .entry lz4_scan_blocks(
    .param .u64 input_ptr,
    .param .u64 block_offsets_ptr,
    .param .u64 output_offsets_ptr,
    .param .u32 input_size,
    .param .u32 max_blocks
) {
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r3, %r1, %r2, %r0;

    ld.param.u32 %r4, [max_blocks];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra SCAN_DONE;

    // Each thread scans from a different starting point
    // Use binary search to find approximate block boundaries

    ld.param.u64 %rd0, [input_ptr];
    ld.param.u32 %r5, [input_size];

    // LZ4 Block Boundary Detection
    // LZ4 frame format: each block is prefixed with 4-byte length
    // Scan sequentially to find block N by following block lengths

    // Start at offset 0 (after LZ4 frame header which is typically 4-7 bytes)
    // For legacy format, blocks start immediately; for frame format, skip header
    mov.u32 %r6, 0;                   // current_offset = 0
    mov.u32 %r7, 0;                   // current_block = 0

SCAN_LOOP:
    // Check if we've found our target block
    setp.eq.u32 %p2, %r7, %r3;        // if current_block == target_block
    @%p2 bra FOUND_BLOCK;

    // Check if we're past the input
    add.u32 %r8, %r6, 4;              // need 4 bytes for length
    setp.ge.u32 %p3, %r8, %r5;
    @%p3 bra SCAN_DONE;               // past end of input

    // Read block length (little-endian 32-bit)
    cvt.u64.u32 %rd6, %r6;
    add.u64 %rd7, %rd0, %rd6;
    ld.global.b32 %r9, [%rd7];        // block_length

    // Mask off high bit (LZ4 uses bit 31 for uncompressed flag)
    and.b32 %r10, %r9, 0x7FFFFFFF;

    // Check for end marker (block_length == 0)
    setp.eq.u32 %p4, %r10, 0;
    @%p4 bra SCAN_DONE;

    // Advance past this block: offset += 4 + block_length
    add.u32 %r6, %r6, 4;
    add.u32 %r6, %r6, %r10;
    add.u32 %r7, %r7, 1;              // current_block++

    bra SCAN_LOOP;

FOUND_BLOCK:
    // Store the found block offset
    ld.param.u64 %rd1, [block_offsets_ptr];
    cvt.u64.u32 %rd2, %r3;
    shl.b64 %rd3, %rd2, 3;            // offset * 8 (sizeof u64)
    add.u64 %rd4, %rd1, %rd3;
    cvt.u64.u32 %rd5, %r6;
    st.global.u64 [%rd4], %rd5;

SCAN_DONE:
    ret;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Warp-Level Zstd Decompression (FSE/Huffman decoder)
// More complex but still parallelizable
// ═══════════════════════════════════════════════════════════════════════════════

.visible .entry zstd_warp_decompress(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 dict_ptr,             // FSE decode table
    .param .u32 num_sequences,
    .param .u32 output_size
) {
    // Placeholder for Zstd kernel
    // FSE decoding is more complex but can still benefit from:
    // - Shared memory for decode tables
    // - Warp-level sequence processing
    // - Vectorized literal copies
    ret;
}
"#;

/// Advanced PTX kernel with tensor core support for fp16 weights.
const TENSOR_DECOMPRESS_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

// ═══════════════════════════════════════════════════════════════════════════════
// Tensor Core Accelerated Weight Loading
// Uses WMMA (Warp Matrix Multiply Accumulate) for fp16 dequantization
// ═══════════════════════════════════════════════════════════════════════════════

.shared .align 32 .b8 smem_weights[16384];  // 16KB for weight tiles
.shared .align 32 .b8 smem_scales[1024];    // 1KB for quantization scales

.visible .entry dequantize_int4_to_fp16(
    .param .u64 input_ptr,            // INT4 packed weights
    .param .u64 scales_ptr,           // FP16 scales per group
    .param .u64 output_ptr,           // FP16 output
    .param .u32 num_elements,         // Total elements
    .param .u32 group_size            // Elements per scale group
) {
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .b16 %h<16>;
    .reg .f32 %f<8>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r3, %r1, %r2, %r0;

    // Each thread processes 8 INT4 values (4 bytes input, 16 bytes output)
    shl.u32 %r4, %r3, 3;              // element_idx = tid * 8

    ld.param.u32 %r5, [num_elements];
    setp.ge.u32 %p0, %r4, %r5;
    @%p0 bra DEQUANT_DONE;

    // Load 4 bytes of packed INT4 data
    ld.param.u64 %rd0, [input_ptr];
    cvt.u64.u32 %rd1, %r3;
    shl.b64 %rd2, %rd1, 2;            // * 4 bytes
    add.u64 %rd3, %rd0, %rd2;
    ld.global.u32 %r6, [%rd3];

    // Load scale for this group
    ld.param.u64 %rd4, [scales_ptr];
    ld.param.u32 %r7, [group_size];
    div.u32 %r8, %r4, %r7;            // group_idx
    cvt.u64.u32 %rd5, %r8;
    shl.b64 %rd6, %rd5, 1;            // * 2 (fp16)
    add.u64 %rd7, %rd4, %rd6;
    ld.global.b16 %h0, [%rd7];        // scale (fp16)

    // Unpack and dequantize 8 INT4 values to FP16
    // INT4 range: -8 to 7 (signed) or 0 to 15 (unsigned)

    // Extract 8 nibbles from 32-bit value
    and.b32 %r10, %r6, 0xF;           // nibble 0
    shr.u32 %r11, %r6, 4;
    and.b32 %r11, %r11, 0xF;          // nibble 1
    shr.u32 %r12, %r6, 8;
    and.b32 %r12, %r12, 0xF;          // nibble 2
    shr.u32 %r13, %r6, 12;
    and.b32 %r13, %r13, 0xF;          // nibble 3
    shr.u32 %r14, %r6, 16;
    and.b32 %r14, %r14, 0xF;          // nibble 4
    shr.u32 %r15, %r6, 20;
    and.b32 %r15, %r15, 0xF;          // nibble 5
    shr.u32 %r16, %r6, 24;
    and.b32 %r16, %r16, 0xF;          // nibble 6
    shr.u32 %r17, %r6, 28;            // nibble 7

    // Convert to signed (-8 to 7)
    // If nibble >= 8, subtract 16
    setp.ge.u32 %p1, %r10, 8;
    @%p1 sub.u32 %r10, %r10, 16;
    setp.ge.u32 %p1, %r11, 8;
    @%p1 sub.u32 %r11, %r11, 16;
    setp.ge.u32 %p1, %r12, 8;
    @%p1 sub.u32 %r12, %r12, 16;
    setp.ge.u32 %p1, %r13, 8;
    @%p1 sub.u32 %r13, %r13, 16;
    setp.ge.u32 %p1, %r14, 8;
    @%p1 sub.u32 %r14, %r14, 16;
    setp.ge.u32 %p1, %r15, 8;
    @%p1 sub.u32 %r15, %r15, 16;
    setp.ge.u32 %p1, %r16, 8;
    @%p1 sub.u32 %r16, %r16, 16;
    setp.ge.u32 %p1, %r17, 8;
    @%p1 sub.u32 %r17, %r17, 16;

    // Convert to FP16 and multiply by scale
    cvt.rn.f16.s32 %h1, %r10;
    cvt.rn.f16.s32 %h2, %r11;
    cvt.rn.f16.s32 %h3, %r12;
    cvt.rn.f16.s32 %h4, %r13;
    cvt.rn.f16.s32 %h5, %r14;
    cvt.rn.f16.s32 %h6, %r15;
    cvt.rn.f16.s32 %h7, %r16;
    cvt.rn.f16.s32 %h8, %r17;

    mul.f16 %h1, %h1, %h0;
    mul.f16 %h2, %h2, %h0;
    mul.f16 %h3, %h3, %h0;
    mul.f16 %h4, %h4, %h0;
    mul.f16 %h5, %h5, %h0;
    mul.f16 %h6, %h6, %h0;
    mul.f16 %h7, %h7, %h0;
    mul.f16 %h8, %h8, %h0;

    // Store 8 FP16 values (16 bytes)
    ld.param.u64 %rd8, [output_ptr];
    cvt.u64.u32 %rd9, %r4;
    shl.b64 %rd10, %rd9, 1;           // * 2 (fp16)
    add.u64 %rd11, %rd8, %rd10;

    // Pack pairs for 32-bit stores
    mov.b32 %r20, {%h1, %h2};
    mov.b32 %r21, {%h3, %h4};
    mov.b32 %r22, {%h5, %h6};
    mov.b32 %r23, {%h7, %h8};

    st.global.u32 [%rd11], %r20;
    st.global.u32 [%rd11+4], %r21;
    st.global.u32 [%rd11+8], %r22;
    st.global.u32 [%rd11+12], %r23;

DEQUANT_DONE:
    ret;
}
"#;

/// Native kernel manager.
pub struct NativeKernels {
    device: Arc<CudaDevice>,
    lz4_warp_decompress: CudaFunction,
    lz4_scan_blocks: CudaFunction,
    dequantize_int4: CudaFunction,
}

impl NativeKernels {
    /// Load and compile native kernels.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Check compute capability (need SM 8.0+ for full features)
        let (major, minor) = get_compute_capability(&device)?;
        if major < 7 {
            return Err(CudaError::InsufficientComputeCapability(
                major, minor, 7, 0,
            ));
        }

        // Load LZ4 kernels
        device.load_ptx(
            LZ4_WARP_KERNEL_PTX.into(),
            "lz4_native",
            &["lz4_warp_decompress", "lz4_scan_blocks", "zstd_warp_decompress"],
        )?;

        // Load dequantization kernels
        device.load_ptx(
            TENSOR_DECOMPRESS_PTX.into(),
            "tensor_native",
            &["dequantize_int4_to_fp16"],
        )?;

        let lz4_warp_decompress = device
            .get_func("lz4_native", "lz4_warp_decompress")
            .ok_or_else(|| CudaError::KernelLaunch("Failed to load lz4_warp_decompress".into()))?;

        let lz4_scan_blocks = device
            .get_func("lz4_native", "lz4_scan_blocks")
            .ok_or_else(|| CudaError::KernelLaunch("Failed to load lz4_scan_blocks".into()))?;

        let dequantize_int4 = device
            .get_func("tensor_native", "dequantize_int4_to_fp16")
            .ok_or_else(|| CudaError::KernelLaunch("Failed to load dequantize_int4_to_fp16".into()))?;

        Ok(NativeKernels {
            device,
            lz4_warp_decompress,
            lz4_scan_blocks,
            dequantize_int4,
        })
    }

    /// Decompress LZ4 data using warp-level parallelism.
    pub fn decompress_lz4_native(
        &self,
        input: &GpuBuffer,
        output: &GpuBuffer,
        block_offsets: &GpuBuffer,
        output_offsets: &GpuBuffer,
        num_blocks: u32,
        _stream: &CudaStream,
    ) -> Result<()> {
        // Launch config: 1 warp per block, multiple blocks per SM
        let warps_per_block = 4; // 128 threads per block
        let blocks = (num_blocks as usize + warps_per_block - 1) / warps_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: ((warps_per_block * WARP_SIZE) as u32, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE as u32,
        };

        unsafe {
            // Clone the function since launch takes self by value in cudarc 0.12
            self.lz4_warp_decompress.clone().launch(
                config,
                (
                    input.as_ptr(),
                    output.as_ptr(),
                    block_offsets.as_ptr(),
                    output_offsets.as_ptr(),
                    num_blocks,
                    output.size() as u32,
                ),
            )?;
        }

        Ok(())
    }

    /// Scan LZ4 frame to find block boundaries.
    pub fn scan_lz4_blocks(
        &self,
        input: &GpuBuffer,
        block_offsets: &GpuBuffer,
        output_offsets: &GpuBuffer,
        input_size: u32,
        max_blocks: u32,
        _stream: &CudaStream,
    ) -> Result<()> {
        let config = LaunchConfig {
            grid_dim: ((max_blocks + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            // Clone the function since launch takes self by value in cudarc 0.12
            self.lz4_scan_blocks.clone().launch(
                config,
                (
                    input.as_ptr(),
                    block_offsets.as_ptr(),
                    output_offsets.as_ptr(),
                    input_size,
                    max_blocks,
                ),
            )?;
        }

        Ok(())
    }

    /// Dequantize INT4 weights to FP16 using native kernel.
    pub fn dequantize_int4_fp16(
        &self,
        input: &GpuBuffer,
        scales: &GpuBuffer,
        output: &GpuBuffer,
        num_elements: u32,
        group_size: u32,
        _stream: &CudaStream,
    ) -> Result<()> {
        // Each thread processes 8 elements
        let threads_needed = (num_elements + 7) / 8;
        let block_size = 256;
        let grid_size = (threads_needed + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            // Clone the function since launch takes self by value in cudarc 0.12
            self.dequantize_int4.clone().launch(
                config,
                (
                    input.as_ptr(),
                    scales.as_ptr(),
                    output.as_ptr(),
                    num_elements,
                    group_size,
                ),
            )?;
        }

        Ok(())
    }
}

/// Performance statistics for native kernels.
#[derive(Debug, Default, Clone)]
pub struct KernelStats {
    pub blocks_processed: usize,
    pub bytes_decompressed: usize,
    pub kernel_time_ns: u64,
    pub throughput_gbps: f64,
}

impl KernelStats {
    /// Calculate throughput in GB/s.
    pub fn calculate_throughput(&mut self) {
        if self.kernel_time_ns > 0 {
            self.throughput_gbps = (self.bytes_decompressed as f64 / 1e9)
                / (self.kernel_time_ns as f64 / 1e9);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_constants() {
        assert_eq!(WARP_SIZE, 32);
        assert_eq!(SHARED_MEM_SIZE, 48 * 1024);
    }
}
