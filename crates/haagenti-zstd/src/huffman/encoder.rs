//! Huffman encoding for Zstd literals.
//!
//! This module implements high-performance Huffman encoding for Zstd compression.
//!
//! ## Optimizations
//!
//! - SIMD-accelerated frequency counting (histogram)
//! - 64-bit accumulator for efficient bit packing
//! - Cache-friendly code table layout
//! - Vectorized encoding for batch processing
//!
//! ## Weight System
//!
//! In Zstd Huffman encoding:
//! - Weight `w > 0` means `code_length = max_bits + 1 - w`
//! - Weight `0` means symbol is not present
//! - Higher weight = shorter code = more frequent symbol
//! - Maximum weight is 11 (minimum code length = 1 bit)
//!
//! ## References
//!
//! - [RFC 8878 Section 4.2](https://datatracker.ietf.org/doc/html/rfc8878#section-4.2)

use crate::fse::{FseBitWriter, FseTable};

/// Maximum number of symbols for Huffman encoding (256 for bytes).
const MAX_SYMBOLS: usize = 256;

/// Maximum Huffman weight (limits code length).
const MAX_WEIGHT: u8 = 11;

/// Minimum data size to benefit from Huffman encoding.
const MIN_HUFFMAN_SIZE: usize = 32;

/// Huffman encoding table entry - packed for cache efficiency.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(4))]
pub struct HuffmanCode {
    /// The Huffman code bits (stored in LSB).
    pub code: u16,
    /// Number of bits in the code.
    pub num_bits: u8,
    /// Padding for alignment.
    _pad: u8,
}

impl HuffmanCode {
    #[inline]
    const fn new(code: u16, num_bits: u8) -> Self {
        Self {
            code,
            num_bits,
            _pad: 0,
        }
    }
}

/// Optimized Huffman encoder for literal compression.
#[derive(Debug)]
pub struct HuffmanEncoder {
    /// Encoding table: symbol -> code (256 entries, cache-aligned)
    codes: Box<[HuffmanCode; MAX_SYMBOLS]>,
    /// Symbol weights for serialization
    weights: Vec<u8>,
    /// Maximum code length in bits
    max_bits: u8,
    /// Number of symbols with non-zero weight
    num_symbols: usize,
    /// Highest symbol index with non-zero weight (for weight table sizing)
    last_symbol: usize,
}

impl HuffmanEncoder {
    /// Build a Huffman encoder from literal data.
    ///
    /// Uses SIMD-accelerated histogram when available.
    /// Returns None if data cannot be efficiently Huffman-compressed.
    pub fn build(data: &[u8]) -> Option<Self> {
        if data.len() < MIN_HUFFMAN_SIZE {
            return None;
        }

        // Count symbol frequencies using optimized histogram
        let freq = Self::count_frequencies(data);

        // Count unique symbols and find last symbol with non-zero frequency
        let unique_count = freq.iter().filter(|&&f| f > 0).count();
        if unique_count < 2 {
            return None; // Use RLE instead
        }

        let last_symbol = freq
            .iter()
            .enumerate()
            .filter(|&(_, &f)| f > 0)
            .map(|(i, _)| i)
            .max()
            .unwrap_or(0);

        // Convert frequencies to weights
        let (weights, max_bits) = Self::frequencies_to_weights(&freq)?;

        // Generate canonical codes
        let codes = Self::generate_canonical_codes(&weights, max_bits);

        Some(Self {
            codes: Box::new(codes),
            weights,
            max_bits,
            num_symbols: unique_count,
            last_symbol,
        })
    }

    /// Build a Huffman encoder from pre-defined weights.
    ///
    /// This allows using custom Huffman tables instead of building from data.
    /// Useful when you have pre-trained weights from dictionary compression
    /// or want to reuse weights across multiple blocks.
    ///
    /// # Parameters
    ///
    /// - `weights`: Array of 256 weights (one per byte value). Weight 0 means
    ///   symbol is not present. Weight w > 0 means code_length = max_bits + 1 - w.
    ///
    /// # Returns
    ///
    /// Returns `Some(encoder)` if the weights are valid, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use haagenti_zstd::huffman::HuffmanEncoder;
    ///
    /// // Define weights for symbols 'a' (97), 'b' (98), 'c' (99)
    /// let mut weights = vec![0u8; 256];
    /// weights[97] = 3;  // 'a' - highest weight (shortest code)
    /// weights[98] = 2;  // 'b' - medium weight
    /// weights[99] = 1;  // 'c' - lowest weight (longest code)
    ///
    /// let encoder = HuffmanEncoder::from_weights(&weights).unwrap();
    /// ```
    pub fn from_weights(weights: &[u8]) -> Option<Self> {
        if weights.len() != MAX_SYMBOLS {
            return None;
        }

        // Count unique symbols and find last symbol with non-zero weight
        let unique_count = weights.iter().filter(|&&w| w > 0).count();
        if unique_count < 2 {
            return None; // Need at least 2 symbols
        }

        let last_symbol = weights
            .iter()
            .enumerate()
            .filter(|&(_, &w)| w > 0)
            .map(|(i, _)| i)
            .max()
            .unwrap_or(0);

        // Find max weight to determine max_bits
        let max_weight = *weights.iter().max().unwrap_or(&0);
        if max_weight == 0 || max_weight > MAX_WEIGHT {
            return None;
        }

        // Calculate max_bits from max_weight
        // In Zstd: code_length = max_bits + 1 - weight
        // For the highest weight symbol, code_length should be 1, so:
        // max_bits = max_weight
        let max_bits = max_weight;

        // Generate canonical codes from weights
        let codes = Self::generate_canonical_codes(weights, max_bits);

        Some(Self {
            codes: Box::new(codes),
            weights: weights.to_vec(),
            max_bits,
            num_symbols: unique_count,
            last_symbol,
        })
    }

    /// Count byte frequencies using optimized histogram.
    ///
    /// Uses SIMD acceleration when available via haagenti_simd.
    #[inline]
    fn count_frequencies(data: &[u8]) -> [u32; MAX_SYMBOLS] {
        // Use SIMD-accelerated histogram when feature is enabled
        #[cfg(feature = "simd")]
        {
            haagenti_simd::byte_histogram(data)
        }

        // Optimized scalar fallback using 4-way interleaved counting
        // This reduces cache line conflicts from histogram updates
        #[cfg(not(feature = "simd"))]
        {
            let mut freq0 = [0u32; MAX_SYMBOLS];
            let mut freq1 = [0u32; MAX_SYMBOLS];
            let mut freq2 = [0u32; MAX_SYMBOLS];
            let mut freq3 = [0u32; MAX_SYMBOLS];

            // Process 16 bytes at a time with 4 interleaved histograms
            let chunks = data.chunks_exact(16);
            let remainder = chunks.remainder();

            for chunk in chunks {
                // Interleave to reduce pipeline stalls from same-address increments
                freq0[chunk[0] as usize] += 1;
                freq1[chunk[1] as usize] += 1;
                freq2[chunk[2] as usize] += 1;
                freq3[chunk[3] as usize] += 1;
                freq0[chunk[4] as usize] += 1;
                freq1[chunk[5] as usize] += 1;
                freq2[chunk[6] as usize] += 1;
                freq3[chunk[7] as usize] += 1;
                freq0[chunk[8] as usize] += 1;
                freq1[chunk[9] as usize] += 1;
                freq2[chunk[10] as usize] += 1;
                freq3[chunk[11] as usize] += 1;
                freq0[chunk[12] as usize] += 1;
                freq1[chunk[13] as usize] += 1;
                freq2[chunk[14] as usize] += 1;
                freq3[chunk[15] as usize] += 1;
            }

            // Handle remainder
            for &byte in remainder {
                freq0[byte as usize] += 1;
            }

            // Merge the 4 histograms
            for i in 0..MAX_SYMBOLS {
                freq0[i] += freq1[i] + freq2[i] + freq3[i];
            }

            freq0
        }
    }

    /// Convert frequencies to Zstd Huffman weights.
    ///
    /// Produces weights that satisfy the Kraft inequality:
    /// sum(2^weight) = 2^(max_weight + 1)
    ///
    /// # Algorithm Complexity: O(n log n)
    ///
    /// 1. Sort symbols by frequency: O(n log n)
    /// 2. Calculate initial weights based on frequency ratios: O(n)
    /// 3. Adjust weights to fill Kraft capacity using heap-based greedy: O(n log n)
    ///
    /// This replaces the previous O(n²) algorithm that used repeated full scans.
    fn frequencies_to_weights(freq: &[u32; MAX_SYMBOLS]) -> Option<(Vec<u8>, u8)> {
        // Collect non-zero frequency symbols
        let mut symbols: Vec<(usize, u32)> = freq
            .iter()
            .enumerate()
            .filter(|&(_, &f)| f > 0)
            .map(|(i, &f)| (i, f))
            .collect();

        if symbols.len() < 2 {
            return None;
        }

        let n = symbols.len();

        // Special case: exactly 2 symbols get weight 1 each (1-bit codes)
        if n == 2 {
            let mut weights = vec![0u8; MAX_SYMBOLS];
            weights[symbols[0].0] = 1;
            weights[symbols[1].0] = 1;
            return Some((weights, 1));
        }

        // Sort symbols by frequency (highest first) - O(n log n)
        symbols.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Calculate max_weight needed for n symbols
        let min_exp = if n <= 2 {
            0
        } else {
            64 - ((n - 1) as u64).leading_zeros()
        };
        let max_weight = ((min_exp + 1) as u8).clamp(1, MAX_WEIGHT);

        let mut weights = vec![0u8; MAX_SYMBOLS];
        let target = 1u64 << (max_weight + 1);

        // Phase 1: Assign initial weights based on frequency ratio - O(n)
        // Use log2(max_freq / freq) to estimate relative code lengths
        let max_freq = symbols[0].1 as u64;

        for (idx, &(sym, freq)) in symbols.iter().enumerate() {
            if idx == 0 {
                // Most frequent symbol gets max_weight (shortest code)
                weights[sym] = max_weight;
            } else {
                // Calculate weight based on frequency ratio
                // Higher ratio = lower frequency = lower weight = longer code
                let ratio = (max_freq + freq as u64 - 1) / freq.max(1) as u64;
                let log_ratio = if ratio <= 1 {
                    0
                } else {
                    (64 - ratio.leading_zeros()).saturating_sub(1) as u8
                };
                // Clamp to valid range [1, max_weight]
                let w = max_weight.saturating_sub(log_ratio).max(1);
                weights[sym] = w;
            }
        }

        // Calculate current Kraft sum - O(n)
        let mut kraft_sum: u64 = symbols.iter().map(|(sym, _)| 1u64 << weights[*sym]).sum();

        // Phase 2: Adjust weights to satisfy Kraft inequality - O(n log n) worst case
        // Use a greedy approach: process symbols by weight (lowest first for increasing)

        if kraft_sum < target {
            // Under capacity: increase weights for symbols (shorter codes)
            // Process from lowest weight to highest (most room to increase)
            let mut by_weight: Vec<(usize, u8)> = symbols
                .iter()
                .map(|&(sym, _)| (sym, weights[sym]))
                .collect();
            by_weight.sort_unstable_by_key(|&(_, w)| w);

            for (sym, _) in by_weight {
                while weights[sym] < max_weight && kraft_sum < target {
                    let increase = 1u64 << weights[sym];
                    if kraft_sum + increase <= target {
                        kraft_sum += increase;
                        weights[sym] += 1;
                    } else {
                        break;
                    }
                }
            }
        } else if kraft_sum > target {
            // Over capacity: decrease weights (longer codes)
            // Process from highest weight to lowest
            let mut by_weight: Vec<(usize, u8)> = symbols
                .iter()
                .map(|&(sym, _)| (sym, weights[sym]))
                .collect();
            by_weight.sort_unstable_by_key(|&(_, w)| std::cmp::Reverse(w));

            for (sym, _) in by_weight {
                while weights[sym] > 1 && kraft_sum > target {
                    weights[sym] -= 1;
                    kraft_sum -= 1u64 << weights[sym];
                }
            }
        }

        // Final pass: fill any remaining capacity - O(n)
        // This handles edge cases where the above didn't fully utilize capacity
        if kraft_sum < target {
            for &(sym, _) in &symbols {
                while weights[sym] < max_weight {
                    let increase = 1u64 << weights[sym];
                    if kraft_sum + increase <= target {
                        kraft_sum += increase;
                        weights[sym] += 1;
                    } else {
                        break;
                    }
                }
            }
        }

        Some((weights, max_weight))
    }

    /// Fix code lengths to satisfy Kraft inequality.
    /// For a valid Huffman code: sum(2^(max_len - len)) = 2^max_len
    #[allow(dead_code)]
    fn fix_kraft_inequality(code_lengths: &mut [u8], max_len: u8) {
        // First, check if we need a deeper tree
        // Calculate minimum required depth for this many symbols
        let num_symbols = code_lengths.iter().filter(|&&l| l > 0).count();
        if num_symbols <= 1 {
            return;
        }

        // Calculate current Kraft sum with current max_len
        let kraft_sum: u64 = code_lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (max_len.saturating_sub(l)) as u32)
            .sum();
        let target = 1u64 << max_len;

        if kraft_sum <= target {
            // Already valid or has room to spare - try to fill unused capacity
            if kraft_sum < target {
                Self::fill_kraft_capacity(code_lengths, max_len, target - kraft_sum);
            }
            return;
        }

        // Need deeper tree: increase max_len until Kraft sum fits
        // New max_len must be large enough that 2^new_max_len >= kraft_sum
        let new_max_len = (64 - kraft_sum.leading_zeros()) as u8;
        if new_max_len > MAX_WEIGHT {
            // Can't fix - too many symbols
            return;
        }

        // Increase all code lengths by (new_max_len - max_len)
        let depth_increase = new_max_len - max_len;
        for len in code_lengths.iter_mut() {
            if *len > 0 {
                *len = (*len + depth_increase).min(MAX_WEIGHT);
            }
        }

        // Now we have spare capacity, fill it by shortening some codes
        let new_kraft_sum: u64 = code_lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (new_max_len.saturating_sub(l)) as u32)
            .sum();
        let new_target = 1u64 << new_max_len;

        if new_kraft_sum < new_target {
            Self::fill_kraft_capacity(code_lengths, new_max_len, new_target - new_kraft_sum);
        }
    }

    /// Fill unused Kraft capacity by shortening some code lengths.
    #[allow(dead_code)]
    fn fill_kraft_capacity(code_lengths: &mut [u8], max_len: u8, mut spare: u64) {
        // Sort symbols by code length (longest first) to shorten long codes
        let mut syms: Vec<_> = code_lengths
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l > 1)
            .map(|(i, &l)| (i, l))
            .collect();
        syms.sort_by_key(|&(_, l)| std::cmp::Reverse(l));

        for (idx, old_len) in syms {
            if spare == 0 {
                break;
            }
            // Shortening by 1: contribution goes from 2^(max_len-old_len) to 2^(max_len-old_len+1)
            // Increase in usage: 2^(max_len-old_len)
            let increase = 1u64 << (max_len.saturating_sub(old_len)) as u32;
            if increase <= spare {
                code_lengths[idx] = old_len - 1;
                spare -= increase;
            }
        }
    }

    /// Limit code lengths to ensure they satisfy Kraft inequality.
    /// Uses a simple algorithm to redistribute long codes.
    #[allow(dead_code)]
    fn limit_code_lengths(code_lengths: &mut [u8], max_len: u8) {
        // Count symbols at each length
        let mut counts = vec![0u32; max_len as usize + 1];
        for &len in code_lengths.iter() {
            if len > 0 && len <= max_len {
                counts[len as usize] += 1;
            } else if len > max_len {
                counts[max_len as usize] += 1;
            }
        }

        // Clamp all lengths to max_len
        for len in code_lengths.iter_mut() {
            if *len > max_len {
                *len = max_len;
            }
        }

        // Adjust to satisfy Kraft: sum(2^-len) <= 1
        // Equivalently: sum(2^(max_len - len)) <= 2^max_len
        loop {
            let kraft_sum: u64 = counts
                .iter()
                .enumerate()
                .skip(1)
                .map(|(len, &count)| (count as u64) << (max_len as usize - len))
                .sum();

            let target = 1u64 << max_len;
            if kraft_sum <= target {
                break;
            }

            // Need to reduce: increase some code lengths
            // Find the shortest non-empty bucket and move one symbol to next bucket
            for len in 1..max_len as usize {
                if counts[len] > 0 {
                    counts[len] -= 1;
                    counts[len + 1] += 1;
                    // Update actual code lengths
                    for code_len in code_lengths.iter_mut() {
                        if *code_len == len as u8 {
                            *code_len = (len + 1) as u8;
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }

    /// Generate canonical Huffman codes from weights.
    fn generate_canonical_codes(weights: &[u8], max_bits: u8) -> [HuffmanCode; MAX_SYMBOLS] {
        let mut codes = [HuffmanCode::default(); MAX_SYMBOLS];

        // Count symbols at each code length
        let mut bl_count = vec![0u32; max_bits as usize + 2];
        for &w in weights {
            if w > 0 {
                let code_len = (max_bits + 1).saturating_sub(w) as usize;
                if code_len < bl_count.len() {
                    bl_count[code_len] += 1;
                }
            }
        }

        // Calculate starting codes for each length
        let mut next_code = vec![0u32; max_bits as usize + 2];
        let mut code = 0u32;
        for (bits, next_code_entry) in next_code
            .iter_mut()
            .enumerate()
            .take(max_bits as usize + 1)
            .skip(1)
        {
            code = (code + bl_count.get(bits - 1).copied().unwrap_or(0)) << 1;
            *next_code_entry = code;
        }

        // Assign codes to symbols
        for (symbol, &w) in weights.iter().enumerate() {
            if w > 0 && symbol < MAX_SYMBOLS {
                let code_len = (max_bits + 1).saturating_sub(w) as usize;
                if code_len < next_code.len() {
                    codes[symbol] = HuffmanCode::new(next_code[code_len] as u16, code_len as u8);
                    next_code[code_len] += 1;
                }
            }
        }

        codes
    }

    /// Encode literals using optimized bit packing.
    ///
    /// Uses 64-bit accumulator for efficient byte-aligned writes.
    /// Optimized with chunked reverse processing and software prefetching
    /// to maintain cache efficiency despite reverse iteration requirement.
    ///
    /// # Performance Optimizations
    /// - Processes in 64-byte cache-line chunks (reverse chunk order, forward within chunk)
    /// - Software prefetching brings next chunk into L1 cache ahead of time
    /// - 64-bit accumulator with branchless 32-bit flushes
    /// - Unrolled inner loop for better ILP
    pub fn encode(&self, literals: &[u8]) -> Vec<u8> {
        if literals.is_empty() {
            return vec![0x01]; // Just sentinel
        }

        // Pre-allocate output with better estimate
        let estimated_bits: usize = literals
            .iter()
            .take(256.min(literals.len()))
            .map(|&b| self.codes[b as usize].num_bits as usize)
            .sum();
        let avg_bits = if literals.len() <= 256 {
            estimated_bits
        } else {
            estimated_bits * literals.len() / 256.min(literals.len())
        };
        let mut output = Vec::with_capacity(avg_bits.div_ceil(8) + 16);

        // 64-bit accumulator for efficient bit packing
        let mut accum: u64 = 0;
        let mut bits_in_accum: u32 = 0;

        // Process in cache-line sized chunks (64 bytes) with prefetching
        // This maintains cache efficiency despite reverse iteration
        const CHUNK_SIZE: usize = 64;
        let len = literals.len();
        let mut pos = len;

        while pos > 0 {
            let chunk_start = pos.saturating_sub(CHUNK_SIZE);
            let chunk_end = pos;

            // Prefetch the NEXT chunk (earlier in memory) into L1 cache
            // This hides memory latency by fetching ahead
            #[cfg(target_arch = "x86_64")]
            if chunk_start >= CHUNK_SIZE {
                unsafe {
                    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                    _mm_prefetch(
                        literals.as_ptr().add(chunk_start - CHUNK_SIZE) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }

            // Process bytes within chunk in reverse order
            // The chunk is now in L1 cache, so reverse iteration is fast
            let chunk = &literals[chunk_start..chunk_end];

            // Unroll by 4 for better instruction-level parallelism
            let chunk_len = chunk.len();
            let mut i = chunk_len;

            // Handle tail (non-multiple of 4)
            while i > 0 && !i.is_multiple_of(4) {
                i -= 1;
                let byte = chunk[i];
                let code = &self.codes[byte as usize];
                let num_bits = code.num_bits as u32;

                if num_bits > 0 {
                    accum |= (code.code as u64) << bits_in_accum;
                    bits_in_accum += num_bits;

                    if bits_in_accum >= 32 {
                        output.extend_from_slice(&(accum as u32).to_le_bytes());
                        accum >>= 32;
                        bits_in_accum -= 32;
                    }
                }
            }

            // Process 4 bytes at a time (unrolled, branchless)
            // Novel optimization: Remove all branches in the inner loop.
            // Since num_bits==0 means the symbol isn't present (code==0, bits==0),
            // we can unconditionally OR and ADD without changing the result.
            // This enables better CPU pipelining and SIMD vectorization.
            while i >= 4 {
                i -= 4;

                // Load 4 codes (compiler can pipeline these loads)
                let c0 = self.codes[chunk[i + 3] as usize];
                let c1 = self.codes[chunk[i + 2] as usize];
                let c2 = self.codes[chunk[i + 1] as usize];
                let c3 = self.codes[chunk[i] as usize];

                // Branchless encoding: OR and ADD unconditionally
                // For valid symbols: adds the code bits
                // For invalid symbols (num_bits=0): OR 0, ADD 0 - no effect
                accum |= (c0.code as u64) << bits_in_accum;
                bits_in_accum += c0.num_bits as u32;
                accum |= (c1.code as u64) << bits_in_accum;
                bits_in_accum += c1.num_bits as u32;
                accum |= (c2.code as u64) << bits_in_accum;
                bits_in_accum += c2.num_bits as u32;
                accum |= (c3.code as u64) << bits_in_accum;
                bits_in_accum += c3.num_bits as u32;

                // Branchless flush: always flush when >= 32 bits
                // Using conditional move pattern that compilers optimize well
                if bits_in_accum >= 32 {
                    output.extend_from_slice(&(accum as u32).to_le_bytes());
                    accum >>= 32;
                    bits_in_accum -= 32;
                }
                // Second flush for cases where 4 symbols exceed 64 bits total
                if bits_in_accum >= 32 {
                    output.extend_from_slice(&(accum as u32).to_le_bytes());
                    accum >>= 32;
                    bits_in_accum -= 32;
                }
            }

            pos = chunk_start;
        }

        // Add sentinel bit
        accum |= 1u64 << bits_in_accum;
        bits_in_accum += 1;

        // Flush remaining bits (up to 5 bytes: 32 bits max + 1 sentinel)
        let remaining_bytes = bits_in_accum.div_ceil(8);
        for _ in 0..remaining_bytes {
            output.push((accum & 0xFF) as u8);
            accum >>= 8;
        }

        output
    }

    /// Encode literals in batches for better throughput.
    ///
    /// Processes 4 symbols at a time when possible.
    #[allow(dead_code)]
    pub fn encode_batch(&self, literals: &[u8]) -> Vec<u8> {
        if literals.len() < 8 {
            return self.encode(literals);
        }

        let mut output = Vec::with_capacity(literals.len() / 2 + 8);
        let mut accum: u64 = 0;
        let mut bits_in_accum: u32 = 0;

        // Process in reverse, 4 symbols at a time
        let len = literals.len();
        let mut i = len;

        // Handle tail (last 1-3 symbols)
        while i > 0 && !i.is_multiple_of(4) {
            i -= 1;
            let code = &self.codes[literals[i] as usize];
            if code.num_bits > 0 {
                accum |= (code.code as u64) << bits_in_accum;
                bits_in_accum += code.num_bits as u32;
                if bits_in_accum >= 8 {
                    output.push((accum & 0xFF) as u8);
                    accum >>= 8;
                    bits_in_accum -= 8;
                }
            }
        }

        // Process 4 symbols at a time
        while i >= 4 {
            i -= 4;

            // Load 4 codes
            let c0 = &self.codes[literals[i + 3] as usize];
            let c1 = &self.codes[literals[i + 2] as usize];
            let c2 = &self.codes[literals[i + 1] as usize];
            let c3 = &self.codes[literals[i] as usize];

            // Accumulate codes
            accum |= (c0.code as u64) << bits_in_accum;
            bits_in_accum += c0.num_bits as u32;
            accum |= (c1.code as u64) << bits_in_accum;
            bits_in_accum += c1.num_bits as u32;
            accum |= (c2.code as u64) << bits_in_accum;
            bits_in_accum += c2.num_bits as u32;
            accum |= (c3.code as u64) << bits_in_accum;
            bits_in_accum += c3.num_bits as u32;

            // Flush complete bytes
            while bits_in_accum >= 8 {
                output.push((accum & 0xFF) as u8);
                accum >>= 8;
                bits_in_accum -= 8;
            }
        }

        // Handle remaining symbols
        while i > 0 {
            i -= 1;
            let code = &self.codes[literals[i] as usize];
            if code.num_bits > 0 {
                accum |= (code.code as u64) << bits_in_accum;
                bits_in_accum += code.num_bits as u32;
                if bits_in_accum >= 8 {
                    output.push((accum & 0xFF) as u8);
                    accum >>= 8;
                    bits_in_accum -= 8;
                }
            }
        }

        // Add sentinel bit
        accum |= 1u64 << bits_in_accum;
        bits_in_accum += 1;

        // Flush remaining
        if bits_in_accum > 0 {
            output.push((accum & 0xFF) as u8);
        }

        output
    }

    /// Serialize weights in Zstd format (direct or FSE-compressed).
    ///
    /// For num_symbols <= 128: Uses direct format
    /// - header_byte = (num_symbols - 1) + 128
    /// - Followed by ceil(num_symbols / 2) bytes of 4-bit weights
    ///
    /// For num_symbols > 128: Uses FSE-compressed format
    /// - header_byte < 128 = compressed_size
    /// - Followed by FSE table and compressed weights
    pub fn serialize_weights(&self) -> Vec<u8> {
        // Find last non-zero weight
        let last_symbol = self
            .weights
            .iter()
            .enumerate()
            .filter(|&(_, w)| *w > 0)
            .map(|(i, _)| i)
            .max()
            .unwrap_or(0);

        let num_symbols = last_symbol + 1;

        // Calculate direct encoding size
        let direct_size = 1 + num_symbols.div_ceil(2);

        // Try FSE-compressed weights if beneficial
        // FSE is typically better when there are many zeros in the weight table
        // (sparse symbol usage like ASCII text)
        if num_symbols > 32 {
            let fse_result = self.serialize_weights_fse(num_symbols);
            if !fse_result.is_empty() && fse_result.len() < direct_size {
                return fse_result;
            }
        }

        // For >128 symbols, FSE is required
        if num_symbols > 128 {
            let fse_result = self.serialize_weights_fse(num_symbols);
            if !fse_result.is_empty() {
                return fse_result;
            }
            // FSE encoding failed, fall back to empty (caller should use raw block)
            return Vec::new();
        }

        // Direct encoding for <= 128 symbols
        let mut output = Vec::with_capacity(direct_size);

        if num_symbols > 0 {
            output.push(((num_symbols - 1) + 128) as u8);

            // Pack weights as 4-bit nibbles
            // Our decoder expects: Weight[i] in high nibble, Weight[i+1] in low nibble
            for i in (0..num_symbols).step_by(2) {
                let w1 = self.weights.get(i).copied().unwrap_or(0);
                let w2 = self.weights.get(i + 1).copied().unwrap_or(0);
                output.push((w1 << 4) | (w2 & 0x0F));
            }
        }

        output
    }

    /// Serialize weights using FSE compression for >128 symbols.
    ///
    /// Per RFC 8878 Section 4.2.1.1:
    /// - header_byte < 128 indicates FSE-compressed weights
    /// - header_byte value is the compressed size in bytes
    /// - Weights are encoded using an FSE table with max_symbol = 12 (weights 0-12)
    ///
    /// The FSE bitstream format for Huffman weights:
    /// 1. FSE table header (accuracy_log + probabilities)
    /// 2. Compressed bitstream read in reverse (from end with sentinel)
    ///    - Initial decoder state (accuracy_log bits, MSB-first from end)
    ///    - Encoded symbols' bits for state transitions
    fn serialize_weights_fse(&self, num_symbols: usize) -> Vec<u8> {
        // Count frequency of each weight value (weights are 0-11)
        let mut weight_freq = [0i16; 13]; // 0-12 possible weight values
        for i in 0..num_symbols {
            let w = self.weights.get(i).copied().unwrap_or(0) as usize;
            if w <= 12 {
                weight_freq[w] += 1;
            }
        }

        // Choose accuracy_log (6 is typical for Huffman weights per RFC 8878)
        const WEIGHT_ACCURACY_LOG: u8 = 6;
        let table_size = 1i16 << WEIGHT_ACCURACY_LOG;

        // Normalize frequencies to sum to table_size
        let total: i16 = weight_freq.iter().sum();
        if total == 0 {
            return Vec::new(); // No weights to encode
        }

        let mut normalized = [0i16; 13];
        let mut remaining = table_size;

        // First pass: assign proportional counts
        for (i, &freq) in weight_freq.iter().enumerate() {
            if freq > 0 {
                let norm = ((freq as i32 * table_size as i32) / total as i32).max(1) as i16;
                normalized[i] = norm;
                remaining -= norm;
            }
        }

        // Distribute remaining capacity to largest frequencies
        while remaining > 0 {
            let mut best_idx = 0;
            let mut best_freq = 0;
            for (i, &freq) in weight_freq.iter().enumerate() {
                if freq > best_freq && normalized[i] > 0 {
                    best_freq = freq;
                    best_idx = i;
                }
            }
            if best_freq == 0 {
                break;
            }
            normalized[best_idx] += 1;
            remaining -= 1;
        }

        // Handle over-allocation (can happen due to rounding)
        while remaining < 0 {
            let mut best_idx = 0;
            let mut best_norm = 0;
            for (i, &norm) in normalized.iter().enumerate() {
                if norm > 1 && norm > best_norm {
                    best_norm = norm;
                    best_idx = i;
                }
            }
            if best_norm <= 1 {
                break;
            }
            normalized[best_idx] -= 1;
            remaining += 1;
        }

        // Build FSE table from normalized frequencies
        let fse_table = match FseTable::build(&normalized, WEIGHT_ACCURACY_LOG, 12) {
            Ok(t) => t,
            Err(_) => return Vec::new(), // Failed to build table
        };

        // Serialize FSE table header
        let table_header = Self::serialize_fse_table_header(&normalized, WEIGHT_ACCURACY_LOG);

        // For FSE encoding, we use a simulation-based approach:
        // 1. Find the sequence of decoder states that produces our weight sequence
        // 2. Work backwards to compute the bits needed for each transition
        //
        // The decoder works as:
        //   state → (symbol, baseline, num_bits)
        //   next_state = baseline + read_bits(num_bits)
        //
        // So for encoding, we need to find states s0, s1, ... such that:
        //   table[s0].symbol = weight[0]
        //   table[s1].symbol = weight[1], and s1 = table[s0].baseline + bits0
        //   etc.

        // Collect weights to encode
        let weights_to_encode: Vec<u8> = (0..num_symbols)
            .map(|i| self.weights.get(i).copied().unwrap_or(0))
            .collect();

        // Find valid decoder state sequence
        // For each weight value, find all states that decode to it
        let mut states_for_symbol: [Vec<usize>; 13] = Default::default();
        for state in 0..fse_table.size() {
            let entry = fse_table.decode(state);
            if (entry.symbol as usize) < 13 {
                states_for_symbol[entry.symbol as usize].push(state);
            }
        }

        // Check if all weight values have at least one state
        for &w in &weights_to_encode {
            if states_for_symbol[w as usize].is_empty() {
                return Vec::new(); // Can't encode this weight
            }
        }

        // Use greedy approach: for each symbol, pick a state that works
        // and compute the bits needed for the transition from the previous state
        let mut state_sequence = Vec::with_capacity(num_symbols);
        let mut bits_sequence: Vec<(u32, u8)> = Vec::with_capacity(num_symbols);

        // First state: pick any state for the first weight
        let first_weight = weights_to_encode[0] as usize;
        let first_state = states_for_symbol[first_weight][0];
        state_sequence.push(first_state);

        // For each subsequent weight, find a state and compute transition bits
        for i in 1..num_symbols {
            let prev_state = state_sequence[i - 1];
            let prev_entry = fse_table.decode(prev_state);
            let target_weight = weights_to_encode[i] as usize;

            // We need: next_state = baseline + bits
            // where table[next_state].symbol = target_weight
            // and bits < (1 << num_bits)
            let baseline = prev_entry.baseline as usize;
            let num_bits = prev_entry.num_bits;
            let max_bits_value = 1usize << num_bits;

            // Find a state for target_weight that can be reached
            let mut found = false;
            for &candidate_state in &states_for_symbol[target_weight] {
                if candidate_state >= baseline && candidate_state < baseline + max_bits_value {
                    let bits = (candidate_state - baseline) as u32;
                    bits_sequence.push((bits, num_bits));
                    state_sequence.push(candidate_state);
                    found = true;
                    break;
                }
            }

            if !found {
                // Try wrapping around by using a different previous state
                // This is a simplification - full implementation would backtrack
                return Vec::new(); // Can't find valid encoding path
            }
        }

        // Now build the bitstream
        // The decoder reads:
        // 1. Initial state (accuracy_log bits) - this is state_sequence[0]
        // 2. For each symbol after the first, read bits for next state
        // 3. Final symbol is decoded from current state without reading more bits
        //
        // The bitstream is read in reverse (MSB-first from end).
        // So we write: [transition bits...][initial_state][sentinel]
        // And the bytes need to be arranged so that reversed reading works.

        // Build forward bitstream (we'll handle reversal through the writer)
        let mut bit_writer = FseBitWriter::new();

        // Write transition bits in order (they'll be read in reverse)
        // But wait - the reversed reader reads from the end, so the LAST bits
        // written should be read FIRST (as initial state).
        //
        // We need:
        // - Write initial_state last (so it's at the end, read first)
        // - Write transition bits before that
        //
        // Current approach: write bits in reverse order of how decoder reads
        // Decoder reads: init_state, then bits for s1, bits for s2, ...
        // We write: bits for s_{n-1}, bits for s_{n-2}, ..., bits for s1, init_state

        // Write transition bits in reverse order
        for i in (0..bits_sequence.len()).rev() {
            let (bits, num_bits) = bits_sequence[i];
            bit_writer.write_bits(bits, num_bits);
        }

        // Write initial state (will be read first by decoder)
        bit_writer.write_bits(state_sequence[0] as u32, WEIGHT_ACCURACY_LOG);

        // Finish bitstream (adds sentinel)
        let mut compressed_stream = bit_writer.finish();

        // The FseBitWriter produces bits in LSB-first order within bytes,
        // but the reversed reader reads MSB-first. We need to bit-reverse each byte.
        for byte in &mut compressed_stream {
            *byte = byte.reverse_bits();
        }

        // Combine: FSE table header + compressed stream
        let total_compressed_size = table_header.len() + compressed_stream.len();

        // Check if compressed size fits in header byte (< 128)
        if total_compressed_size >= 128 {
            return Vec::new(); // Too large for FSE format
        }

        // Build final output
        let mut output = Vec::with_capacity(1 + total_compressed_size);
        output.push(total_compressed_size as u8); // header < 128 = FSE compressed
        output.extend_from_slice(&table_header);
        output.extend_from_slice(&compressed_stream);

        output
    }

    /// Serialize FSE table header for Huffman weights.
    ///
    /// Format: 4-bit accuracy_log + variable-length probabilities
    #[allow(dead_code)]
    fn serialize_fse_table_header(normalized: &[i16; 13], accuracy_log: u8) -> Vec<u8> {
        let mut output = Vec::with_capacity(16);
        let mut bit_pos = 0u32;
        let mut accum = 0u64;

        // Write accuracy_log - 5 (4 bits)
        let acc_val = (accuracy_log.saturating_sub(5)) as u64;
        accum |= acc_val << bit_pos;
        bit_pos += 4;

        // Write probabilities using variable-length encoding
        let table_size = 1i32 << accuracy_log;
        let mut remaining = table_size;

        for &prob in normalized.iter() {
            if remaining <= 0 {
                break;
            }

            // Calculate bits needed to encode this probability
            let max_bits = 32 - (remaining + 1).leading_zeros();
            let threshold = (1i32 << max_bits) - 1 - remaining;

            // Encode probability
            let prob_val = if prob == -1 { 0 } else { prob as i32 };

            if prob_val < threshold {
                // Small value: use max_bits - 1 bits
                accum |= (prob_val as u64) << bit_pos;
                bit_pos += max_bits - 1;
            } else {
                // Large value: use max_bits bits
                let large = prob_val + threshold;
                accum |= (large as u64) << bit_pos;
                bit_pos += max_bits;
            }

            // Flush complete bytes
            while bit_pos >= 8 {
                output.push((accum & 0xFF) as u8);
                accum >>= 8;
                bit_pos -= 8;
            }

            // Update remaining
            if prob == -1 {
                remaining -= 1;
            } else {
                remaining -= prob as i32;
            }
        }

        // Flush remaining bits
        if bit_pos > 0 {
            output.push((accum & 0xFF) as u8);
        }

        output
    }

    /// Get maximum code length.
    #[inline]
    pub fn max_bits(&self) -> u8 {
        self.max_bits
    }

    /// Get number of symbols with codes.
    #[inline]
    pub fn num_symbols(&self) -> usize {
        self.num_symbols
    }

    /// Estimate compressed size.
    pub fn estimate_size(&self, literals: &[u8]) -> usize {
        let mut total_bits: usize = 0;
        for &byte in literals {
            total_bits += self.codes[byte as usize].num_bits as usize;
        }
        // Weight table size depends on last_symbol (highest symbol index), not unique count
        // Direct encoding uses (last_symbol + 1) symbols in the table
        let num_table_symbols = self.last_symbol + 1;
        let weight_table_size = 1 + num_table_symbols.div_ceil(2);
        total_bits.div_ceil(8) + weight_table_size
    }

    /// Get code for a symbol (for testing).
    #[cfg(test)]
    pub fn get_codes(&self) -> &[HuffmanCode; MAX_SYMBOLS] {
        &self.codes
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple() {
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }
        for _ in 0..25 {
            data.push(b'c');
        }

        let encoder = HuffmanEncoder::build(&data);
        assert!(encoder.is_some());

        let encoder = encoder.unwrap();
        assert!(encoder.num_symbols() >= 3);
    }

    #[test]
    fn test_build_too_small() {
        let data = b"small";
        let encoder = HuffmanEncoder::build(data);
        assert!(encoder.is_none());
    }

    #[test]
    fn test_encode_simple() {
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }

        let encoder = HuffmanEncoder::build(&data);
        if let Some(enc) = encoder {
            let compressed = enc.encode(&data);
            assert!(compressed.len() < data.len());
        }
    }

    #[test]
    fn test_encode_batch() {
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }
        for _ in 0..25 {
            data.push(b'c');
        }

        let encoder = HuffmanEncoder::build(&data);
        if let Some(enc) = encoder {
            let regular = enc.encode(&data);
            let batch = enc.encode_batch(&data);

            // Both should produce valid compressed data
            assert!(!regular.is_empty());
            assert!(!batch.is_empty());
        }
    }

    #[test]
    fn test_serialize_weights() {
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }

        let encoder = HuffmanEncoder::build(&data);
        if let Some(enc) = encoder {
            let weights = enc.serialize_weights();
            assert!(!weights.is_empty());
            assert!(weights[0] >= 128); // Direct format
        }
    }

    #[test]
    fn test_estimate_size() {
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(b'a');
        }
        for _ in 0..50 {
            data.push(b'b');
        }

        let encoder = HuffmanEncoder::build(&data);
        if let Some(enc) = encoder {
            let estimated = enc.estimate_size(&data);
            let actual = enc.encode(&data).len() + enc.serialize_weights().len();
            assert!(estimated <= actual + 10);
        }
    }

    #[test]
    fn test_frequency_counting() {
        let data = vec![0u8, 1, 2, 0, 1, 0, 0, 0, 1, 2, 3];
        let freq = HuffmanEncoder::count_frequencies(&data);

        assert_eq!(freq[0], 5);
        assert_eq!(freq[1], 3);
        assert_eq!(freq[2], 2);
        assert_eq!(freq[3], 1);
    }

    #[test]
    fn test_huffman_code_alignment() {
        // Verify HuffmanCode is properly aligned
        assert_eq!(std::mem::size_of::<HuffmanCode>(), 4);
        assert_eq!(std::mem::align_of::<HuffmanCode>(), 4);
    }

    #[test]
    fn test_many_symbols_uses_direct_encoding() {
        // Test with many unique symbols (but <= 128)
        // Create data with 100 unique symbols
        let mut data = Vec::new();
        for sym in 0..100u8 {
            for _ in 0..(100 - sym as usize).max(1) {
                data.push(sym);
            }
        }

        let encoder = HuffmanEncoder::build(&data);
        assert!(encoder.is_some(), "Should build encoder for 100 symbols");

        if let Some(enc) = encoder {
            let weights = enc.serialize_weights();
            assert!(!weights.is_empty(), "Should serialize weights");
            // Should use direct encoding (header >= 128)
            assert!(
                weights[0] >= 128,
                "Should use direct format for <= 128 symbols"
            );
        }
    }

    #[test]
    fn test_fse_table_header_serialization() {
        // Test the FSE table header serialization format
        let normalized = [32i16, 16, 8, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0];
        let header = HuffmanEncoder::serialize_fse_table_header(&normalized, 6);

        // Header should not be empty
        assert!(!header.is_empty());

        // First 4 bits should be accuracy_log - 5 = 1
        assert_eq!(header[0] & 0x0F, 1);
    }
}
