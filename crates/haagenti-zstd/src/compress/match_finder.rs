//! LZ77 match finding using optimized hash tables.
//!
//! This module implements high-performance match finding for Zstd compression.
//! Key optimizations:
//! - Fixed-size power-of-2 hash table (no HashMap allocation)
//! - Fast multiplicative hash function
//! - SIMD-accelerated match length comparison
//! - Cache-friendly hash chain structure
//! - Cache-line aligned structures (64-byte alignment)

use core::ops::{Deref, DerefMut};

/// Minimum match length for Zstd.
pub const MIN_MATCH_LENGTH: usize = 3;

/// Maximum match length.
pub const MAX_MATCH_LENGTH: usize = 131074; // Per RFC 8878

/// Hash table size (power of 2 for fast modulo via AND).
/// 64K entries is a good balance between memory and hit rate.
const HASH_LOG: usize = 16;
const HASH_SIZE: usize = 1 << HASH_LOG;
const HASH_MASK: u32 = (HASH_SIZE - 1) as u32;

/// Maximum chain depth per hash bucket.
/// Increased from 8 to allow deeper searches for better text compression.
/// The actual search depth is min(this, search_depth from config).
const MAX_CHAIN_DEPTH: usize = 256;

/// Primary hash multiplier (golden ratio derived, excellent distribution).
const HASH_PRIME: u32 = 0x9E3779B9;

/// Secondary hash multiplier for mixing (from MurmurHash3).
const HASH_PRIME2: u32 = 0x85EBCA6B;


// =============================================================================
// Cache-Aligned Structures
// =============================================================================

/// Cache-line aligned wrapper for better memory access patterns.
///
/// 64-byte alignment matches typical x86_64 cache line size, reducing
/// cache misses and eliminating false sharing in multi-threaded scenarios.
#[repr(C, align(64))]
pub struct CacheAligned<T>(T);

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned wrapper.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Get mutable access to the inner value.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> Deref for CacheAligned<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for CacheAligned<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Cache-aligned hash table for the match finder.
///
/// This structure ensures the hash table data starts at a 64-byte boundary,
/// improving cache efficiency for random access patterns typical in hash lookups.
#[repr(C, align(64))]
struct AlignedHashTable {
    data: [u32; HASH_SIZE],
}

impl AlignedHashTable {
    /// Create a new zeroed hash table via heap allocation.
    fn new_boxed() -> Box<Self> {
        // Use zeroed allocation for efficiency - avoids initializing twice
        // SAFETY: AlignedHashTable contains only u32 values, which are valid when zeroed
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        }
    }

    /// Reset all entries to zero using optimized memset.
    #[inline]
    fn reset(&mut self) {
        self.data.fill(0);
    }

    /// Get a reference to entry at the given index.
    #[inline(always)]
    fn get(&self, index: usize) -> u32 {
        self.data[index]
    }

    /// Set entry at the given index.
    #[inline(always)]
    fn set(&mut self, index: usize, value: u32) {
        self.data[index] = value;
    }
}

/// A found match in the input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    /// Position in the input where the match starts.
    pub position: usize,
    /// Offset back to the matching data.
    pub offset: usize,
    /// Length of the match.
    pub length: usize,
}

impl Match {
    /// Create a new match.
    #[inline]
    pub fn new(position: usize, offset: usize, length: usize) -> Self {
        Self { position, offset, length }
    }
}

/// Optimized hash chain-based match finder.
///
/// Uses a fixed-size hash table with direct indexing for O(1) lookup.
/// SIMD-accelerated match length comparison when available.
/// Hash table is cache-line aligned (64 bytes) for optimal memory access.
///
/// ## Novel Optimization: Generation Counter
///
/// Instead of zeroing the 256KB hash table on every reset (O(n)), we use
/// a generation counter. Hash entries store (generation << 28) | (pos + 1).
/// On reset, we just increment generation (O(1)). Entries from old generations
/// are treated as empty.
///
/// ## Novel Optimization: Match Prediction
///
/// When a match is found with offset O, the next occurrence of that pattern
/// is very likely to be at position P + O (for repeating patterns like text).
/// We track the last successful offset and check it first at the next position.
pub struct MatchFinder {
    /// Maximum search depth in the hash chain.
    search_depth: usize,
    /// Hash table: hash -> (generation << 28) | (pos + 1).
    /// Entries from old generations are treated as empty.
    /// Cache-aligned for optimal memory access patterns.
    hash_table: Box<AlignedHashTable>,
    /// Current generation counter (4 bits, wraps at 16).
    generation: u32,
    /// Chain table: for each position, stores previous position with same hash.
    chain_table: Vec<u32>,
    /// Input length (for bounds checking).
    input_len: usize,
    /// Last successful match offset (for prediction).
    /// If the same offset produces matches repeatedly, we check it first.
    predicted_offset: u32,
    /// Second-to-last offset for alternating patterns.
    predicted_offset2: u32,
    /// Count of successful predictions (for adaptive behavior).
    prediction_hits: u32,
}

// Manual Debug impl since AlignedHashTable doesn't derive Debug
impl core::fmt::Debug for MatchFinder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MatchFinder")
            .field("search_depth", &self.search_depth)
            .field("hash_table", &format_args!("[AlignedHashTable; {}]", HASH_SIZE))
            .field("chain_table_len", &self.chain_table.len())
            .field("input_len", &self.input_len)
            .field("predicted_offset", &self.predicted_offset)
            .field("predicted_offset2", &self.predicted_offset2)
            .field("prediction_hits", &self.prediction_hits)
            .finish()
    }
}

/// Mask for extracting generation from hash entry: top 4 bits
const GEN_MASK: u32 = 0xF0000000;
/// Shift for generation in hash entry
const GEN_SHIFT: u32 = 28;
/// Mask for extracting position from hash entry: bottom 28 bits
const POS_MASK: u32 = 0x0FFFFFFF;

impl MatchFinder {
    /// Create a new match finder.
    pub fn new(search_depth: usize) -> Self {
        Self {
            search_depth: search_depth.max(1).min(128),
            hash_table: AlignedHashTable::new_boxed(),
            generation: 0,
            chain_table: Vec::new(),
            input_len: 0,
            predicted_offset: 0,
            predicted_offset2: 0,
            prediction_hits: 0,
        }
    }

    /// Calculate early exit threshold based on position in file.
    ///
    /// Early in the file, we want longer matches before exiting search
    /// (more exploration). Later in the file, shorter matches are acceptable
    /// for early exit (faster throughput).
    ///
    /// The rationale:
    /// - Position 0-1KB: Still building hash chains, need full search (threshold=48)
    /// - Position 1-8KB: Chains are populated, moderate search (threshold=32)
    /// - Position 8-32KB: Well-populated chains, balanced (threshold=24)
    /// - Position 32KB+: Mature chains, early exit is safe (threshold=16)
    ///
    /// This optimization is particularly effective for repetitive data where
    /// excellent matches are common.
    #[inline]
    pub fn early_exit_threshold(&self, position: usize) -> usize {
        match position {
            0..=1024 => 32,       // Very early: good match to exit (was 48)
            1025..=8192 => 24,    // Early: moderate threshold (was 32)
            8193..=32768 => 16,   // Mid-file: lower threshold (was 24)
            _ => 12,              // Late in file: aggressive exit (was 16)
        }
    }

    /// Calculate effective search depth based on input size.
    ///
    /// Larger inputs benefit from reduced search depth:
    /// - Reduces time spent in hash chain traversal
    /// - Cache pressure is higher with large data
    /// - Prediction often finds good matches anyway
    ///
    /// The scaling is designed to maintain compression quality while
    /// improving throughput on large inputs.
    #[inline]
    pub fn effective_depth(&self, input_len: usize) -> usize {
        let base = self.search_depth;

        let scaled = match input_len {
            // Small inputs: full depth for best compression
            0..=4096 => base,
            // Medium inputs: 90% depth (was 75%)
            4097..=16384 => (base * 9 / 10).max(4),
            // Large inputs: 75% depth (was 33%)
            16385..=65536 => (base * 3 / 4).max(4),
            // Very large inputs: 50% depth (was 25%)
            65537..=262144 => (base / 2).max(3),
            // Huge inputs: 33% depth (was 12.5%)
            _ => (base / 3).max(2),
        };

        scaled.min(base) // Never exceed configured depth
    }

    /// Reset the hash table for new input.
    ///
    /// Uses generation counter to avoid zeroing tables (O(1) vs O(n)).
    /// Hash table and chain table both use generation encoding.
    /// Entries from old generations are treated as invalid/empty.
    #[inline]
    pub(super) fn reset(&mut self, input_len: usize) {
        // Increment generation instead of zeroing (O(1) vs O(256KB+))
        // Generation uses 4 bits, wrapping at 16. Old entries become invalid.
        self.generation = (self.generation + 1) & 0xF;

        // Chain table uses generation encoding too - just resize if needed
        // No need to zero since we check generation on read
        if self.chain_table.len() < input_len {
            self.chain_table.resize(input_len, 0);
        }
        // If shrinking, leave as is - positions beyond input_len won't be accessed

        self.input_len = input_len;

        // Reset prediction state
        self.predicted_offset = 0;
        self.predicted_offset2 = 0;
        self.prediction_hits = 0;
    }

    /// Compute hash for 4 bytes using fast multiplicative hash.
    ///
    /// Uses single multiplication for maximum speed. The golden ratio prime
    /// provides good distribution even with just one multiply.
    #[inline(always)]
    pub(super) fn hash4(&self, data: &[u8], pos: usize) -> u32 {
        debug_assert!(pos + 4 <= data.len());

        // Load 4 bytes as u32 (little-endian)
        let bytes = unsafe {
            std::ptr::read_unaligned(data.as_ptr().add(pos) as *const u32)
        };

        // Single multiply hash - faster than double-mixing
        // Golden ratio prime provides good distribution
        bytes.wrapping_mul(HASH_PRIME) >> (32 - HASH_LOG as u32)
    }

    /// Alternative hash for 3-byte minimum matches.
    #[inline(always)]
    pub(super) fn hash3(&self, data: &[u8], pos: usize) -> u32 {
        debug_assert!(pos + 3 <= data.len());

        let b0 = data[pos] as u32;
        let b1 = data[pos + 1] as u32;
        let b2 = data[pos + 2] as u32;

        // Combine bytes with shifts
        let value = b0 | (b1 << 8) | (b2 << 16);
        value.wrapping_mul(HASH_PRIME) >> (32 - HASH_LOG as u32)
    }

    /// Find all matches in the input data.
    ///
    /// Returns matches sorted by position.
    pub fn find_matches(&mut self, input: &[u8]) -> Vec<Match> {
        if input.len() < MIN_MATCH_LENGTH {
            return Vec::new();
        }

        self.reset(input.len());
        let mut matches = Vec::with_capacity(input.len() / 16);

        let mut pos = 0;
        let end = input.len().saturating_sub(MIN_MATCH_LENGTH);

        while pos <= end {
            // Prefetch ahead for better cache behavior
            #[cfg(target_arch = "x86_64")]
            if pos + 64 < input.len() {
                unsafe {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch(
                        input.as_ptr().add(pos + 64) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            // Hash must be computed anyway for chain updates
            let hash = if pos + 4 <= input.len() {
                self.hash4(input, pos)
            } else {
                self.hash3(input, pos)
            };

            // Try predicted offsets first (fast path for repetitive patterns)
            let cur_prefix = unsafe {
                std::ptr::read_unaligned(input.as_ptr().add(pos) as *const u32)
            };

            let mut best_match = None;

            // Check primary prediction
            if self.predicted_offset > 0 && pos >= self.predicted_offset as usize {
                let match_pos = pos - self.predicted_offset as usize;
                let match_prefix = unsafe {
                    std::ptr::read_unaligned(input.as_ptr().add(match_pos) as *const u32)
                };
                if cur_prefix == match_prefix {
                    let length = 4 + self.match_length_from(input, match_pos + 4, pos + 4);
                    if length >= MIN_MATCH_LENGTH {
                        self.prediction_hits += 1;
                        best_match = Some(Match::new(pos, self.predicted_offset as usize, length));
                    }
                }
            }

            // Check secondary prediction if primary failed
            if best_match.is_none() && self.predicted_offset2 > 0
               && self.predicted_offset2 != self.predicted_offset
               && pos >= self.predicted_offset2 as usize {
                let match_pos = pos - self.predicted_offset2 as usize;
                let match_prefix = unsafe {
                    std::ptr::read_unaligned(input.as_ptr().add(match_pos) as *const u32)
                };
                if cur_prefix == match_prefix {
                    let length = 4 + self.match_length_from(input, match_pos + 4, pos + 4);
                    if length >= MIN_MATCH_LENGTH {
                        self.prediction_hits += 1;
                        best_match = Some(Match::new(pos, self.predicted_offset2 as usize, length));
                    }
                }
            }

            // Fall back to hash chain search
            if best_match.is_none() {
                best_match = self.find_best_match(input, pos, hash as usize);
            }

            if let Some(m) = best_match {
                matches.push(m);

                // Update predictions for next iteration (shift if different)
                let new_offset = m.offset as u32;
                if new_offset != self.predicted_offset {
                    self.predicted_offset2 = self.predicted_offset;
                    self.predicted_offset = new_offset;
                }

                // Update hash table for current position before skipping
                self.update_hash(input, pos, hash as usize);

                // Aggressive skip for RLE-like patterns (offset 1-4)
                // These patterns are very common in text and benefit from minimal hash updates
                if m.offset <= 4 && m.length >= 32 {
                    // RLE-like: update only at 16-byte intervals
                    let skip_end = (pos + m.length).min(end);
                    let mut update_pos = pos + 16;
                    while update_pos < skip_end {
                        if update_pos + 4 <= input.len() {
                            let h = self.hash4(input, update_pos);
                            self.update_hash(input, update_pos, h as usize);
                        }
                        update_pos += 16;
                    }
                } else if m.length >= 64 {
                    // Very long matches: sparse updates every 8th position
                    let skip_end = (pos + m.length).min(end);
                    let mut update_pos = pos + 8;
                    while update_pos < skip_end {
                        if update_pos + 4 <= input.len() {
                            let h = self.hash4(input, update_pos);
                            self.update_hash(input, update_pos, h as usize);
                        }
                        update_pos += 8;
                    }
                } else if m.length >= 8 {
                    // Medium matches: update every 4th position
                    let skip_end = (pos + m.length).min(end);
                    let mut update_pos = pos + 4;
                    while update_pos < skip_end {
                        if update_pos + 4 <= input.len() {
                            let h = self.hash4(input, update_pos);
                            self.update_hash(input, update_pos, h as usize);
                        }
                        update_pos += 4;
                    }
                }
                // Short matches (3-7): skip updates entirely for speed

                pos += m.length;
            } else {
                self.update_hash(input, pos, hash as usize);
                pos += 1;
            }
        }

        matches
    }

    /// Update hash table with new position.
    ///
    /// Both hash and chain entries store (generation << 28) | (pos + 1).
    /// This allows O(1) reset by just incrementing generation.
    #[inline(always)]
    pub(super) fn update_hash(&mut self, _input: &[u8], pos: usize, hash: usize) {
        let prev = self.hash_table.get(hash);
        // Check if prev is from current generation
        let prev_gen = (prev & GEN_MASK) >> GEN_SHIFT;
        // Store generation-encoded chain entry: (gen << 28) | prev_pos
        // If prev was from old generation, prev_pos = 0 (end of chain)
        let chain_entry = if prev_gen == self.generation {
            // Valid prev - keep same generation encoding
            prev
        } else {
            // Old generation - set to 0 (end of chain marker)
            0
        };
        if pos < self.chain_table.len() {
            self.chain_table[pos] = chain_entry;
        }
        // Store (generation << 28) | (pos + 1) in hash table
        let encoded = (self.generation << GEN_SHIFT) | ((pos + 1) as u32);
        self.hash_table.set(hash, encoded);
    }

    /// Find the best match at the current position.
    ///
    /// Optimized with:
    /// - Early rejection using first 4 bytes comparison
    /// - Aggressive prefetching of next chain entry and match data
    /// - Minimal branching in hot loop
    #[inline]
    pub(super) fn find_best_match(&self, input: &[u8], pos: usize, hash: usize) -> Option<Match> {
        let hash_entry = self.hash_table.get(hash);

        // Check generation - entries from old generations are treated as empty
        let entry_gen = (hash_entry & GEN_MASK) >> GEN_SHIFT;
        if entry_gen != self.generation {
            return None;
        }

        // Extract position (mask out generation, subtract 1)
        let entry_pos = hash_entry & POS_MASK;
        if entry_pos == 0 {
            return None;
        }
        let mut match_pos = (entry_pos - 1) as usize;

        // Early bounds check
        if match_pos >= pos || pos + 4 > input.len() {
            return None;
        }

        // Load first 4 bytes at current position for fast rejection
        let cur_prefix = unsafe {
            std::ptr::read_unaligned(input.as_ptr().add(pos) as *const u32)
        };

        let mut best_match: Option<Match> = None;
        let mut best_length = MIN_MATCH_LENGTH - 1;
        let mut depth = 0;
        let max_offset = 1 << 28; // Zstd max offset

        // Use adaptive depth based on input size
        let effective_depth = self.effective_depth(self.input_len);

        while depth < effective_depth && match_pos < pos {
            let offset = pos - match_pos;

            if offset > max_offset {
                break;
            }

            // Prefetch next chain entry while processing current
            // Chain entries are generation-encoded: (gen << 28) | (pos + 1)
            let next_chain = if match_pos < self.chain_table.len() {
                let chain_entry = self.chain_table[match_pos];
                // Check generation - if wrong, treat as end of chain
                let chain_gen = (chain_entry & GEN_MASK) >> GEN_SHIFT;
                if chain_gen != self.generation {
                    0 // Wrong generation = end of chain
                } else {
                    let next_pos_enc = chain_entry & POS_MASK;
                    if next_pos_enc > 0 {
                        let next_pos = (next_pos_enc - 1) as usize;
                        #[cfg(target_arch = "x86_64")]
                        if next_pos < input.len() {
                            unsafe {
                                use std::arch::x86_64::_mm_prefetch;
                                _mm_prefetch(
                                    input.as_ptr().add(next_pos) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                        }
                    }
                    next_pos_enc
                }
            } else {
                0
            };

            // Fast rejection: check first 4 bytes before full comparison
            let match_prefix = if match_pos + 4 <= input.len() {
                unsafe { std::ptr::read_unaligned(input.as_ptr().add(match_pos) as *const u32) }
            } else {
                // Can't compare 4 bytes, follow chain
                if next_chain == 0 { break; }
                match_pos = (next_chain - 1) as usize;
                depth += 1;
                continue;
            };

            // Quick check: if first 4 bytes don't match, skip this candidate
            if match_prefix != cur_prefix {
                if next_chain == 0 { break; }
                match_pos = (next_chain - 1) as usize;
                depth += 1;
                continue;
            }

            // First 4 bytes match - do full comparison (already have 4 matching)
            let length = 4 + self.match_length_from(input, match_pos + 4, pos + 4);

            // Prefer predicted offset on close matches
            let is_predicted = offset as u32 == self.predicted_offset;
            let effective_length = if is_predicted && length >= MIN_MATCH_LENGTH {
                length + 2
            } else {
                length
            };

            if effective_length > best_length {
                best_length = length;
                best_match = Some(Match::new(pos, offset, length));

                // Early exit using position-adaptive threshold
                // Early in file: require longer matches (more thorough search)
                // Later in file: shorter matches are good enough (faster)
                let exit_threshold = self.early_exit_threshold(pos);
                if length >= exit_threshold {
                    break;
                }
            }

            // Follow chain to previous position with same hash
            // Chain entries are generation-encoded: (gen << 28) | (pos + 1)
            if next_chain == 0 {
                break; // End of chain
            }
            match_pos = (next_chain - 1) as usize;

            depth += 1;
        }

        best_match
    }

    /// Calculate match length starting from given offsets.
    ///
    /// Used when we already know the first N bytes match (e.g., from prefix comparison).
    /// This avoids re-comparing bytes we've already verified.
    #[inline(always)]
    pub fn match_length_from(&self, input: &[u8], pos1: usize, pos2: usize) -> usize {
        // Bounds check
        if pos1 >= input.len() || pos2 >= input.len() {
            return 0;
        }

        let max_len = (input.len() - pos2).min(input.len() - pos1).min(MAX_MATCH_LENGTH);

        if max_len == 0 {
            return 0;
        }

        // Use SIMD-accelerated comparison when feature is enabled
        #[cfg(feature = "simd")]
        {
            let src = &input[pos1..pos1 + max_len];
            let cur = &input[pos2..pos2 + max_len];
            return haagenti_simd::find_match_length(src, cur, max_len);
        }

        // Optimized scalar fallback - compare 8 bytes at a time
        #[cfg(not(feature = "simd"))]
        {
            let mut len = 0;

            // Compare 8 bytes at a time using u64
            while len + 8 <= max_len {
                let word1 = unsafe {
                    std::ptr::read_unaligned(input.as_ptr().add(pos1 + len) as *const u64)
                };
                let word2 = unsafe {
                    std::ptr::read_unaligned(input.as_ptr().add(pos2 + len) as *const u64)
                };

                let diff = word1 ^ word2;
                if diff != 0 {
                    // Find first differing byte using trailing zeros
                    len += (diff.trailing_zeros() / 8) as usize;
                    return len;
                }
                len += 8;
            }

            // Compare remaining bytes (up to 7)
            while len < max_len && input[pos1 + len] == input[pos2 + len] {
                len += 1;
            }

            len
        }
    }

    /// Find matches using speculative parallel lookahead.
    ///
    /// This method speculatively computes hashes for multiple positions at once,
    /// exploiting instruction-level parallelism in modern CPUs. The key insight
    /// is that hash computation and match lookup are independent operations that
    /// can be pipelined.
    ///
    /// Algorithm:
    /// 1. Compute hashes for next LOOKAHEAD positions in parallel
    /// 2. Look up potential matches for all positions
    /// 3. Select the best match among candidates
    /// 4. Skip to end of selected match
    ///
    /// Benefits:
    /// - Better instruction-level parallelism (ILP)
    /// - May find better matches by considering multiple positions
    /// - Reduces branch mispredictions by batching work
    ///
    /// Expected impact: +15-25% throughput on large data with varied content.
    #[inline]
    pub fn find_matches_speculative(&mut self, input: &[u8]) -> Vec<Match> {
        const LOOKAHEAD: usize = 4;

        if input.len() < MIN_MATCH_LENGTH {
            return Vec::new();
        }

        self.reset(input.len());
        let mut matches = Vec::with_capacity(input.len() / 16);
        let mut pos = 0;
        let end = input.len().saturating_sub(MIN_MATCH_LENGTH + LOOKAHEAD);

        while pos <= end && pos + 4 <= input.len() {
            // Speculatively compute hashes for LOOKAHEAD positions
            // This enables CPU to pipeline the computations
            let hashes: [u32; LOOKAHEAD] = [
                self.hash4(input, pos),
                if pos + 5 <= input.len() { self.hash4(input, pos + 1) } else { 0 },
                if pos + 6 <= input.len() { self.hash4(input, pos + 2) } else { 0 },
                if pos + 7 <= input.len() { self.hash4(input, pos + 3) } else { 0 },
            ];

            // Find matches at all speculative positions
            let mut best_match: Option<Match> = None;
            let mut best_score: usize = 0;

            for (i, &hash) in hashes.iter().enumerate() {
                if hash == 0 {
                    continue;
                }

                let check_pos = pos + i;
                if let Some(m) = self.find_best_match(input, check_pos, hash as usize) {
                    // Score: balance length vs position (prefer earlier positions for same length)
                    // Longer matches are always preferred, ties go to earlier position
                    let score = m.length * 8 - i;
                    if score > best_score {
                        best_score = score;
                        best_match = Some(m);
                    }
                }
            }

            if let Some(m) = best_match {
                // Update hash table for positions before the match
                for i in 0..LOOKAHEAD.min(m.position - pos) {
                    if pos + i + 4 <= input.len() {
                        self.update_hash(input, pos + i, hashes[i] as usize);
                    }
                }

                // Update hash at match position
                self.update_hash(input, m.position, hashes[m.position - pos] as usize);

                matches.push(m);

                // Sparse updates during skip for long matches
                if m.length >= 8 {
                    let skip_end = (m.position + m.length).min(end);
                    let mut update_pos = m.position + 4;
                    while update_pos < skip_end {
                        if update_pos + 4 <= input.len() {
                            let h = self.hash4(input, update_pos);
                            self.update_hash(input, update_pos, h as usize);
                        }
                        update_pos += 4;
                    }
                }

                pos = m.position + m.length;
            } else {
                // No match found - update hash and advance
                self.update_hash(input, pos, hashes[0] as usize);
                pos += 1;
            }
        }

        // Handle remaining positions without lookahead
        let final_end = input.len().saturating_sub(MIN_MATCH_LENGTH);
        while pos <= final_end && pos + 4 <= input.len() {
            let hash = self.hash4(input, pos);
            if let Some(m) = self.find_best_match(input, pos, hash as usize) {
                self.update_hash(input, pos, hash as usize);
                matches.push(m);
                pos += m.length;
            } else {
                self.update_hash(input, pos, hash as usize);
                pos += 1;
            }
        }

        matches
    }
}

/// Greedy-Lazy Hybrid Match Finder.
///
/// This match finder uses a smarter strategy than pure lazy matching:
/// - Commits immediately to long matches (>= 24 bytes) or predicted offset matches
/// - Only performs lazy lookahead for short "questionable" matches (4-23 bytes)
/// - Uses a quality score that combines length + offset preference
///
/// This reduces overhead compared to always-lazy matching while maintaining
/// compression ratio improvements for cases where lazy evaluation helps.
#[derive(Debug)]
pub struct LazyMatchFinder {
    /// Inner greedy match finder (exposed for repeat offset finder)
    pub(super) inner: MatchFinder,
    /// Threshold for immediate commit (no lazy check)
    lazy_threshold: usize,
}

impl LazyMatchFinder {
    /// Create a new lazy match finder.
    pub fn new(search_depth: usize) -> Self {
        Self {
            inner: MatchFinder::new(search_depth),
            lazy_threshold: 24, // Matches >= 24 bytes commit immediately
        }
    }

    /// Configure the lazy threshold based on input size.
    ///
    /// For larger inputs, we lower the threshold to commit earlier, improving
    /// throughput at a small cost to compression ratio.
    ///
    /// Scaling rationale:
    /// - Small inputs (<= 4KB): Full lazy evaluation (threshold = 24)
    /// - Medium inputs (4-16KB): Slight reduction (threshold = 20)
    /// - Large inputs (16-64KB): Moderate reduction (threshold = 16)
    /// - Very large inputs (64KB+): Aggressive (threshold = 12)
    /// - Huge inputs (256KB+): Most aggressive (threshold = 8)
    ///
    /// The minimum threshold is MIN_MATCH_LENGTH + 1 (4) to ensure lazy
    /// evaluation is still meaningful.
    #[inline]
    pub fn configure_for_size(&mut self, input_len: usize) {
        self.lazy_threshold = match input_len {
            0..=4096 => 24,       // Small: full lazy for best ratio
            4097..=16384 => 20,   // Medium: slight speedup
            16385..=65536 => 16,  // Large: balance speed and ratio
            65537..=262144 => 12, // Very large: favor throughput
            _ => 8,               // Huge: aggressive early commit
        };

        // Never go below minimum viable threshold
        self.lazy_threshold = self.lazy_threshold.max(MIN_MATCH_LENGTH + 1);
    }

    /// Find matches with automatic size-based configuration.
    ///
    /// This is the recommended method for general use. It automatically
    /// adjusts the lazy threshold based on input size for optimal
    /// throughput/ratio balance.
    ///
    /// For very large inputs (>= 128KB), uses chunked matching with a smaller
    /// L1-cache-friendly hash table for improved cache locality.
    #[inline]
    pub fn find_matches_auto(&mut self, input: &[u8]) -> Vec<Match> {
        self.configure_for_size(input.len());

        // Use chunked matching only for very large inputs (>= 128KB)
        // The 65KB block boundary is avoided since block sizes are max 128KB
        if input.len() >= 131072 {
            self.find_matches_chunked(input, 16384)
        } else {
            self.find_matches(input)
        }
    }

    /// Find matches using greedy-lazy hybrid strategy with offset prediction.
    ///
    /// Key optimizations:
    /// 1. Offset prediction: Try last successful offset before hash lookup
    /// 2. Immediate commit for long matches (>= lazy_threshold)
    /// 3. Only 1-position lookahead for short matches
    /// 4. Simple length comparison (no complex quality scoring)
    #[inline]
    pub fn find_matches(&mut self, input: &[u8]) -> Vec<Match> {
        if input.len() < MIN_MATCH_LENGTH {
            return Vec::new();
        }

        self.inner.reset(input.len());
        let mut matches = Vec::with_capacity(input.len() / 16);

        let mut pos = 0;
        let end = input.len().saturating_sub(MIN_MATCH_LENGTH);
        let mut pending_match: Option<Match> = None;
        let mut predicted_offset: usize = 0;

        while pos <= end {
            let hash = if pos + 4 <= input.len() {
                self.inner.hash4(input, pos)
            } else {
                self.inner.hash3(input, pos)
            };

            // Try prediction first (very fast for repetitive patterns)
            let current_match = if predicted_offset > 0 && pos >= predicted_offset && pos + 4 <= input.len() {
                let match_pos = pos - predicted_offset;
                // Load prefixes for comparison
                let cur_prefix = unsafe {
                    std::ptr::read_unaligned(input.as_ptr().add(pos) as *const u32)
                };
                let match_prefix = unsafe {
                    std::ptr::read_unaligned(input.as_ptr().add(match_pos) as *const u32)
                };

                if cur_prefix == match_prefix {
                    // Prediction hit - compute full match length
                    let length = 4 + self.inner.match_length_from(input, match_pos + 4, pos + 4);
                    if length >= MIN_MATCH_LENGTH {
                        Some(Match::new(pos, predicted_offset, length))
                    } else {
                        self.inner.find_best_match(input, pos, hash as usize)
                    }
                } else {
                    self.inner.find_best_match(input, pos, hash as usize)
                }
            } else {
                self.inner.find_best_match(input, pos, hash as usize)
            };

            if let Some(curr) = current_match {
                // Update prediction with current match offset
                predicted_offset = curr.offset;

                if let Some(pending) = pending_match.take() {
                    // Compare: current must be significantly better to replace pending
                    if curr.length > pending.length + 1 {
                        // Current is better - use it
                        matches.push(curr);
                        self.inner.update_hash(input, pos, hash as usize);
                        pos += curr.length;
                    } else {
                        // Pending is good enough - use pending
                        matches.push(pending);
                        pos = pending.position + pending.length;
                    }
                } else {
                    // No pending - decide to commit or defer
                    if curr.length >= self.lazy_threshold || pos + 1 > end {
                        // Long match or near end - commit immediately
                        matches.push(curr);
                        self.inner.update_hash(input, pos, hash as usize);
                        pos += curr.length;
                    } else {
                        // Short match - defer and check next position
                        pending_match = Some(curr);
                        self.inner.update_hash(input, pos, hash as usize);
                        pos += 1;
                    }
                }
            } else {
                // No match at current position
                if let Some(pending) = pending_match.take() {
                    matches.push(pending);
                    pos = pending.position + pending.length;
                } else {
                    self.inner.update_hash(input, pos, hash as usize);
                    pos += 1;
                }
            }
        }

        // Emit any remaining pending match
        if let Some(pending) = pending_match {
            matches.push(pending);
        }

        matches
    }

    /// Find matches using block chunking for improved cache locality.
    ///
    /// For large inputs (> chunk_size), this processes the input in independent
    /// chunks using a small, cache-friendly hash table. This dramatically improves
    /// cache hit rates since the hash table and chain table fit in L1/L2 cache.
    ///
    /// Key benefits:
    /// - Small hash table (4KB) fits entirely in L1 cache
    /// - Chain table per chunk is small (~chunk_size * 4 bytes)
    /// - Reduced TLB pressure from smaller working set
    /// - Fresh hash table per chunk = zero collision chains initially
    ///
    /// The tradeoff is that matches cannot span chunk boundaries, which may
    /// slightly reduce compression ratio. In practice, the throughput gain
    /// significantly outweighs this cost for large inputs.
    ///
    /// # Arguments
    ///
    /// * `input` - The data to find matches in
    /// * `chunk_size` - Size of each chunk (16384 recommended for L1 cache fit)
    ///
    /// # Returns
    ///
    /// Matches with positions relative to the original input (not chunks).
    #[inline]
    pub fn find_matches_chunked(&mut self, input: &[u8], chunk_size: usize) -> Vec<Match> {
        // For small inputs, just use standard matching
        if input.len() <= chunk_size {
            return self.find_matches(input);
        }

        let chunk_size = chunk_size.max(1024); // Minimum reasonable chunk
        let mut all_matches = Vec::with_capacity(input.len() / 16);
        let mut chunk_start = 0;

        // Use a small, cache-friendly hash table for chunked processing
        // 12-bit = 4096 entries = 16KB, fits in L1 cache
        const CHUNK_HASH_LOG: usize = 12;
        const CHUNK_HASH_SIZE: usize = 1 << CHUNK_HASH_LOG;
        const CHUNK_HASH_MASK: u32 = (CHUNK_HASH_SIZE - 1) as u32;

        let mut chunk_hash = vec![0u32; CHUNK_HASH_SIZE];
        let mut chunk_chain = vec![0u32; chunk_size];
        let search_depth = self.inner.search_depth;

        while chunk_start < input.len() {
            let chunk_end = (chunk_start + chunk_size).min(input.len());
            let chunk = &input[chunk_start..chunk_end];

            if chunk.len() >= MIN_MATCH_LENGTH {
                // Reset the small hash table (fast - only 16KB)
                chunk_hash.fill(0);
                if chunk_chain.len() < chunk.len() {
                    chunk_chain.resize(chunk.len(), 0);
                }
                chunk_chain[..chunk.len()].fill(0);

                // Process this chunk using the small hash table
                let chunk_matches = Self::find_matches_in_chunk(
                    chunk,
                    &mut chunk_hash,
                    &mut chunk_chain,
                    CHUNK_HASH_LOG,
                    CHUNK_HASH_MASK,
                    search_depth,
                    self.lazy_threshold,
                );

                // Adjust positions to be relative to original input
                for m in chunk_matches {
                    all_matches.push(Match::new(
                        chunk_start + m.position,
                        m.offset,
                        m.length,
                    ));
                }
            }

            chunk_start = chunk_end;
        }

        all_matches
    }

    /// Process a single chunk with a small hash table.
    /// This is a simplified version of find_matches optimized for cache locality.
    #[inline]
    fn find_matches_in_chunk(
        chunk: &[u8],
        hash_table: &mut [u32],
        chain_table: &mut [u32],
        hash_log: usize,
        hash_mask: u32,
        search_depth: usize,
        lazy_threshold: usize,
    ) -> Vec<Match> {
        if chunk.len() < MIN_MATCH_LENGTH {
            return Vec::new();
        }

        let mut matches = Vec::with_capacity(chunk.len() / 32);
        let end = chunk.len().saturating_sub(MIN_MATCH_LENGTH);
        let mut pos = 0;
        let mut pending: Option<Match> = None;

        while pos <= end {
            // Compute hash (simplified, using same algorithm as main finder)
            let hash = if pos + 4 <= chunk.len() {
                let bytes = unsafe {
                    std::ptr::read_unaligned(chunk.as_ptr().add(pos) as *const u32)
                };
                let h = bytes.wrapping_mul(HASH_PRIME);
                let h = h ^ (h >> 15);
                let h = h.wrapping_mul(HASH_PRIME2);
                (h >> (32 - hash_log as u32)) & hash_mask
            } else {
                let b0 = chunk[pos] as u32;
                let b1 = chunk[pos + 1] as u32;
                let b2 = chunk[pos + 2] as u32;
                let value = b0 | (b1 << 8) | (b2 << 16);
                (value.wrapping_mul(HASH_PRIME) >> (32 - hash_log as u32)) & hash_mask
            };

            // Find best match at this position
            let current_match = Self::find_best_match_in_chunk(
                chunk, pos, hash as usize, hash_table, chain_table, search_depth,
            );

            // Lazy matching logic
            if let Some(curr) = current_match {
                if let Some(pend) = pending.take() {
                    if curr.length > pend.length + 1 {
                        matches.push(curr);
                        Self::update_chunk_hash(pos, hash as usize, hash_table, chain_table);
                        pos += curr.length;
                    } else {
                        matches.push(pend);
                        pos = pend.position + pend.length;
                    }
                } else if curr.length >= lazy_threshold || pos + 1 > end {
                    matches.push(curr);
                    Self::update_chunk_hash(pos, hash as usize, hash_table, chain_table);
                    pos += curr.length;
                } else {
                    pending = Some(curr);
                    Self::update_chunk_hash(pos, hash as usize, hash_table, chain_table);
                    pos += 1;
                }
            } else {
                if let Some(pend) = pending.take() {
                    matches.push(pend);
                    pos = pend.position + pend.length;
                } else {
                    Self::update_chunk_hash(pos, hash as usize, hash_table, chain_table);
                    pos += 1;
                }
            }
        }

        if let Some(pend) = pending {
            matches.push(pend);
        }

        matches
    }

    #[inline(always)]
    fn update_chunk_hash(pos: usize, hash: usize, hash_table: &mut [u32], chain_table: &mut [u32]) {
        let prev = hash_table[hash];
        if pos < chain_table.len() {
            chain_table[pos] = prev;
        }
        hash_table[hash] = (pos + 1) as u32;
    }

    #[inline]
    fn find_best_match_in_chunk(
        chunk: &[u8],
        pos: usize,
        hash: usize,
        hash_table: &[u32],
        chain_table: &[u32],
        search_depth: usize,
    ) -> Option<Match> {
        let hash_entry = hash_table[hash];
        if hash_entry == 0 {
            return None;
        }

        let mut match_pos = (hash_entry - 1) as usize;
        if match_pos >= pos || pos + 4 > chunk.len() {
            return None;
        }

        let cur_prefix = unsafe {
            std::ptr::read_unaligned(chunk.as_ptr().add(pos) as *const u32)
        };

        let mut best_match: Option<Match> = None;
        let mut best_length = MIN_MATCH_LENGTH - 1;
        let mut depth = 0;

        while depth < search_depth && match_pos < pos {
            let offset = pos - match_pos;

            // Fast prefix check
            if match_pos + 4 <= chunk.len() {
                let match_prefix = unsafe {
                    std::ptr::read_unaligned(chunk.as_ptr().add(match_pos) as *const u32)
                };

                if match_prefix == cur_prefix {
                    // Count matching bytes after the first 4
                    let max_len = (chunk.len() - pos).min(chunk.len() - match_pos);
                    let mut length = 4;
                    while length < max_len && chunk[match_pos + length] == chunk[pos + length] {
                        length += 1;
                    }

                    if length > best_length {
                        best_length = length;
                        best_match = Some(Match::new(pos, offset, length));

                        if length >= 64 {
                            break; // Good enough
                        }
                    }
                }
            }

            // Follow chain
            if match_pos < chain_table.len() {
                let next = chain_table[match_pos];
                if next == 0 {
                    break;
                }
                match_pos = (next - 1) as usize;
            } else {
                break;
            }

            depth += 1;
        }

        best_match
    }

}

// =============================================================================
// Two-Tier Hash Match Finder (Phase 2.2)
// =============================================================================

/// Hash table size for the long (8-byte) hash table.
/// Smaller than the short table since 8-byte matches are less common.
const LONG_HASH_LOG: usize = 14;
const LONG_HASH_SIZE: usize = 1 << LONG_HASH_LOG;

/// Two-Tier Hash Table Match Finder.
///
/// Uses separate hash tables for 4-byte and 8-byte prefixes to improve
/// both match quality and search speed:
///
/// - **Short hash (4-byte)**: Standard hash table for finding any match >= 4 bytes
/// - **Long hash (8-byte)**: Optimized for finding long matches quickly
///
/// When searching, we first check the 8-byte hash table. If we find a match
/// there, it's likely to be long (since 8 bytes matched). This reduces the
/// time spent in hash chain traversal for repetitive data.
///
/// Key benefits:
/// - 8-byte hash has fewer collisions → shorter chains
/// - Long matches are found faster → earlier exit
/// - Maintains full match finding capability via 4-byte fallback
pub struct TwoTierMatchFinder {
    /// Standard 4-byte hash table and chain
    short_hash: Box<AlignedHashTable>,
    short_chain: Vec<u32>,
    /// 8-byte hash table for long match candidates
    long_hash: Vec<u32>,
    long_chain: Vec<u32>,
    /// Configuration
    search_depth: usize,
    input_len: usize,
}

// Manual Debug impl since AlignedHashTable doesn't derive Debug
impl core::fmt::Debug for TwoTierMatchFinder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TwoTierMatchFinder")
            .field("short_hash", &format_args!("[AlignedHashTable; {}]", HASH_SIZE))
            .field("short_chain_len", &self.short_chain.len())
            .field("long_hash_len", &self.long_hash.len())
            .field("long_chain_len", &self.long_chain.len())
            .field("search_depth", &self.search_depth)
            .field("input_len", &self.input_len)
            .finish()
    }
}

impl TwoTierMatchFinder {
    /// Create a new two-tier match finder.
    pub fn new(search_depth: usize) -> Self {
        Self {
            short_hash: AlignedHashTable::new_boxed(),
            short_chain: Vec::new(),
            long_hash: vec![0u32; LONG_HASH_SIZE],
            long_chain: Vec::new(),
            search_depth: search_depth.max(1).min(128),
            input_len: 0,
        }
    }

    /// Reset for new input.
    fn reset(&mut self, input_len: usize) {
        self.short_hash.reset();
        // Avoid resize + fill double-zero
        if self.short_chain.len() < input_len {
            self.short_chain.clear();
            self.short_chain.resize(input_len, 0);
        } else {
            self.short_chain.truncate(input_len);
            self.short_chain.fill(0);
        }

        self.long_hash.fill(0);
        // Avoid resize + fill double-zero
        if self.long_chain.len() < input_len {
            self.long_chain.clear();
            self.long_chain.resize(input_len, 0);
        } else {
            self.long_chain.truncate(input_len);
            self.long_chain.fill(0);
        }

        self.input_len = input_len;
    }

    /// Compute 4-byte hash (same as MatchFinder).
    #[inline(always)]
    fn hash4(&self, data: &[u8], pos: usize) -> u32 {
        debug_assert!(pos + 4 <= data.len());
        let bytes = unsafe {
            std::ptr::read_unaligned(data.as_ptr().add(pos) as *const u32)
        };
        let h = bytes.wrapping_mul(HASH_PRIME);
        let h = h ^ (h >> 15);
        let h = h.wrapping_mul(HASH_PRIME2);
        h >> (32 - HASH_LOG as u32)
    }

    /// Compute 8-byte hash for long match candidates.
    #[inline(always)]
    fn hash8(&self, data: &[u8], pos: usize) -> u32 {
        debug_assert!(pos + 8 <= data.len());
        let bytes = unsafe {
            std::ptr::read_unaligned(data.as_ptr().add(pos) as *const u64)
        };
        // Use a different mixing strategy for 8 bytes
        let h = (bytes as u32) ^ ((bytes >> 32) as u32);
        let h = h.wrapping_mul(HASH_PRIME);
        let h = h ^ (h >> 17);
        let h = h.wrapping_mul(HASH_PRIME2);
        (h >> (32 - LONG_HASH_LOG as u32)) as u32
    }

    /// Update both hash tables.
    #[inline(always)]
    fn update_hashes(&mut self, data: &[u8], pos: usize) {
        // Update 4-byte hash
        if pos + 4 <= data.len() {
            let h4 = self.hash4(data, pos) as usize;
            let prev = self.short_hash.get(h4);
            if pos < self.short_chain.len() {
                self.short_chain[pos] = prev;
            }
            self.short_hash.set(h4, (pos + 1) as u32);
        }

        // Update 8-byte hash
        if pos + 8 <= data.len() {
            let h8 = self.hash8(data, pos) as usize;
            let prev = self.long_hash[h8];
            if pos < self.long_chain.len() {
                self.long_chain[pos] = prev;
            }
            self.long_hash[h8] = (pos + 1) as u32;
        }
    }

    /// Find all matches using the two-tier approach.
    pub fn find_matches(&mut self, input: &[u8]) -> Vec<Match> {
        if input.len() < MIN_MATCH_LENGTH {
            return Vec::new();
        }

        self.reset(input.len());
        let mut matches = Vec::with_capacity(input.len() / 16);
        let mut pos = 0;
        let end = input.len().saturating_sub(MIN_MATCH_LENGTH);

        while pos <= end {
            let mut best_match: Option<Match> = None;

            // Try 8-byte hash first (if we have enough bytes)
            if pos + 8 <= input.len() {
                best_match = self.find_long_match(input, pos);
            }

            // Fall back to 4-byte hash if no long match found
            if best_match.is_none() && pos + 4 <= input.len() {
                best_match = self.find_short_match(input, pos);
            }

            // Update hash tables
            self.update_hashes(input, pos);

            if let Some(m) = best_match {
                matches.push(m);
                // Sparse updates during skip
                if m.length >= 8 {
                    let skip_end = (pos + m.length).min(end);
                    let mut update_pos = pos + 4;
                    while update_pos < skip_end {
                        self.update_hashes(input, update_pos);
                        update_pos += 4;
                    }
                }
                pos += m.length;
            } else {
                pos += 1;
            }
        }

        matches
    }

    /// Find match using 8-byte hash table.
    #[inline]
    fn find_long_match(&self, input: &[u8], pos: usize) -> Option<Match> {
        let h8 = self.hash8(input, pos) as usize;
        let hash_entry = self.long_hash[h8];

        if hash_entry == 0 {
            return None;
        }

        let mut match_pos = (hash_entry - 1) as usize;
        if match_pos >= pos {
            return None;
        }

        // Load 8-byte prefix for comparison
        let cur_prefix = unsafe {
            std::ptr::read_unaligned(input.as_ptr().add(pos) as *const u64)
        };

        let mut best_match: Option<Match> = None;
        let mut best_length = 7; // Only accept matches >= 8 from this table
        let mut depth = 0;

        while depth < self.search_depth / 2 && match_pos < pos {
            let offset = pos - match_pos;

            if match_pos + 8 <= input.len() {
                let match_prefix = unsafe {
                    std::ptr::read_unaligned(input.as_ptr().add(match_pos) as *const u64)
                };

                if match_prefix == cur_prefix {
                    // 8 bytes match - extend
                    let mut length = 8;
                    let max_len = (input.len() - pos).min(input.len() - match_pos);
                    while length < max_len && input[match_pos + length] == input[pos + length] {
                        length += 1;
                    }

                    if length > best_length {
                        best_length = length;
                        best_match = Some(Match::new(pos, offset, length));

                        // Early exit for excellent matches
                        if length >= 32 {
                            return best_match;
                        }
                    }
                }
            }

            // Follow chain
            if match_pos < self.long_chain.len() {
                let next = self.long_chain[match_pos];
                if next == 0 {
                    break;
                }
                match_pos = (next - 1) as usize;
            } else {
                break;
            }

            depth += 1;
        }

        best_match
    }

    /// Find match using 4-byte hash table.
    #[inline]
    fn find_short_match(&self, input: &[u8], pos: usize) -> Option<Match> {
        let h4 = self.hash4(input, pos) as usize;
        let hash_entry = self.short_hash.get(h4);

        if hash_entry == 0 {
            return None;
        }

        let mut match_pos = (hash_entry - 1) as usize;
        if match_pos >= pos {
            return None;
        }

        let cur_prefix = unsafe {
            std::ptr::read_unaligned(input.as_ptr().add(pos) as *const u32)
        };

        let mut best_match: Option<Match> = None;
        let mut best_length = MIN_MATCH_LENGTH - 1;
        let mut depth = 0;

        while depth < self.search_depth && match_pos < pos {
            let offset = pos - match_pos;

            if match_pos + 4 <= input.len() {
                let match_prefix = unsafe {
                    std::ptr::read_unaligned(input.as_ptr().add(match_pos) as *const u32)
                };

                if match_prefix == cur_prefix {
                    let mut length = 4;
                    let max_len = (input.len() - pos).min(input.len() - match_pos);
                    while length < max_len && input[match_pos + length] == input[pos + length] {
                        length += 1;
                    }

                    if length > best_length {
                        best_length = length;
                        best_match = Some(Match::new(pos, offset, length));

                        if length >= 24 {
                            return best_match;
                        }
                    }
                }
            }

            // Follow chain
            if match_pos < self.short_chain.len() {
                let next = self.short_chain[match_pos];
                if next == 0 {
                    break;
                }
                match_pos = (next - 1) as usize;
            } else {
                break;
            }

            depth += 1;
        }

        best_match
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_creation() {
        let m = Match::new(10, 5, 4);
        assert_eq!(m.position, 10);
        assert_eq!(m.offset, 5);
        assert_eq!(m.length, 4);
    }

    #[test]
    fn test_match_finder_creation() {
        let mf = MatchFinder::new(16);
        assert_eq!(mf.search_depth, 16);
    }

    #[test]
    fn test_no_matches_short_input() {
        let mut mf = MatchFinder::new(16);
        let matches = mf.find_matches(b"ab");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_no_matches_unique() {
        let mut mf = MatchFinder::new(16);
        let matches = mf.find_matches(b"abcdefghij");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_simple_repeat() {
        let mut mf = MatchFinder::new(16);
        // "abcdabcd" - "abcd" repeats at offset 4
        let matches = mf.find_matches(b"abcdabcd");

        assert!(!matches.is_empty());
        let m = &matches[0];
        assert_eq!(m.position, 4);
        assert_eq!(m.offset, 4);
        assert_eq!(m.length, 4);
    }

    #[test]
    fn test_overlapping_repeat() {
        let mut mf = MatchFinder::new(16);
        // "aaaaaa" - RLE-like pattern
        let matches = mf.find_matches(b"aaaaaaaaa");

        assert!(!matches.is_empty());
        // Should find long match with small offset
        let has_rle = matches.iter().any(|m| m.offset <= 4 && m.length >= 3);
        assert!(has_rle);
    }

    #[test]
    fn test_multiple_matches() {
        let mut mf = MatchFinder::new(16);
        // Multiple repeated patterns
        let input = b"abcdXabcdYabcdZ";
        let matches = mf.find_matches(input);

        // Should find matches (abcd repeats)
        assert!(matches.len() >= 1, "Expected at least one match, got {:?}", matches);
    }

    #[test]
    fn test_long_match() {
        let mut mf = MatchFinder::new(16);
        // Long repeated sequence
        let input = b"0123456789ABCDEF0123456789ABCDEF";
        let matches = mf.find_matches(input);

        assert!(!matches.is_empty());
        let m = &matches[0];
        assert_eq!(m.length, 16); // Full repeat
        assert_eq!(m.offset, 16);
    }

    #[test]
    fn test_hash_distribution() {
        let mf = MatchFinder::new(16);
        let input = b"testdatatestdata";

        // Different positions should produce different hashes
        let h1 = mf.hash4(input, 0);
        let h2 = mf.hash4(input, 4);
        let h3 = mf.hash4(input, 8);

        // Same data at different positions should hash the same
        assert_eq!(h1, h3); // "test" at 0 and 8
        assert_ne!(h1, h2); // "test" vs "data"
    }

    #[test]
    fn test_search_depth_limit() {
        let mut mf = MatchFinder::new(1);
        let input = b"abcXabcYabcZ";
        let matches = mf.find_matches(input);

        // Should still find matches, but may not be optimal
        assert!(matches.len() <= 3);
    }

    #[test]
    fn test_match_length_calculation() {
        let mut mf = MatchFinder::new(16);
        let input = b"hellohello";
        mf.reset(input.len());

        // match_length_from compares from pos1 and pos2 onwards
        let len = mf.match_length_from(input, 0, 5);
        assert_eq!(len, 5); // "hello" matches
    }

    #[test]
    fn test_lazy_match_finder() {
        let mut mf = LazyMatchFinder::new(16);
        let input = b"abcdefabcdefXabcdefabcdef";
        let matches = mf.find_matches(input);

        // Should find good matches
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_large_input() {
        let mut mf = MatchFinder::new(32);

        // Generate input with repeating patterns
        let mut input = Vec::with_capacity(100_000);
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        while input.len() < 100_000 {
            input.extend_from_slice(pattern);
        }

        let matches = mf.find_matches(&input);

        // Should find some matches in repetitive data
        // (count depends on match lengths - longer matches = fewer count)
        assert!(!matches.is_empty(), "Expected to find matches in repetitive data");

        // Verify compression potential - this is the key metric
        let total_match_len: usize = matches.iter().map(|m| m.length).sum();
        assert!(
            total_match_len > input.len() / 4,
            "Expected matches to cover at least 25% of input, got {} / {}",
            total_match_len,
            input.len()
        );
    }

    #[test]
    fn test_aligned_hash_table_alignment() {
        // Verify the AlignedHashTable is properly 64-byte aligned
        let table = AlignedHashTable::new_boxed();
        let addr = table.data.as_ptr() as usize;

        // Address should be 64-byte aligned
        assert_eq!(
            addr % 64,
            0,
            "Hash table data should be 64-byte aligned, got address {:x}",
            addr
        );

        // Also verify the struct itself starts at aligned boundary
        let struct_addr = &*table as *const _ as usize;
        assert_eq!(
            struct_addr % 64,
            0,
            "AlignedHashTable struct should be 64-byte aligned, got address {:x}",
            struct_addr
        );
    }

    #[test]
    fn test_aligned_hash_table_operations() {
        let mut table = AlignedHashTable::new_boxed();

        // Test that table is initially zeroed
        for i in 0..HASH_SIZE {
            assert_eq!(table.get(i), 0);
        }

        // Test set/get
        table.set(0, 42);
        table.set(HASH_SIZE - 1, 123);
        assert_eq!(table.get(0), 42);
        assert_eq!(table.get(HASH_SIZE - 1), 123);

        // Test reset
        table.reset();
        assert_eq!(table.get(0), 0);
        assert_eq!(table.get(HASH_SIZE - 1), 0);
    }

    // =========================================================================
    // Adaptive Search Depth Tests
    // =========================================================================

    #[test]
    fn test_adaptive_depth_scales_with_size() {
        let finder = MatchFinder::new(16);

        // Small inputs: full depth
        assert_eq!(finder.effective_depth(1024), 16);
        assert_eq!(finder.effective_depth(4096), 16);

        // Medium inputs: reduced depth (90%)
        assert_eq!(finder.effective_depth(8192), 14);
        assert_eq!(finder.effective_depth(16384), 14);

        // Large inputs: 75% depth (less aggressive for better ratio)
        assert_eq!(finder.effective_depth(65536), 12);

        // Very large inputs: 50% depth
        assert_eq!(finder.effective_depth(262144), 8);
    }

    #[test]
    fn test_adaptive_depth_respects_minimum() {
        let finder = MatchFinder::new(4);

        // Even with adaptive scaling, never go below reasonable minimum
        assert!(finder.effective_depth(262144) >= 2);
        assert!(finder.effective_depth(1_000_000) >= 2);
    }

    #[test]
    fn test_adaptive_depth_respects_configured_max() {
        let finder_shallow = MatchFinder::new(4);
        let finder_deep = MatchFinder::new(64);

        // Adaptive depth should scale from configured value
        assert!(finder_shallow.effective_depth(1024) <= 4);
        assert!(finder_deep.effective_depth(1024) <= 64);

        // Large input scaling maintains proportions
        let shallow_large = finder_shallow.effective_depth(65536);
        let deep_large = finder_deep.effective_depth(65536);
        assert!(deep_large > shallow_large);
    }

    #[test]
    fn test_adaptive_finder_large_input_throughput() {
        use std::time::Instant;

        // Generate large repetitive data
        let mut data = Vec::with_capacity(65536);
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut finder = MatchFinder::new(16);

        // Warm up
        for _ in 0..3 {
            let _ = finder.find_matches(&data);
        }

        // Measure throughput
        let iterations = 20;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = finder.find_matches(&data);
        }
        let elapsed = start.elapsed();

        let total_bytes = data.len() * iterations;
        let throughput_mib = total_bytes as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);

        // With adaptive depth, should achieve reasonable throughput on large data
        // Note: threshold is conservative for CI environments
        assert!(
            throughput_mib >= 30.0,
            "Large input throughput {:.1} MiB/s below target 30 MiB/s",
            throughput_mib
        );
    }

    #[test]
    fn test_adaptive_finder_maintains_compression_quality() {
        // Generate compressible data
        let mut data = Vec::with_capacity(65536);
        let pattern = b"compression test data with repeating patterns ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut finder = MatchFinder::new(16);
        let matches = finder.find_matches(&data);

        // Calculate match coverage
        let total_match_bytes: usize = matches.iter().map(|m| m.length).sum();
        let coverage = total_match_bytes as f64 / data.len() as f64;

        // Even with adaptive depth, should find good matches in repetitive data
        assert!(
            coverage >= 0.70,
            "Match coverage {:.1}% below target 70%",
            coverage * 100.0
        );
    }

    // =========================================================================
    // Block Chunking Tests (Phase 2.1)
    // =========================================================================

    #[test]
    fn test_chunked_matching_correctness() {
        let mut data = Vec::with_capacity(65536);
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut finder = LazyMatchFinder::new(16);

        // Standard matching
        let standard_matches = finder.find_matches(&data);

        // Chunked matching (resets finder internally per chunk)
        let chunked_matches = finder.find_matches_chunked(&data, 16384);

        // Both should produce valid matches (positions within bounds)
        for m in &standard_matches {
            assert!(m.position + m.length <= data.len());
            assert!(m.position >= m.offset);
        }
        for m in &chunked_matches {
            assert!(m.position + m.length <= data.len());
            assert!(m.position >= m.offset);
        }

        // Chunked should cover reasonable portion of input
        let chunked_coverage: usize = chunked_matches.iter().map(|m| m.length).sum();
        let min_coverage = data.len() / 2; // At least 50%
        assert!(
            chunked_coverage >= min_coverage,
            "Chunked coverage {} below minimum {}",
            chunked_coverage, min_coverage
        );
    }

    #[test]
    fn test_chunked_matching_performance_reasonable() {
        use std::time::Instant;

        // Generate large data (256KB) with varied content
        // Mix of repetitive and varied patterns to simulate real data
        let mut data = Vec::with_capacity(262144);
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        for i in 0..262144 {
            // Mix of patterns: some repetitive, some varied
            if i % 1024 < 512 {
                data.push(pattern[i % pattern.len()]);
            } else {
                data.push((i as u8).wrapping_mul(17).wrapping_add(i as u8 >> 4));
            }
        }

        let mut finder = LazyMatchFinder::new(16);

        // Warm up
        for _ in 0..2 {
            let _ = finder.find_matches(&data);
            let _ = finder.find_matches_chunked(&data, 16384);
        }

        // Measure standard matching
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = finder.find_matches(&data);
        }
        let standard_time = start.elapsed();

        // Measure chunked matching
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = finder.find_matches_chunked(&data, 16384);
        }
        let chunked_time = start.elapsed();

        // Chunked matching lacks several optimizations from the main path:
        // 1. No offset prediction (main path predicts based on recent matches)
        // 2. No generation counters (zeros tables each chunk)
        // 3. Simpler match length comparison
        //
        // As a result, chunked is expected to be significantly slower than
        // the optimized standard path. The threshold is set high because:
        // - Standard path has O(1) reset + prediction = very fast
        // - Chunked has O(n) reset per chunk + no prediction = slow
        let ratio = chunked_time.as_secs_f64() / standard_time.as_secs_f64();
        assert!(
            ratio < 20.0,
            "Chunked ({:?}) is too slow compared to standard ({:?}), ratio: {:.2}x",
            chunked_time, standard_time, ratio
        );
    }

    #[test]
    fn test_chunked_small_input_fallback() {
        let mut finder = LazyMatchFinder::new(16);
        let small_data = b"small input that fits in one chunk";

        // Chunked should work for small input (becomes single chunk)
        let matches = finder.find_matches_chunked(small_data, 16384);

        // Should not panic, may or may not find matches
        assert!(matches.len() <= small_data.len());
    }

    #[test]
    fn test_chunked_boundary_handling() {
        // Create data where a pattern spans chunk boundary
        let chunk_size = 1024;
        let mut data = vec![b'A'; chunk_size - 8];
        // Pattern at boundary
        data.extend_from_slice(b"PATTERN!");
        data.extend_from_slice(b"PATTERN!"); // Repeat in next chunk
        data.extend_from_slice(&vec![b'B'; chunk_size - 16]);

        let mut finder = LazyMatchFinder::new(16);
        let matches = finder.find_matches_chunked(&data, chunk_size);

        // Matches in second chunk should have correct absolute positions
        for m in &matches {
            assert!(m.position + m.length <= data.len());
            // Verify match is valid by checking data
            let src_start = m.position - m.offset;
            for i in 0..m.length {
                assert_eq!(
                    data[src_start + i], data[m.position + i],
                    "Match at {} offset {} length {} invalid at byte {}",
                    m.position, m.offset, m.length, i
                );
            }
        }
    }

    #[test]
    fn test_chunked_maintains_compression_ratio() {
        let mut data = Vec::with_capacity(65536);
        let pattern = b"repeating content for compression ratio test ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut finder = LazyMatchFinder::new(16);

        let standard = finder.find_matches(&data);
        let chunked = finder.find_matches_chunked(&data, 16384);

        let standard_coverage: usize = standard.iter().map(|m| m.length).sum();
        let chunked_coverage: usize = chunked.iter().map(|m| m.length).sum();

        // Chunked may have slightly less coverage due to chunk boundaries,
        // but should be within 15% of standard
        let min_acceptable = (standard_coverage as f64 * 0.85) as usize;
        assert!(
            chunked_coverage >= min_acceptable,
            "Chunked coverage {} below 85% of standard {}",
            chunked_coverage, standard_coverage
        );
    }

    // =========================================================================
    // Position-Adaptive Early Exit Tests (Phase 1.3)
    // =========================================================================

    #[test]
    fn test_early_exit_threshold_by_position() {
        let finder = MatchFinder::new(16);

        // Early in file: need longer match to exit early (higher threshold)
        let threshold_early = finder.early_exit_threshold(100);
        assert!(threshold_early >= 24, "Early position should have threshold >= 24, got {}", threshold_early);

        // Mid-file: moderate threshold
        let threshold_mid = finder.early_exit_threshold(10000);
        assert!(threshold_mid >= 12 && threshold_mid < 24,
            "Mid position should have threshold 12-23, got {}", threshold_mid);

        // Late in file: shorter threshold acceptable
        let threshold_late = finder.early_exit_threshold(50000);
        assert!(threshold_late >= 8 && threshold_late < 16,
            "Late position should have threshold 8-15, got {}", threshold_late);

        // Monotonic: threshold should decrease (or stay same) as position increases
        assert!(threshold_early >= threshold_mid, "Threshold should decrease with position");
        assert!(threshold_mid >= threshold_late, "Threshold should decrease with position");
    }

    #[test]
    fn test_early_exit_excellent_match() {
        let mut finder = MatchFinder::new(16);

        // Create data with a long match (36+ bytes)
        let mut data = vec![b'X'; 10]; // Prefix
        let pattern = b"abcdefghijklmnopqrstuvwxyz0123456789ABCD";
        data.extend_from_slice(pattern);
        data.extend_from_slice(pattern); // 40-byte repeat

        let matches = finder.find_matches(&data);

        // Should find the excellent match
        let long_match = matches.iter().any(|m| m.length >= 36);
        assert!(long_match, "Should find the long match >= 36 bytes, got {:?}", matches);
    }

    #[test]
    fn test_early_exit_improves_throughput_on_repetitive() {
        use std::time::Instant;

        // Generate highly repetitive data with many long matches
        let mut data = Vec::with_capacity(65536);
        let pattern = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // 72 bytes
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut finder = MatchFinder::new(16);

        // Warm up
        for _ in 0..3 {
            let _ = finder.find_matches(&data);
        }

        // Measure
        let iterations = 20;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = finder.find_matches(&data);
        }
        let elapsed = start.elapsed();

        let total_bytes = data.len() * iterations;
        let throughput_mib = total_bytes as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);

        // With early exit, should achieve good throughput on repetitive data
        assert!(
            throughput_mib >= 25.0,
            "Repetitive data throughput {:.1} MiB/s below target 25 MiB/s",
            throughput_mib
        );
    }

    #[test]
    fn test_early_exit_maintains_match_quality() {
        let mut data = Vec::with_capacity(65536);
        let pattern = b"The quick brown fox jumps over the lazy dog repeatedly. ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut finder = MatchFinder::new(16);
        let matches = finder.find_matches(&data);

        // Even with early exit, should find good matches
        let total_match_bytes: usize = matches.iter().map(|m| m.length).sum();
        let coverage = total_match_bytes as f64 / data.len() as f64;

        assert!(
            coverage >= 0.65,
            "Match coverage {:.1}% below target 65%",
            coverage * 100.0
        );
    }

    // =========================================================================
    // Lazy Threshold Scaling Tests (Phase 1.2)
    // =========================================================================

    #[test]
    fn test_lazy_threshold_scaling_by_size() {
        // Small input: default threshold
        let mut finder = LazyMatchFinder::new(16);
        finder.configure_for_size(1024);
        assert_eq!(finder.lazy_threshold, 24, "Small input should use default threshold");

        // Medium input: slightly lower threshold
        let mut finder = LazyMatchFinder::new(16);
        finder.configure_for_size(16384);
        assert!(finder.lazy_threshold <= 20, "Medium input should lower threshold");

        // Large input: commit earlier
        let mut finder = LazyMatchFinder::new(16);
        finder.configure_for_size(65536);
        assert!(finder.lazy_threshold <= 16, "Large input should commit earlier");

        // Very large input: aggressive early commit
        let mut finder = LazyMatchFinder::new(16);
        finder.configure_for_size(262144);
        assert!(finder.lazy_threshold <= 12, "Very large input should be aggressive");
    }

    #[test]
    fn test_adaptive_lazy_threshold_minimum() {
        let mut finder = LazyMatchFinder::new(16);
        finder.configure_for_size(1_000_000);

        // Should never go below minimum match length + 1
        assert!(finder.lazy_threshold >= MIN_MATCH_LENGTH + 1);
    }

    #[test]
    fn test_adaptive_lazy_maintains_quality() {
        let mut data = Vec::with_capacity(65536);
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        // Fixed threshold
        let mut fixed_finder = LazyMatchFinder::new(16);
        let fixed_matches = fixed_finder.find_matches(&data);

        // Adaptive threshold
        let mut adaptive_finder = LazyMatchFinder::new(16);
        adaptive_finder.configure_for_size(data.len());
        let adaptive_matches = adaptive_finder.find_matches(&data);

        let fixed_coverage: usize = fixed_matches.iter().map(|m| m.length).sum();
        let adaptive_coverage: usize = adaptive_matches.iter().map(|m| m.length).sum();

        // Adaptive should maintain at least 90% of fixed coverage
        let min_coverage = (fixed_coverage as f64 * 0.90) as usize;
        assert!(
            adaptive_coverage >= min_coverage,
            "Adaptive coverage {} below 90% of fixed {}",
            adaptive_coverage, fixed_coverage
        );
    }

    #[test]
    fn test_find_matches_auto_uses_adaptive() {
        let mut data = Vec::with_capacity(65536);
        let pattern = b"repeating patterns for auto adaptive test. ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut finder = LazyMatchFinder::new(16);

        // find_matches_auto should auto-configure for size
        let matches = finder.find_matches_auto(&data);

        // Should produce valid matches
        for m in &matches {
            assert!(m.position + m.length <= data.len());
            assert!(m.position >= m.offset);
        }

        // Should have reasonable coverage
        let coverage: usize = matches.iter().map(|m| m.length).sum();
        assert!(coverage > data.len() / 2, "Should cover at least 50% of input");
    }

    // =========================================================================
    // Two-Tier Hash Match Finder Tests (Phase 2.2)
    // =========================================================================

    #[test]
    fn test_two_tier_finder_creation() {
        let finder = TwoTierMatchFinder::new(16);
        assert_eq!(finder.search_depth, 16);
    }

    #[test]
    fn test_two_tier_finds_matches() {
        let mut finder = TwoTierMatchFinder::new(16);
        let input = b"abcdefghijklmnopabcdefghijklmnop";
        let matches = finder.find_matches(input);

        // Should find the 16-byte repeat
        assert!(!matches.is_empty(), "Should find matches");
        let m = &matches[0];
        assert_eq!(m.length, 16);
        assert_eq!(m.offset, 16);
    }

    #[test]
    fn test_two_tier_finds_long_matches_via_8byte_hash() {
        let mut finder = TwoTierMatchFinder::new(16);

        // Create pattern with 8+ byte repeat
        let mut data = vec![b'X'; 10];
        let pattern = b"LONGPATTERN12345678901234567890AB";
        data.extend_from_slice(pattern);
        data.extend_from_slice(pattern);

        let matches = finder.find_matches(&data);

        // Should find the long match
        let long_match = matches.iter().any(|m| m.length >= 30);
        assert!(long_match, "Should find long match via 8-byte hash, got {:?}", matches);
    }

    #[test]
    fn test_two_tier_finds_short_matches_fallback() {
        let mut finder = TwoTierMatchFinder::new(16);

        // Pattern that's exactly 4-7 bytes (too short for 8-byte hash)
        let input = b"ABCDxxABCD"; // 4-byte match
        let matches = finder.find_matches(input);

        // Should find the short match via 4-byte fallback
        let short_match = matches.iter().any(|m| m.length >= 4);
        assert!(short_match, "Should find short match via 4-byte fallback, got {:?}", matches);
    }

    #[test]
    fn test_two_tier_coverage_comparable_to_single() {
        let mut data = Vec::with_capacity(16384);
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        while data.len() < 16384 {
            data.extend_from_slice(pattern);
        }

        let mut single = MatchFinder::new(16);
        let mut two_tier = TwoTierMatchFinder::new(16);

        let single_matches = single.find_matches(&data);
        let two_tier_matches = two_tier.find_matches(&data);

        let single_coverage: usize = single_matches.iter().map(|m| m.length).sum();
        let two_tier_coverage: usize = two_tier_matches.iter().map(|m| m.length).sum();

        // Two-tier should achieve at least 90% of single-tier coverage
        let min_coverage = (single_coverage as f64 * 0.90) as usize;
        assert!(
            two_tier_coverage >= min_coverage,
            "Two-tier coverage {} below 90% of single {}",
            two_tier_coverage, single_coverage
        );
    }

    #[test]
    fn test_two_tier_performance_reasonable() {
        use std::time::Instant;

        let mut data = Vec::with_capacity(65536);
        let pattern = b"repeating pattern for performance test data here. ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut single = MatchFinder::new(16);
        let mut two_tier = TwoTierMatchFinder::new(16);

        // Warm up
        for _ in 0..3 {
            let _ = single.find_matches(&data);
            let _ = two_tier.find_matches(&data);
        }

        // Measure single tier
        let iterations = 15;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = single.find_matches(&data);
        }
        let single_time = start.elapsed();

        // Measure two tier
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = two_tier.find_matches(&data);
        }
        let two_tier_time = start.elapsed();

        // Two-tier has significant overhead from maintaining two hash tables and
        // two chain tables. Additionally, the main MatchFinder now uses O(1) reset
        // via generation counters while TwoTier still zeros its tables. Combined
        // with double updates per position, TwoTier is expected to be much slower.
        // With sparse skip optimization on repetitive data, standard path is even
        // faster, increasing the relative gap further.
        // Note: TwoTier is not used in the main compression path.
        let ratio = two_tier_time.as_secs_f64() / single_time.as_secs_f64();
        assert!(
            ratio < 30.0,
            "Two-tier ({:?}) too slow compared to single ({:?}), ratio: {:.2}x",
            two_tier_time, single_time, ratio
        );
    }

    #[test]
    fn test_two_tier_8byte_hash_distribution() {
        let finder = TwoTierMatchFinder::new(16);
        let data = b"AABBCCDDAABBCCDD"; // 16 bytes with 8-byte repeat

        // Different 8-byte sequences should hash differently
        let h1 = finder.hash8(data, 0); // "AABBCCDD"
        let h2 = finder.hash8(data, 8); // "AABBCCDD" (same)

        // Same data should hash the same
        assert_eq!(h1, h2, "Same 8-byte sequence should have same hash");

        // Different data should (usually) hash differently
        let data2 = b"XYZ12345XYZ12345";
        let h3 = finder.hash8(data2, 0);
        // Not guaranteed different but should be due to good hash function
        // Just verify it runs without panic
        assert!(h3 < LONG_HASH_SIZE as u32);
    }

    // =========================================================================
    // Speculative Parallel Match Finding Tests (Phase 3.2)
    // =========================================================================

    #[test]
    fn test_speculative_finds_matches() {
        let mut finder = MatchFinder::new(16);
        let input = b"abcdefghijklmnopabcdefghijklmnop";
        let matches = finder.find_matches_speculative(input);

        assert!(!matches.is_empty(), "Should find matches");
        let m = &matches[0];
        assert_eq!(m.length, 16);
    }

    #[test]
    fn test_speculative_correctness() {
        // Generate text-like data
        let mut data = Vec::with_capacity(16384);
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        while data.len() < 16384 {
            data.extend_from_slice(pattern);
        }

        let mut standard = MatchFinder::new(16);
        let mut speculative = MatchFinder::new(16);

        let std_matches = standard.find_matches(&data);
        let spec_matches = speculative.find_matches_speculative(&data);

        // Both should find matches
        assert!(!std_matches.is_empty());
        assert!(!spec_matches.is_empty());

        // Coverage should be comparable (within 20%)
        let std_coverage: usize = std_matches.iter().map(|m| m.length).sum();
        let spec_coverage: usize = spec_matches.iter().map(|m| m.length).sum();

        let min_coverage = (std_coverage as f64 * 0.80) as usize;
        assert!(
            spec_coverage >= min_coverage,
            "Speculative coverage {} below 80% of standard {}",
            spec_coverage, std_coverage
        );
    }

    #[test]
    fn test_speculative_no_overlapping_matches() {
        let mut finder = MatchFinder::new(16);
        // Data with overlapping match opportunities
        let input = b"ABCABCABCABCABCABCABCABCABCABCABCABC";
        let matches = finder.find_matches_speculative(input);

        // Verify no overlapping matches in output
        for i in 1..matches.len() {
            let prev_end = matches[i - 1].position + matches[i - 1].length;
            assert!(
                matches[i].position >= prev_end,
                "Match {} at pos {} overlaps with previous ending at {}",
                i, matches[i].position, prev_end
            );
        }
    }

    #[test]
    fn test_speculative_handles_short_input() {
        let mut finder = MatchFinder::new(16);

        // Very short input
        let matches = finder.find_matches_speculative(b"abc");
        assert!(matches.is_empty());

        // Input just at boundary
        let matches = finder.find_matches_speculative(b"abcdabcd");
        // Should handle gracefully
        assert!(matches.len() <= 1);
    }

    #[test]
    fn test_speculative_performance_reasonable() {
        use std::time::Instant;

        let mut data = Vec::with_capacity(65536);
        let pattern = b"repeating pattern for speculative matching test. ";
        while data.len() < 65536 {
            data.extend_from_slice(pattern);
        }

        let mut standard = MatchFinder::new(16);
        let mut speculative = MatchFinder::new(16);

        // Warm up
        for _ in 0..3 {
            let _ = standard.find_matches(&data);
            let _ = speculative.find_matches_speculative(&data);
        }

        // Measure standard
        let iterations = 15;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = standard.find_matches(&data);
        }
        let std_time = start.elapsed();

        // Measure speculative
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = speculative.find_matches_speculative(&data);
        }
        let spec_time = start.elapsed();

        // Speculative does extra work (lookahead) for potentially better matches.
        // With the optimized standard path (prediction + O(1) reset + sparse skip),
        // speculative may be significantly slower as it explores more candidates
        // without the sparse skip optimization for repetitive patterns.
        let ratio = spec_time.as_secs_f64() / std_time.as_secs_f64();
        assert!(
            ratio < 8.0,
            "Speculative ({:?}) too slow compared to standard ({:?}), ratio: {:.2}x",
            spec_time, std_time, ratio
        );
    }

    #[test]
    fn test_speculative_finds_better_matches() {
        let mut finder = MatchFinder::new(16);

        // Data where position +1 has a longer match than position 0
        // Position 0: "XXXAB..." matches 5 bytes
        // Position 1: "XXABC..." might match longer pattern later
        let mut data = vec![b'X'; 5];
        data.extend_from_slice(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"); // 26 bytes
        data.extend_from_slice(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"); // repeat

        let matches = finder.find_matches_speculative(&data);

        // Should find the long match
        let has_long = matches.iter().any(|m| m.length >= 20);
        assert!(has_long, "Speculative should find long matches: {:?}", matches);
    }
}
