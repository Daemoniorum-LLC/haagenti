//! Repeat Offset-Aware Match Finding
//!
//! This module enhances match finding by exploiting zstd's repeat offset mechanism.
//! Repeat offsets (rep0/rep1/rep2) encode with far fewer bits than new offsets:
//!
//! | Offset Type | FSE Code | Extra Bits | Total Bits |
//! |-------------|----------|------------|------------|
//! | rep0        | ~1 bit   | 0          | ~1 bit     |
//! | rep1        | ~1 bit   | 0          | ~1 bit     |
//! | rep2        | ~1 bit   | 0          | ~1 bit     |
//! | new (100)   | ~4 bits  | 6 bits     | ~10 bits   |
//!
//! A match at a repeat offset saves ~8 bits compared to a new offset.
//! This means a 4-byte match at rep0 is as good as a 5-byte match at a new offset!
//!
//! ## Strategy
//!
//! 1. **Proactive Probing**: Before hash chain search, probe all 3 repeat offsets
//! 2. **Bonus Scoring**: Give repeat offset matches a length bonus
//! 3. **State Tracking**: Keep repeat offsets synchronized with encoder

use super::match_finder::{LazyMatchFinder, Match, MAX_MATCH_LENGTH, MIN_MATCH_LENGTH};

/// Bonus (in bytes) for matches at repeat offsets.
/// A match at rep0 saves ~8 bits = 1 byte, so a bonus of 1-2 is appropriate.
const REP_OFFSET_BONUS: usize = 2;

/// Match finder with repeat offset awareness.
///
/// Tracks the same repeat offset state as the encoder and proactively searches
/// for matches at repeat offsets before falling back to hash chain search.
#[derive(Debug)]
pub struct RepeatOffsetMatchFinder {
    /// Inner match finder for hash chain search
    inner: LazyMatchFinder,
    /// Current repeat offsets [rep0, rep1, rep2]
    /// Initialized to [1, 4, 8] per RFC 8878
    rep_offsets: [usize; 3],
}

impl RepeatOffsetMatchFinder {
    /// Create a new repeat offset-aware match finder.
    pub fn new(search_depth: usize) -> Self {
        Self {
            inner: LazyMatchFinder::new(search_depth),
            rep_offsets: [1, 4, 8], // RFC 8878 initial values
        }
    }

    /// Reset repeat offsets to initial state.
    fn reset_rep_offsets(&mut self) {
        self.rep_offsets = [1, 4, 8];
    }

    /// Update repeat offsets after using an offset.
    ///
    /// This must match the encoder's RepeatOffsetsEncoder logic exactly
    /// to keep state synchronized.
    fn update_rep_offsets(&mut self, actual_offset: usize, literal_length: usize) {
        if literal_length > 0 {
            // Normal case
            if actual_offset == self.rep_offsets[0] {
                // rep0 - no change
            } else if actual_offset == self.rep_offsets[1] {
                // rep1 -> rotate to front
                self.rep_offsets.swap(1, 0);
            } else if actual_offset == self.rep_offsets[2] {
                // rep2 -> rotate to front
                let temp = self.rep_offsets[2];
                self.rep_offsets[2] = self.rep_offsets[1];
                self.rep_offsets[1] = self.rep_offsets[0];
                self.rep_offsets[0] = temp;
            } else {
                // New offset -> push to front
                self.rep_offsets[2] = self.rep_offsets[1];
                self.rep_offsets[1] = self.rep_offsets[0];
                self.rep_offsets[0] = actual_offset;
            }
        } else {
            // LL = 0: special case
            if actual_offset == self.rep_offsets[1] {
                self.rep_offsets.swap(0, 1);
            } else if actual_offset == self.rep_offsets[2] {
                let temp = self.rep_offsets[2];
                self.rep_offsets[2] = self.rep_offsets[1];
                self.rep_offsets[1] = self.rep_offsets[0];
                self.rep_offsets[0] = temp;
            } else if actual_offset == self.rep_offsets[0].saturating_sub(1).max(1) {
                let new_offset = self.rep_offsets[0].saturating_sub(1).max(1);
                self.rep_offsets[2] = self.rep_offsets[1];
                self.rep_offsets[1] = self.rep_offsets[0];
                self.rep_offsets[0] = new_offset;
            } else {
                // New offset
                self.rep_offsets[2] = self.rep_offsets[1];
                self.rep_offsets[1] = self.rep_offsets[0];
                self.rep_offsets[0] = actual_offset;
            }
        }
    }

    /// Check if offset is a repeat offset and return its index (0, 1, 2) or None.
    #[inline]
    fn rep_offset_index(&self, offset: usize) -> Option<usize> {
        if offset == self.rep_offsets[0] {
            Some(0)
        } else if offset == self.rep_offsets[1] {
            Some(1)
        } else if offset == self.rep_offsets[2] {
            Some(2)
        } else {
            None
        }
    }

    /// Probe for a match at a specific offset.
    ///
    /// Returns the match length if a valid match exists, 0 otherwise.
    #[inline]
    fn probe_offset(&self, input: &[u8], pos: usize, offset: usize) -> usize {
        if offset == 0 || pos < offset {
            return 0;
        }

        let match_pos = pos - offset;
        let remaining = input.len() - pos;
        let max_len = remaining.min(MAX_MATCH_LENGTH);

        if max_len < MIN_MATCH_LENGTH {
            return 0;
        }

        // Quick 4-byte prefix check
        if pos + 4 <= input.len() && match_pos + 4 <= input.len() {
            let cur = unsafe { std::ptr::read_unaligned(input.as_ptr().add(pos) as *const u32) };
            let prev =
                unsafe { std::ptr::read_unaligned(input.as_ptr().add(match_pos) as *const u32) };

            if cur != prev {
                return 0;
            }

            // Extend match
            let mut len = 4;
            while len < max_len && input[match_pos + len] == input[pos + len] {
                len += 1;
            }
            len
        } else {
            // Byte-by-byte for edge cases
            let mut len = 0;
            while len < max_len && input[match_pos + len] == input[pos + len] {
                len += 1;
            }
            if len >= MIN_MATCH_LENGTH {
                len
            } else {
                0
            }
        }
    }

    /// Find the best match considering repeat offsets.
    ///
    /// Probes all 3 repeat offsets first, then falls back to hash chain search.
    /// Returns the match with the best "effective length" (length + bonus for rep offsets).
    fn find_best_match_with_rep(
        &mut self,
        input: &[u8],
        pos: usize,
        _literal_length: usize,
    ) -> Option<Match> {
        let mut best_match: Option<Match> = None;
        let mut best_score: usize = 0;

        // Probe repeat offsets first (cheap - no hash lookup)
        for (rep_idx, &rep_offset) in self.rep_offsets.iter().enumerate() {
            let len = self.probe_offset(input, pos, rep_offset);
            if len >= MIN_MATCH_LENGTH {
                // Score = length + bonus for repeat offset
                // rep0 gets slightly more bonus than rep1/rep2
                let bonus = REP_OFFSET_BONUS + (2 - rep_idx);
                let score = len + bonus;

                if score > best_score {
                    best_score = score;
                    best_match = Some(Match::new(pos, rep_offset, len));
                }
            }
        }

        // If we found a really good repeat offset match, skip hash search
        if best_score >= MIN_MATCH_LENGTH + REP_OFFSET_BONUS + 8 {
            return best_match;
        }

        // Fall back to hash chain search via inner match finder
        // We need to check if hash search finds something better
        let hash = if pos + 4 <= input.len() {
            self.inner.inner.hash4(input, pos)
        } else if pos + 3 <= input.len() {
            self.inner.inner.hash3(input, pos)
        } else {
            return best_match;
        };

        if let Some(hash_match) = self.inner.inner.find_best_match(input, pos, hash as usize) {
            // Score without repeat bonus
            let _hash_score = hash_match.length;

            // Check if hash match is at a repeat offset (gets bonus retroactively)
            let hash_is_rep = self.rep_offset_index(hash_match.offset);
            let hash_score = if hash_is_rep.is_some() {
                hash_match.length + REP_OFFSET_BONUS
            } else {
                hash_match.length
            };

            if hash_score > best_score {
                best_match = Some(hash_match);
            }
        }

        best_match
    }

    /// Find all matches in the input with repeat offset awareness.
    pub fn find_matches(&mut self, input: &[u8]) -> Vec<Match> {
        if input.len() < MIN_MATCH_LENGTH {
            return Vec::new();
        }

        self.reset_rep_offsets();
        self.inner.inner.reset(input.len());
        self.inner.configure_for_size(input.len());

        let mut matches = Vec::with_capacity(input.len() / 16);
        let mut pos = 0;
        let end = input.len().saturating_sub(MIN_MATCH_LENGTH);
        let mut literal_run = 0usize; // Track literals since last match

        while pos <= end {
            if let Some(m) = self.find_best_match_with_rep(input, pos, literal_run) {
                matches.push(m);

                // Update repeat offsets (must match encoder)
                self.update_rep_offsets(m.offset, literal_run);

                // Update hash table for skipped positions
                let skip_end = (pos + m.length).min(end);
                for update_pos in pos..skip_end.min(pos + 8) {
                    if update_pos + 4 <= input.len() {
                        let h = self.inner.inner.hash4(input, update_pos);
                        self.inner.inner.update_hash(input, update_pos, h as usize);
                    }
                }

                pos += m.length;
                literal_run = 0;
            } else {
                // No match - update hash and advance
                if pos + 4 <= input.len() {
                    let h = self.inner.inner.hash4(input, pos);
                    self.inner.inner.update_hash(input, pos, h as usize);
                }
                pos += 1;
                literal_run += 1;
            }
        }

        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_offset_finder_basic() {
        let mut finder = RepeatOffsetMatchFinder::new(16);

        // Pattern with obvious repeat at offset 4
        let input = b"abcdabcdabcdabcd";
        let matches = finder.find_matches(input);

        // Should find matches
        assert!(!matches.is_empty());

        // First match should be at offset 4 (which is rep1 initially!)
        if let Some(m) = matches.first() {
            assert_eq!(m.offset, 4);
        }
    }

    #[test]
    fn test_repeat_offset_tracking() {
        let mut finder = RepeatOffsetMatchFinder::new(16);

        // Check initial state
        assert_eq!(finder.rep_offsets, [1, 4, 8]);

        // Simulate using offset 100 with LL > 0
        finder.update_rep_offsets(100, 5);
        assert_eq!(finder.rep_offsets, [100, 1, 4]);

        // Use offset 100 again (should stay at front)
        finder.update_rep_offsets(100, 3);
        assert_eq!(finder.rep_offsets, [100, 1, 4]);

        // Use offset 1 (was rep1, should rotate to front)
        finder.update_rep_offsets(1, 2);
        assert_eq!(finder.rep_offsets, [1, 100, 4]);
    }

    #[test]
    fn test_rep_offset_bonus() {
        let mut finder = RepeatOffsetMatchFinder::new(16);

        // Long pattern that can match at offset 1 (rep0)
        let input = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let matches = finder.find_matches(input);

        // Should find a match at offset 1 (rep0)
        if let Some(m) = matches.first() {
            assert_eq!(m.offset, 1);
            assert!(m.length > 3);
        }
    }
}
