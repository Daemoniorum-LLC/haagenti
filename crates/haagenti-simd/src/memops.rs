//! Memory operations for compression.
//!
//! Optimized memory copy and fill operations used in LZ decompression.

/// Copy a match from earlier in the output buffer.
///
/// This handles overlapping copies correctly, which is essential for LZ77
/// decompression where the match source overlaps with the destination.
///
/// # Arguments
///
/// * `output` - Output buffer (destination is at end, source is earlier)
/// * `offset` - Distance back from current position to match start
/// * `length` - Number of bytes to copy
///
/// # Panics
///
/// Panics if offset is greater than output length or if there's not enough
/// space for the copy.
#[inline]
pub fn copy_match(output: &mut Vec<u8>, offset: usize, length: usize) {
    debug_assert!(offset > 0, "offset must be positive");
    debug_assert!(offset <= output.len(), "offset exceeds buffer");

    let start = output.len() - offset;

    // Handle overlapping copies (common in LZ77)
    if offset >= length {
        // Non-overlapping: can use efficient extend
        output.reserve(length);

        // Safety: we've reserved space and verified bounds
        unsafe {
            let ptr = output.as_ptr().add(start);
            let dst = output.as_mut_ptr().add(output.len());
            std::ptr::copy_nonoverlapping(ptr, dst, length);
            output.set_len(output.len() + length);
        }
    } else {
        // Overlapping: copy byte by byte (repeat pattern)
        output.reserve(length);

        // For short offsets, unroll the pattern
        if offset == 1 {
            // Run-length encoding: repeat single byte
            let byte = output[start];
            output.extend(std::iter::repeat_n(byte, length));
        } else if offset < 8 {
            // Small offset: copy pattern repeatedly
            for i in 0..length {
                let byte = output[start + (i % offset)];
                output.push(byte);
            }
        } else {
            // Medium offset: copy in chunks
            let mut remaining = length;
            while remaining > 0 {
                let copy_len = remaining.min(offset);
                let src_start = output.len() - offset;
                output.reserve(copy_len);

                unsafe {
                    let ptr = output.as_ptr().add(src_start);
                    let dst = output.as_mut_ptr().add(output.len());
                    std::ptr::copy_nonoverlapping(ptr, dst, copy_len);
                    output.set_len(output.len() + copy_len);
                }

                remaining -= copy_len;
            }
        }
    }
}

/// Fill buffer with a repeating pattern.
///
/// Used for run-length encoding where a pattern repeats many times.
///
/// # Arguments
///
/// * `output` - Output buffer to extend
/// * `pattern` - Pattern to repeat
/// * `count` - Number of times to repeat the pattern
#[inline]
pub fn fill_repeat(output: &mut Vec<u8>, pattern: &[u8], count: usize) {
    if pattern.is_empty() || count == 0 {
        return;
    }

    let total_len = pattern.len() * count;
    output.reserve(total_len);

    if pattern.len() == 1 {
        // Single byte: use extend with repeat iterator
        output.extend(std::iter::repeat_n(pattern[0], count));
    } else {
        // Multi-byte pattern: copy repeatedly
        for _ in 0..count {
            output.extend_from_slice(pattern);
        }
    }
}

/// Copy bytes from source to destination with potential overlap.
///
/// This is a building block for match copying that handles the case
/// where source and destination may overlap.
#[inline]
pub fn copy_within_extend(output: &mut Vec<u8>, src_start: usize, length: usize) {
    output.reserve(length);

    let offset = output.len() - src_start;

    if offset >= length {
        // Non-overlapping
        unsafe {
            let ptr = output.as_ptr().add(src_start);
            let dst = output.as_mut_ptr().add(output.len());
            std::ptr::copy_nonoverlapping(ptr, dst, length);
            output.set_len(output.len() + length);
        }
    } else {
        // Overlapping - copy in chunks
        let mut copied = 0;
        while copied < length {
            let chunk = (length - copied).min(offset);
            unsafe {
                let ptr = output.as_ptr().add(src_start + (copied % offset));
                let dst = output.as_mut_ptr().add(output.len());
                std::ptr::copy_nonoverlapping(ptr, dst, chunk);
                output.set_len(output.len() + chunk);
            }
            copied += chunk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_match_non_overlapping() {
        let mut output = vec![1, 2, 3, 4, 5];
        copy_match(&mut output, 5, 3);
        assert_eq!(output, vec![1, 2, 3, 4, 5, 1, 2, 3]);
    }

    #[test]
    fn test_copy_match_overlapping() {
        // Copy with small offset (pattern repeat)
        let mut output = vec![b'A', b'B', b'C'];
        copy_match(&mut output, 2, 6);
        assert_eq!(output, b"ABCBCBCBC");
    }

    #[test]
    fn test_copy_match_rle() {
        // offset=1 is run-length encoding
        let mut output = vec![b'X'];
        copy_match(&mut output, 1, 5);
        assert_eq!(output, b"XXXXXX");
    }

    #[test]
    fn test_copy_match_offset_3() {
        let mut output = vec![b'A', b'B', b'C'];
        copy_match(&mut output, 3, 9);
        assert_eq!(output, b"ABCABCABCABC");
    }

    #[test]
    fn test_fill_repeat_single() {
        let mut output = Vec::new();
        fill_repeat(&mut output, b"X", 5);
        assert_eq!(output, b"XXXXX");
    }

    #[test]
    fn test_fill_repeat_pattern() {
        let mut output = Vec::new();
        fill_repeat(&mut output, b"AB", 4);
        assert_eq!(output, b"ABABABAB");
    }

    #[test]
    fn test_fill_repeat_empty() {
        let mut output = Vec::new();
        fill_repeat(&mut output, b"", 5);
        assert!(output.is_empty());

        fill_repeat(&mut output, b"X", 0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_copy_match_medium_offset() {
        let mut output: Vec<u8> = (0..20).collect();
        let original_len = output.len();
        copy_match(&mut output, 10, 25);

        // First 10 should be a copy of bytes 10-19
        assert_eq!(
            &output[original_len..original_len + 10],
            &(10..20).collect::<Vec<u8>>()
        );
        // Pattern continues
        assert_eq!(output.len(), original_len + 25);
    }

    #[test]
    fn test_copy_within_extend_non_overlapping() {
        let mut output = vec![1, 2, 3, 4, 5];
        copy_within_extend(&mut output, 0, 3);
        assert_eq!(output, vec![1, 2, 3, 4, 5, 1, 2, 3]);
    }

    #[test]
    fn test_copy_within_extend_overlapping() {
        let mut output = vec![1, 2, 3];
        copy_within_extend(&mut output, 1, 5);
        // Copies from position 1 (value=2), with pattern length 2
        assert_eq!(output, vec![1, 2, 3, 2, 3, 2, 3, 2]);
    }
}
