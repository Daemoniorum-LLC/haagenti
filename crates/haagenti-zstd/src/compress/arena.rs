//! Arena allocator for per-frame temporary allocations.
//!
//! This module provides a simple bump allocator that reduces allocation overhead
//! during compression by reusing memory between frames.
//!
//! ## Benefits
//!
//! - **Reduced allocation overhead**: Single allocation per reset
//! - **Cache-friendly**: Sequential memory access
//! - **Fast reset**: O(1) reset between frames
//!
//! ## Usage
//!
//! ```ignore
//! let mut arena = Arena::new(64 * 1024);  // 64KB arena
//!
//! // Allocate temporary buffers
//! let literals = arena.alloc_slice(1024);
//! let sequences = arena.alloc_slice(256);
//!
//! // Reset for next frame (O(1) operation)
//! arena.reset();
//! ```

use core::cell::Cell;

/// Default arena size (64KB) - covers most per-frame allocations.
pub const DEFAULT_ARENA_SIZE: usize = 64 * 1024;

/// A simple bump allocator for temporary allocations.
///
/// The arena pre-allocates a contiguous block of memory and hands out
/// slices from it via bump pointer allocation. When the frame is complete,
/// the arena is reset (O(1)) and the memory is reused.
#[derive(Debug)]
pub struct Arena {
    /// The underlying memory buffer.
    buffer: Vec<u8>,
    /// Current allocation position (bump pointer).
    /// Uses Cell for interior mutability to allow allocation from &self.
    pos: Cell<usize>,
    /// Peak usage tracking for diagnostics.
    peak_usage: Cell<usize>,
}

impl Arena {
    /// Create a new arena with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0; capacity],
            pos: Cell::new(0),
            peak_usage: Cell::new(0),
        }
    }

    /// Create a new arena with the default size (64KB).
    pub fn with_default_size() -> Self {
        Self::new(DEFAULT_ARENA_SIZE)
    }

    /// Reset the arena for reuse.
    ///
    /// This is an O(1) operation - it just resets the bump pointer.
    /// The memory contents are not cleared.
    #[inline]
    pub fn reset(&self) {
        // Track peak usage before reset
        let current = self.pos.get();
        if current > self.peak_usage.get() {
            self.peak_usage.set(current);
        }
        self.pos.set(0);
    }

    /// Allocate a mutable slice of bytes.
    ///
    /// Returns `None` if there's not enough space in the arena.
    /// SAFETY: This returns a mutable reference from a shared reference, which is safe
    /// because we use interior mutability (Cell) to track the allocation position,
    /// ensuring each region is only handed out once.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice(&self, len: usize) -> Option<&mut [u8]> {
        let pos = self.pos.get();
        let new_pos = pos.checked_add(len)?;

        if new_pos > self.buffer.len() {
            return None;
        }

        self.pos.set(new_pos);

        // SAFETY: We're the only ones with access to this region
        // because we just bumped the position past it.
        unsafe {
            let ptr = self.buffer.as_ptr().add(pos) as *mut u8;
            Some(core::slice::from_raw_parts_mut(ptr, len))
        }
    }

    /// Allocate a mutable slice of bytes, zeroed.
    ///
    /// Returns `None` if there's not enough space in the arena.
    #[inline]
    pub fn alloc_slice_zeroed(&self, len: usize) -> Option<&mut [u8]> {
        let slice = self.alloc_slice(len)?;
        slice.fill(0);
        Some(slice)
    }

    /// Allocate a Vec-like buffer backed by the arena.
    ///
    /// Returns an ArenaVec that can grow up to the remaining capacity.
    pub fn alloc_vec(&self, initial_capacity: usize) -> Option<ArenaVec<'_>> {
        let pos = self.pos.get();
        let max_capacity = self.buffer.len().saturating_sub(pos);

        if initial_capacity > max_capacity {
            return None;
        }

        // Reserve the initial capacity
        self.pos.set(pos + initial_capacity);

        Some(ArenaVec {
            arena: self,
            start: pos,
            len: 0,
            capacity: initial_capacity,
        })
    }

    /// Get the current usage of the arena.
    #[inline]
    pub fn usage(&self) -> usize {
        self.pos.get()
    }

    /// Get the total capacity of the arena.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get the remaining capacity.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buffer.len().saturating_sub(self.pos.get())
    }

    /// Get the peak usage (highest watermark since creation).
    #[inline]
    pub fn peak_usage(&self) -> usize {
        self.peak_usage.get().max(self.pos.get())
    }
}

/// A vector-like structure backed by arena memory.
///
/// This provides push/extend operations while using arena memory.
/// When dropped, the arena space is not reclaimed (arena is bump-only).
pub struct ArenaVec<'a> {
    arena: &'a Arena,
    start: usize,
    len: usize,
    capacity: usize,
}

impl<'a> ArenaVec<'a> {
    /// Push a byte to the vector.
    ///
    /// Returns `false` if the vector is at capacity and cannot grow.
    #[inline]
    pub fn push(&mut self, value: u8) -> bool {
        if self.len >= self.capacity {
            // Try to grow
            if !self.grow(1) {
                return false;
            }
        }

        // SAFETY: We have exclusive access to this region via the arena
        unsafe {
            let ptr = self.arena.buffer.as_ptr().add(self.start + self.len) as *mut u8;
            *ptr = value;
        }
        self.len += 1;
        true
    }

    /// Extend the vector with bytes from a slice.
    ///
    /// Returns `false` if there's not enough space.
    pub fn extend_from_slice(&mut self, data: &[u8]) -> bool {
        if self.len + data.len() > self.capacity {
            // Try to grow
            if !self.grow(data.len()) {
                return false;
            }
        }

        // SAFETY: We have exclusive access to this region via the arena
        unsafe {
            let ptr = self.arena.buffer.as_ptr().add(self.start + self.len) as *mut u8;
            core::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        self.len += data.len();
        true
    }

    /// Try to grow the capacity.
    fn grow(&mut self, additional: usize) -> bool {
        let needed = self.len + additional;
        if needed <= self.capacity {
            return true;
        }

        // Check if we're at the end of the arena and can extend
        let arena_pos = self.arena.pos.get();
        let our_end = self.start + self.capacity;

        if arena_pos == our_end {
            // We're at the end, can grow in place
            let new_capacity = (needed * 2).min(self.arena.remaining() + self.capacity);
            if new_capacity >= needed {
                let growth = new_capacity - self.capacity;
                self.arena.pos.set(arena_pos + growth);
                self.capacity = new_capacity;
                return true;
            }
        }

        false
    }

    /// Get the length of the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a slice of the vector contents.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: We have valid data from start to start+len
        unsafe {
            let ptr = self.arena.buffer.as_ptr().add(self.start);
            core::slice::from_raw_parts(ptr, self.len)
        }
    }

    /// Convert to a regular Vec (copies the data).
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = Arena::new(1024);
        assert_eq!(arena.capacity(), 1024);
        assert_eq!(arena.usage(), 0);
        assert_eq!(arena.remaining(), 1024);
    }

    #[test]
    fn test_arena_alloc_slice() {
        let arena = Arena::new(1024);

        let slice1 = arena.alloc_slice(100).unwrap();
        assert_eq!(slice1.len(), 100);
        assert_eq!(arena.usage(), 100);

        let slice2 = arena.alloc_slice(200).unwrap();
        assert_eq!(slice2.len(), 200);
        assert_eq!(arena.usage(), 300);
    }

    #[test]
    fn test_arena_alloc_slice_zeroed() {
        let arena = Arena::new(1024);

        // First allocate and write some data
        let slice = arena.alloc_slice(100).unwrap();
        slice.fill(0xFF);

        arena.reset();

        // Now allocate zeroed - should be zeroed
        let zeroed = arena.alloc_slice_zeroed(100).unwrap();
        for &byte in zeroed.iter() {
            assert_eq!(byte, 0);
        }
    }

    #[test]
    fn test_arena_reset() {
        let arena = Arena::new(1024);

        let _ = arena.alloc_slice(500).unwrap();
        assert_eq!(arena.usage(), 500);

        arena.reset();
        assert_eq!(arena.usage(), 0);
        assert_eq!(arena.peak_usage(), 500);
    }

    #[test]
    fn test_arena_overflow() {
        let arena = Arena::new(100);

        assert!(arena.alloc_slice(50).is_some());
        assert!(arena.alloc_slice(60).is_none()); // Would overflow
        assert_eq!(arena.usage(), 50); // Usage unchanged on failure
    }

    #[test]
    fn test_arena_vec_basic() {
        let arena = Arena::new(1024);

        let mut vec = arena.alloc_vec(100).unwrap();
        assert!(vec.is_empty());

        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_arena_vec_extend() {
        let arena = Arena::new(1024);

        let mut vec = arena.alloc_vec(100).unwrap();
        vec.extend_from_slice(b"Hello, ");
        vec.extend_from_slice(b"World!");

        assert_eq!(vec.as_slice(), b"Hello, World!");
    }

    #[test]
    fn test_arena_vec_grow() {
        let arena = Arena::new(1024);

        let mut vec = arena.alloc_vec(10).unwrap();

        // Fill past initial capacity
        for i in 0..50u8 {
            assert!(vec.push(i));
        }

        assert_eq!(vec.len(), 50);

        // Verify contents
        for (i, &b) in vec.as_slice().iter().enumerate() {
            assert_eq!(b, i as u8);
        }
    }

    #[test]
    fn test_arena_vec_to_vec() {
        let arena = Arena::new(1024);

        let mut arena_vec = arena.alloc_vec(100).unwrap();
        arena_vec.extend_from_slice(b"Test data");

        let regular_vec = arena_vec.to_vec();
        assert_eq!(regular_vec, b"Test data");
    }

    #[test]
    fn test_arena_peak_usage() {
        let arena = Arena::new(1024);

        // Allocate some memory
        let _ = arena.alloc_slice(400);
        arena.reset();

        // Allocate less
        let _ = arena.alloc_slice(200);

        // Peak should still be 400
        assert_eq!(arena.peak_usage(), 400);
    }

    #[test]
    fn test_default_arena_size() {
        let arena = Arena::with_default_size();
        assert_eq!(arena.capacity(), DEFAULT_ARENA_SIZE);
    }
}
