//! Streaming compression and decompression utilities.

/// Flush modes for streaming compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Flush {
    /// No flush - buffer data for optimal compression.
    #[default]
    None,

    /// Sync flush - emit all pending output, remain compressible.
    /// Use for: periodic checkpoints, network packets.
    Sync,

    /// Full flush - emit all pending output, reset state.
    /// Use for: seeking support, error recovery.
    Full,

    /// Block flush - complete current block only.
    /// Use for: block-level parallelism.
    Block,

    /// Finish - complete stream with trailer.
    /// Use for: end of stream.
    Finish,
}

/// Configuration for stream buffers.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Input buffer size (default: 64 KB).
    pub input_buffer_size: usize,

    /// Output buffer size (default: 64 KB).
    pub output_buffer_size: usize,

    /// Maximum memory usage (default: 8 MB).
    pub max_memory: usize,

    /// Enable checksum verification.
    pub verify_checksum: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            input_buffer_size: 65536,
            output_buffer_size: 65536,
            max_memory: 8 * 1024 * 1024,
            verify_checksum: true,
        }
    }
}

/// Stream state for tracking progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamState {
    /// Stream not started.
    #[default]
    Initial,
    /// Stream in progress.
    Active,
    /// Stream finished successfully.
    Finished,
    /// Stream encountered error.
    Error,
}

impl StreamState {
    /// Check if stream is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, StreamState::Finished | StreamState::Error)
    }

    /// Check if stream can accept more input.
    pub fn can_write(self) -> bool {
        matches!(self, StreamState::Initial | StreamState::Active)
    }
}
