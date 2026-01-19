//! Error types for streaming generation

use thiserror::Error;

/// Errors that can occur during streaming
#[derive(Debug, Error)]
pub enum StreamError {
    /// Stream cancelled by user
    #[error("Stream cancelled: {0}")]
    Cancelled(String),

    /// Stream timeout
    #[error("Stream timeout after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    /// Decode error
    #[error("Decode failed: {0}")]
    DecodeError(String),

    /// Invalid frame
    #[error("Invalid frame at step {step}: {reason}")]
    InvalidFrame { step: u32, reason: String },

    /// Channel error
    #[error("Channel error: {0}")]
    ChannelError(String),

    /// Protocol error
    #[error("Protocol error: {0}")]
    ProtocolError(String),

    /// Stream already finished
    #[error("Stream already finished")]
    StreamFinished,

    /// Resource unavailable
    #[error("Resource unavailable: {0}")]
    ResourceUnavailable(String),
}

/// Result type for streaming operations
pub type Result<T> = std::result::Result<T, StreamError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancelled_error_display() {
        let err = StreamError::Cancelled("user requested".to_string());
        assert!(err.to_string().contains("cancelled"));
        assert!(err.to_string().contains("user requested"));
    }

    #[test]
    fn test_timeout_error_display() {
        let err = StreamError::Timeout { duration_ms: 5000 };
        let msg = err.to_string();
        assert!(msg.contains("timeout"));
        assert!(msg.contains("5000ms"));
    }

    #[test]
    fn test_decode_error_display() {
        let err = StreamError::DecodeError("invalid latent".to_string());
        assert!(err.to_string().contains("Decode failed"));
        assert!(err.to_string().contains("invalid latent"));
    }

    #[test]
    fn test_invalid_frame_error_display() {
        let err = StreamError::InvalidFrame {
            step: 42,
            reason: "corrupted data".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Invalid frame"));
        assert!(msg.contains("step 42"));
        assert!(msg.contains("corrupted data"));
    }

    #[test]
    fn test_channel_error_display() {
        let err = StreamError::ChannelError("receiver dropped".to_string());
        assert!(err.to_string().contains("Channel error"));
        assert!(err.to_string().contains("receiver dropped"));
    }

    #[test]
    fn test_protocol_error_display() {
        let err = StreamError::ProtocolError("invalid message type".to_string());
        assert!(err.to_string().contains("Protocol error"));
        assert!(err.to_string().contains("invalid message type"));
    }

    #[test]
    fn test_stream_finished_error_display() {
        let err = StreamError::StreamFinished;
        assert!(err.to_string().contains("already finished"));
    }

    #[test]
    fn test_resource_unavailable_error_display() {
        let err = StreamError::ResourceUnavailable("GPU memory exhausted".to_string());
        assert!(err.to_string().contains("Resource unavailable"));
        assert!(err.to_string().contains("GPU memory exhausted"));
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_error() -> Result<()> {
            Err(StreamError::StreamFinished)
        }

        let result = returns_error();
        assert!(result.is_err());
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StreamError>();
    }
}
