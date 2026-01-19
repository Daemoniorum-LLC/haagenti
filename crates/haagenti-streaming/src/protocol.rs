//! Streaming protocol for WebSocket/HTTP communication

use crate::{PreviewFrame, PreviewQuality, StreamState};
use serde::{Deserialize, Serialize};

/// Message type for streaming protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Stream started
    Started,
    /// Preview frame available
    Preview,
    /// Progress update
    Progress,
    /// Stream paused
    Paused,
    /// Stream resumed
    Resumed,
    /// Stream completed
    Completed,
    /// Stream cancelled
    Cancelled,
    /// Error occurred
    Error,
    /// Control command
    Control,
    /// Heartbeat/ping
    Heartbeat,
}

/// A message in the streaming protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMessage {
    /// Message type
    pub msg_type: MessageType,
    /// Sequence number
    pub sequence: u64,
    /// Timestamp (ms since epoch)
    pub timestamp: u64,
    /// Payload (serialized content)
    pub payload: MessagePayload,
}

/// Message payload variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Empty payload
    Empty,
    /// Started message with config
    Started {
        total_steps: u32,
        width: u32,
        height: u32,
        model_id: String,
    },
    /// Preview frame data
    Preview {
        step: u32,
        total_steps: u32,
        quality: PreviewQuality,
        data_format: DataFormat,
        data: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Progress update
    Progress {
        step: u32,
        total_steps: u32,
        progress_percent: f32,
        estimated_remaining_ms: u64,
    },
    /// Error message
    Error { code: String, message: String },
    /// Control command
    Control { command: String, args: Vec<String> },
}

/// Data format for preview frames
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataFormat {
    /// Raw RGBA bytes
    RawRGBA,
    /// JPEG compressed
    Jpeg,
    /// PNG compressed
    Png,
    /// Base64 encoded JPEG
    Base64Jpeg,
    /// WebP compressed
    WebP,
}

impl StreamMessage {
    /// Create a new message
    pub fn new(msg_type: MessageType, sequence: u64, payload: MessagePayload) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            msg_type,
            sequence,
            timestamp,
            payload,
        }
    }

    /// Create started message
    pub fn started(sequence: u64, total_steps: u32, width: u32, height: u32, model_id: String) -> Self {
        Self::new(
            MessageType::Started,
            sequence,
            MessagePayload::Started {
                total_steps,
                width,
                height,
                model_id,
            },
        )
    }

    /// Create preview message
    pub fn preview(
        sequence: u64,
        step: u32,
        total_steps: u32,
        quality: PreviewQuality,
        format: DataFormat,
        data: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Self {
        Self::new(
            MessageType::Preview,
            sequence,
            MessagePayload::Preview {
                step,
                total_steps,
                quality,
                data_format: format,
                data,
                width,
                height,
            },
        )
    }

    /// Create progress message
    pub fn progress(sequence: u64, step: u32, total_steps: u32, estimated_remaining_ms: u64) -> Self {
        let progress_percent = if total_steps > 0 {
            step as f32 / total_steps as f32 * 100.0
        } else {
            0.0
        };

        Self::new(
            MessageType::Progress,
            sequence,
            MessagePayload::Progress {
                step,
                total_steps,
                progress_percent,
                estimated_remaining_ms,
            },
        )
    }

    /// Create completed message
    pub fn completed(sequence: u64) -> Self {
        Self::new(MessageType::Completed, sequence, MessagePayload::Empty)
    }

    /// Create error message
    pub fn error(sequence: u64, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(
            MessageType::Error,
            sequence,
            MessagePayload::Error {
                code: code.into(),
                message: message.into(),
            },
        )
    }

    /// Create heartbeat message
    pub fn heartbeat(sequence: u64) -> Self {
        Self::new(MessageType::Heartbeat, sequence, MessagePayload::Empty)
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to binary (bincode)
    pub fn to_binary(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from binary
    pub fn from_binary(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

/// Protocol handler for stream communication
#[derive(Debug)]
pub struct StreamProtocol {
    /// Current sequence number
    sequence: u64,
    /// Binary mode (vs JSON)
    binary_mode: bool,
    /// Preferred data format
    data_format: DataFormat,
}

impl StreamProtocol {
    /// Create a new protocol handler
    pub fn new() -> Self {
        Self {
            sequence: 0,
            binary_mode: false,
            data_format: DataFormat::Jpeg,
        }
    }

    /// Set binary mode
    pub fn with_binary_mode(mut self, binary: bool) -> Self {
        self.binary_mode = binary;
        self
    }

    /// Set data format
    pub fn with_data_format(mut self, format: DataFormat) -> Self {
        self.data_format = format;
        self
    }

    /// Get next sequence number
    fn next_sequence(&mut self) -> u64 {
        let seq = self.sequence;
        self.sequence += 1;
        seq
    }

    /// Create started message
    pub fn started(&mut self, total_steps: u32, width: u32, height: u32, model_id: &str) -> StreamMessage {
        StreamMessage::started(
            self.next_sequence(),
            total_steps,
            width,
            height,
            model_id.to_string(),
        )
    }

    /// Create preview message from frame
    pub fn preview(&mut self, frame: &PreviewFrame, data: Vec<u8>) -> StreamMessage {
        StreamMessage::preview(
            self.next_sequence(),
            frame.step,
            frame.total_steps,
            frame.quality,
            self.data_format,
            data,
            frame.width,
            frame.height,
        )
    }

    /// Create progress message
    pub fn progress(&mut self, step: u32, total_steps: u32, estimated_remaining_ms: u64) -> StreamMessage {
        StreamMessage::progress(self.next_sequence(), step, total_steps, estimated_remaining_ms)
    }

    /// Create completed message
    pub fn completed(&mut self) -> StreamMessage {
        StreamMessage::completed(self.next_sequence())
    }

    /// Create error message
    pub fn error(&mut self, code: &str, message: &str) -> StreamMessage {
        StreamMessage::error(self.next_sequence(), code, message)
    }

    /// Encode message
    pub fn encode(&self, message: &StreamMessage) -> Vec<u8> {
        if self.binary_mode {
            message.to_binary()
        } else {
            message.to_json().unwrap_or_default().into_bytes()
        }
    }

    /// Decode message
    pub fn decode(&self, data: &[u8]) -> Result<StreamMessage, String> {
        if self.binary_mode {
            StreamMessage::from_binary(data).map_err(|e| e.to_string())
        } else {
            let json = std::str::from_utf8(data).map_err(|e| e.to_string())?;
            StreamMessage::from_json(json).map_err(|e| e.to_string())
        }
    }

    /// Current data format
    pub fn data_format(&self) -> DataFormat {
        self.data_format
    }
}

impl Default for StreamProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = StreamMessage::started(1, 20, 512, 512, "sdxl".to_string());

        assert_eq!(msg.msg_type, MessageType::Started);
        assert_eq!(msg.sequence, 1);

        if let MessagePayload::Started { total_steps, .. } = msg.payload {
            assert_eq!(total_steps, 20);
        } else {
            panic!("Wrong payload type");
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let msg = StreamMessage::progress(5, 10, 20, 5000);

        let json = msg.to_json().unwrap();
        let restored = StreamMessage::from_json(&json).unwrap();

        assert_eq!(msg.sequence, restored.sequence);
        assert_eq!(msg.msg_type, restored.msg_type);
    }

    #[test]
    fn test_binary_roundtrip() {
        let msg = StreamMessage::error(1, "TEST_ERR", "Test error message");

        let binary = msg.to_binary();
        let restored = StreamMessage::from_binary(&binary).unwrap();

        assert_eq!(msg.sequence, restored.sequence);

        if let MessagePayload::Error { code, .. } = restored.payload {
            assert_eq!(code, "TEST_ERR");
        }
    }

    #[test]
    fn test_protocol_sequence() {
        let mut protocol = StreamProtocol::new();

        let msg1 = protocol.progress(1, 20, 1000);
        let msg2 = protocol.progress(2, 20, 900);
        let msg3 = protocol.completed();

        assert_eq!(msg1.sequence, 0);
        assert_eq!(msg2.sequence, 1);
        assert_eq!(msg3.sequence, 2);
    }
}
