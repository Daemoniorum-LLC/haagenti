//! Real-Time Streaming Generation
//!
//! Display progressively improving images during generation with
//! streaming preview and mid-generation control.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                 Real-Time Streaming                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Timeline:                                                       │
//! │  ════════════════════════════════════════════════════════════   │
//! │  t=0.0s  t=0.5s  t=1.0s  t=1.5s  t=2.0s  t=2.5s  t=3.0s        │
//! │     │       │       │       │       │       │       │           │
//! │     ↓       ↓       ↓       ↓       ↓       ↓       ↓           │
//! │  [Start] [Blob]  [Shape] [Color] [Detail][Sharp] [Final]       │
//! │   30%     50%     70%     85%     95%    100%    100%           │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │              Streaming Pipeline                           │   │
//! │  │                                                           │   │
//! │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
//! │  │  │ Denoise │->│ Decode  │->│ Upscale │->│ Display │     │   │
//! │  │  │ Step N  │  │ Latent  │  │ Preview │  │ Canvas  │     │   │
//! │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │   │
//! │  │       ↓                                                   │   │
//! │  │  Continue to Step N+1 (parallel with display)            │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Preview Modes
//!
//! | Mode | Preview Frequency | Overhead | UX |
//! |------|-------------------|----------|-----|
//! | Instant | Every step | 30% | Smooth but slow |
//! | Balanced | Every 5 steps | 10% | Good compromise |
//! | Fast | Steps 1, 10, 20 | 5% | Minimal previews |
//! | Thumbnail | 64x64 every step | 2% | Tiny but instant |

mod adaptive;
mod controller;
mod decoder;
mod error;
mod preview;
mod protocol;
mod scheduler;
mod stream;

pub use adaptive::{AdaptiveStreamManager, NetworkConditions, QualityPolicy, RecommendedQuality};
pub use controller::{StreamController, ControlCommand, ControlResponse};
pub use decoder::{StreamDecoder, DecoderConfig, DecodedFrame};
pub use error::{StreamError, Result};
pub use preview::{PreviewConfig, PreviewData, PreviewFrame, PreviewQuality};
pub use protocol::{StreamMessage, StreamProtocol, MessageType};
pub use scheduler::{PreviewScheduler, ScheduleMode, PreviewEvent};
pub use stream::{GenerationStream, StreamConfig, StreamState};

/// Default preview frequency (every N steps)
pub const DEFAULT_PREVIEW_INTERVAL: u32 = 5;

/// Default thumbnail size
pub const DEFAULT_THUMBNAIL_SIZE: u32 = 64;

/// Maximum preview latency target in ms
pub const MAX_PREVIEW_LATENCY_MS: u32 = 100;

/// Prelude for common imports
pub mod prelude {
    pub use super::{
        GenerationStream, PreviewFrame, PreviewScheduler, ScheduleMode,
        StreamConfig, StreamController, Result,
    };
}
