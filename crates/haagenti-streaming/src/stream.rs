//! Generation stream management

use crate::{
    PreviewConfig, PreviewFrame, PreviewQuality, PreviewScheduler, Result, ScheduleMode,
    StreamError,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

/// Configuration for generation stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Total denoising steps
    pub total_steps: u32,
    /// Preview schedule mode
    pub schedule_mode: ScheduleMode,
    /// Preview configuration
    pub preview_config: PreviewConfig,
    /// Buffer size for preview frames
    pub buffer_size: usize,
    /// Enable frame skipping under load
    pub allow_frame_skip: bool,
    /// Maximum concurrent decodes
    pub max_concurrent_decodes: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            total_steps: 20,
            schedule_mode: ScheduleMode::default(),
            preview_config: PreviewConfig::default(),
            buffer_size: 10,
            allow_frame_skip: true,
            max_concurrent_decodes: 2,
        }
    }
}

/// State of the generation stream
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamState {
    /// Stream not started
    Idle,
    /// Generation in progress
    Running,
    /// Paused (can resume)
    Paused,
    /// Cancelled by user
    Cancelled,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed,
}

/// Generation stream handle
#[derive(Debug)]
pub struct GenerationStream {
    /// Stream configuration
    config: StreamConfig,
    /// Current state
    state: Arc<std::sync::RwLock<StreamState>>,
    /// Current step
    current_step: Arc<AtomicU32>,
    /// Cancel flag
    cancelled: Arc<AtomicBool>,
    /// Preview scheduler
    scheduler: PreviewScheduler,
    /// Frame sender
    frame_tx: mpsc::Sender<PreviewFrame>,
    /// Frame receiver
    frame_rx: Option<mpsc::Receiver<PreviewFrame>>,
    /// State broadcaster
    state_tx: watch::Sender<StreamState>,
    /// State receiver
    state_rx: watch::Receiver<StreamState>,
}

impl GenerationStream {
    /// Create a new generation stream
    pub fn new(config: StreamConfig) -> Self {
        let scheduler = PreviewScheduler::new(config.schedule_mode, config.total_steps);
        let (frame_tx, frame_rx) = mpsc::channel(config.buffer_size);
        let (state_tx, state_rx) = watch::channel(StreamState::Idle);

        Self {
            config,
            state: Arc::new(std::sync::RwLock::new(StreamState::Idle)),
            current_step: Arc::new(AtomicU32::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
            scheduler,
            frame_tx,
            frame_rx: Some(frame_rx),
            state_tx,
            state_rx,
        }
    }

    /// Start the stream
    pub fn start(&self) -> Result<()> {
        self.set_state(StreamState::Running);
        self.cancelled.store(false, Ordering::SeqCst);
        self.current_step.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Pause the stream
    pub fn pause(&self) -> Result<()> {
        if self.get_state() != StreamState::Running {
            return Err(StreamError::StreamFinished);
        }
        self.set_state(StreamState::Paused);
        Ok(())
    }

    /// Resume the stream
    pub fn resume(&self) -> Result<()> {
        if self.get_state() != StreamState::Paused {
            return Err(StreamError::StreamFinished);
        }
        self.set_state(StreamState::Running);
        Ok(())
    }

    /// Cancel the stream
    pub fn cancel(&self) -> Result<()> {
        self.cancelled.store(true, Ordering::SeqCst);
        self.set_state(StreamState::Cancelled);
        Ok(())
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Get current state
    pub fn get_state(&self) -> StreamState {
        *self.state.read().unwrap()
    }

    /// Set state
    fn set_state(&self, state: StreamState) {
        *self.state.write().unwrap() = state;
        let _ = self.state_tx.send(state);
    }

    /// Get current step
    pub fn current_step(&self) -> u32 {
        self.current_step.load(Ordering::SeqCst)
    }

    /// Advance to next step
    pub fn advance_step(&self) -> u32 {
        let step = self.current_step.fetch_add(1, Ordering::SeqCst) + 1;
        if step >= self.config.total_steps {
            self.set_state(StreamState::Completed);
        }
        step
    }

    /// Submit a preview frame
    pub async fn submit_frame(&self, frame: PreviewFrame) -> Result<()> {
        if self.is_cancelled() {
            return Err(StreamError::Cancelled("Stream cancelled".into()));
        }

        self.frame_tx
            .send(frame)
            .await
            .map_err(|_| StreamError::ChannelError("Frame channel closed".into()))
    }

    /// Take the frame receiver (can only be called once)
    pub fn take_frame_receiver(&mut self) -> Option<mpsc::Receiver<PreviewFrame>> {
        self.frame_rx.take()
    }

    /// Subscribe to state changes
    pub fn subscribe_state(&self) -> watch::Receiver<StreamState> {
        self.state_rx.clone()
    }

    /// Get progress percentage
    pub fn progress(&self) -> f32 {
        let step = self.current_step.load(Ordering::SeqCst);
        if self.config.total_steps > 0 {
            (step as f32 / self.config.total_steps as f32 * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Check if preview should be generated for current step
    pub fn should_preview(&self) -> bool {
        let step = self.current_step.load(Ordering::SeqCst);
        self.scheduler.should_preview(step)
    }

    /// Get quality for current step
    pub fn current_quality(&self) -> PreviewQuality {
        let step = self.current_step.load(Ordering::SeqCst);
        self.scheduler.quality_for_step(step)
    }

    /// Get scheduler reference
    pub fn scheduler(&self) -> &PreviewScheduler {
        &self.scheduler
    }

    /// Get config reference
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Create a clonable handle
    pub fn handle(&self) -> StreamHandle {
        StreamHandle {
            state: Arc::clone(&self.state),
            current_step: Arc::clone(&self.current_step),
            cancelled: Arc::clone(&self.cancelled),
            total_steps: self.config.total_steps,
        }
    }
}

/// Clonable handle for stream status
#[derive(Debug, Clone)]
pub struct StreamHandle {
    state: Arc<std::sync::RwLock<StreamState>>,
    current_step: Arc<AtomicU32>,
    cancelled: Arc<AtomicBool>,
    total_steps: u32,
}

impl StreamHandle {
    /// Get current state
    pub fn state(&self) -> StreamState {
        *self.state.read().unwrap()
    }

    /// Get current step
    pub fn step(&self) -> u32 {
        self.current_step.load(Ordering::SeqCst)
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Get progress
    pub fn progress(&self) -> f32 {
        if self.total_steps > 0 {
            self.step() as f32 / self.total_steps as f32 * 100.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preview::{PreviewData, PreviewFrame, PreviewQuality};

    #[tokio::test]
    async fn test_stream_lifecycle() {
        let stream = GenerationStream::new(StreamConfig::default());

        assert_eq!(stream.get_state(), StreamState::Idle);

        stream.start().unwrap();
        assert_eq!(stream.get_state(), StreamState::Running);

        stream.pause().unwrap();
        assert_eq!(stream.get_state(), StreamState::Paused);

        stream.resume().unwrap();
        assert_eq!(stream.get_state(), StreamState::Running);

        stream.cancel().unwrap();
        assert_eq!(stream.get_state(), StreamState::Cancelled);
        assert!(stream.is_cancelled());
    }

    #[test]
    fn test_step_advancement() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 5,
            ..Default::default()
        });

        stream.start().unwrap();

        assert_eq!(stream.current_step(), 0);
        assert_eq!(stream.advance_step(), 1);
        assert_eq!(stream.advance_step(), 2);
        assert_eq!(stream.current_step(), 2);
    }

    #[test]
    fn test_progress() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 10,
            ..Default::default()
        });

        stream.start().unwrap();
        assert_eq!(stream.progress(), 0.0);

        stream.advance_step(); // 1
        stream.advance_step(); // 2
        stream.advance_step(); // 3
        stream.advance_step(); // 4
        stream.advance_step(); // 5

        assert!((stream.progress() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_handle() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 10,
            ..Default::default()
        });

        let handle = stream.handle();

        stream.start().unwrap();
        assert_eq!(handle.state(), StreamState::Running);

        stream.advance_step();
        assert_eq!(handle.step(), 1);

        stream.cancel().unwrap();
        assert!(handle.is_cancelled());
    }

    #[tokio::test]
    async fn test_preview_early_termination() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 20,
            ..Default::default()
        });

        stream.start().unwrap();
        assert_eq!(stream.get_state(), StreamState::Running);

        // Simulate some steps
        for _ in 0..5 {
            stream.advance_step();
        }
        assert_eq!(stream.current_step(), 5);

        // Cancel mid-generation
        stream.cancel().unwrap();
        assert!(stream.is_cancelled());
        assert_eq!(stream.get_state(), StreamState::Cancelled);

        // Verify progress stopped at cancellation point
        assert!((stream.progress() - 25.0).abs() < 0.01);

        // Verify frame submission fails after cancellation
        let frame = PreviewFrame::new(
            6,
            20,
            PreviewData::Raw(vec![]),
            64,
            64,
            PreviewQuality::Thumbnail,
        );
        let result = stream.submit_frame(frame).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_preview_async_streaming() {
        let mut stream = GenerationStream::new(StreamConfig {
            total_steps: 10,
            buffer_size: 5,
            ..Default::default()
        });

        // Take receiver before spawning
        let mut rx = stream.take_frame_receiver().expect("Should have receiver");

        stream.start().unwrap();

        // Submit frames asynchronously
        for step in 0..5 {
            let frame = PreviewFrame::new(
                step,
                10,
                PreviewData::Raw(vec![128u8; 64 * 64 * 4]),
                64,
                64,
                PreviewQuality::Thumbnail,
            );
            stream.submit_frame(frame).await.unwrap();
            stream.advance_step();
        }

        // Receive frames
        let mut received = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            received.push(frame);
        }

        assert_eq!(received.len(), 5);
        assert_eq!(received[0].step, 0);
        assert_eq!(received[4].step, 4);
    }

    #[tokio::test]
    async fn test_state_subscription() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 5,
            ..Default::default()
        });

        let mut state_rx = stream.subscribe_state();

        // Initial state
        assert_eq!(*state_rx.borrow(), StreamState::Idle);

        // Start
        stream.start().unwrap();
        state_rx.changed().await.unwrap();
        assert_eq!(*state_rx.borrow(), StreamState::Running);

        // Pause
        stream.pause().unwrap();
        state_rx.changed().await.unwrap();
        assert_eq!(*state_rx.borrow(), StreamState::Paused);

        // Resume
        stream.resume().unwrap();
        state_rx.changed().await.unwrap();
        assert_eq!(*state_rx.borrow(), StreamState::Running);
    }

    #[test]
    fn test_stream_completion() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 3,
            ..Default::default()
        });

        stream.start().unwrap();

        // Advance to completion
        stream.advance_step(); // 1
        stream.advance_step(); // 2
        stream.advance_step(); // 3

        assert_eq!(stream.get_state(), StreamState::Completed);
        assert!((stream.progress() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_should_preview_scheduling() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 20,
            schedule_mode: ScheduleMode::Interval { steps: 5 },
            ..Default::default()
        });

        stream.start().unwrap();

        // Step 0 should preview
        assert!(stream.should_preview());

        stream.advance_step();
        assert!(!stream.should_preview()); // Step 1

        // Advance to step 5
        for _ in 0..4 {
            stream.advance_step();
        }
        assert!(stream.should_preview()); // Step 5
    }

    #[test]
    fn test_current_quality_progression() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 100,
            ..Default::default()
        });

        stream.start().unwrap();

        // Early step: thumbnail
        let early_quality = stream.current_quality();
        assert_eq!(early_quality, PreviewQuality::Thumbnail);

        // Advance past 60%
        for _ in 0..70 {
            stream.advance_step();
        }
        let late_quality = stream.current_quality();
        assert!(
            late_quality == PreviewQuality::Medium || late_quality == PreviewQuality::Full,
            "Late quality should be Medium or Full"
        );
    }

    #[test]
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();

        assert_eq!(config.total_steps, 20);
        assert_eq!(config.buffer_size, 10);
        assert!(config.allow_frame_skip);
        assert_eq!(config.max_concurrent_decodes, 2);
    }

    #[test]
    fn test_stream_handle_progress() {
        let stream = GenerationStream::new(StreamConfig {
            total_steps: 10,
            ..Default::default()
        });

        let handle = stream.handle();
        stream.start().unwrap();

        assert_eq!(handle.progress(), 0.0);

        stream.advance_step();
        stream.advance_step();
        stream.advance_step();

        // 3 out of 10 = 30%
        assert!((handle.progress() - 30.0).abs() < 0.01);
    }
}
