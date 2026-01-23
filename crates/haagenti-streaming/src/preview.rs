//! Preview frame handling

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Preview quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreviewQuality {
    /// 64x64 thumbnail
    Thumbnail,
    /// 256x256 low quality
    Low,
    /// 512x512 medium quality
    Medium,
    /// Full resolution
    Full,
}

impl PreviewQuality {
    /// Get resolution for this quality
    pub fn resolution(&self) -> (u32, u32) {
        match self {
            PreviewQuality::Thumbnail => (64, 64),
            PreviewQuality::Low => (256, 256),
            PreviewQuality::Medium => (512, 512),
            PreviewQuality::Full => (1024, 1024),
        }
    }

    /// Estimated decode time in ms
    pub fn decode_time_ms(&self) -> u32 {
        match self {
            PreviewQuality::Thumbnail => 5,
            PreviewQuality::Low => 20,
            PreviewQuality::Medium => 50,
            PreviewQuality::Full => 150,
        }
    }

    /// Memory size estimate in bytes
    pub fn memory_size(&self) -> usize {
        let (w, h) = self.resolution();
        (w * h * 4) as usize // RGBA
    }
}

/// Configuration for preview generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewConfig {
    /// Quality level
    pub quality: PreviewQuality,
    /// Whether to include full resolution with final preview
    pub final_full_res: bool,
    /// JPEG quality (0-100) for compressed previews
    pub jpeg_quality: u8,
    /// Whether to use progressive JPEG
    pub progressive_jpeg: bool,
    /// Maximum preview latency in ms
    pub max_latency_ms: u32,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            quality: PreviewQuality::Medium,
            final_full_res: true,
            jpeg_quality: 85,
            progressive_jpeg: true,
            max_latency_ms: 100,
        }
    }
}

/// A single preview frame
#[derive(Debug, Clone)]
pub struct PreviewFrame {
    /// Step number
    pub step: u32,
    /// Total steps
    pub total_steps: u32,
    /// Image data (raw RGBA or compressed)
    pub data: PreviewData,
    /// Width
    pub width: u32,
    /// Height
    pub height: u32,
    /// Generation timestamp
    pub timestamp: Instant,
    /// Estimated completion percentage
    pub progress: f32,
    /// Whether this is the final frame
    pub is_final: bool,
    /// Quality level of this preview
    pub quality: PreviewQuality,
    /// Decode time for this frame
    pub decode_time_ms: u32,
}

/// Preview image data
#[derive(Debug, Clone)]
pub enum PreviewData {
    /// Raw RGBA pixels
    Raw(Vec<u8>),
    /// JPEG compressed
    Jpeg(Vec<u8>),
    /// PNG compressed
    Png(Vec<u8>),
    /// Latent space (not decoded)
    Latent(Vec<f32>),
}

impl PreviewData {
    /// Size in bytes
    pub fn size(&self) -> usize {
        match self {
            PreviewData::Raw(data) => data.len(),
            PreviewData::Jpeg(data) => data.len(),
            PreviewData::Png(data) => data.len(),
            PreviewData::Latent(data) => data.len() * 4,
        }
    }

    /// Whether this is compressed
    pub fn is_compressed(&self) -> bool {
        matches!(self, PreviewData::Jpeg(_) | PreviewData::Png(_))
    }

    /// Get raw bytes if available
    pub fn as_raw(&self) -> Option<&[u8]> {
        match self {
            PreviewData::Raw(data) => Some(data),
            _ => None,
        }
    }
}

impl PreviewFrame {
    /// Create a new preview frame
    pub fn new(
        step: u32,
        total_steps: u32,
        data: PreviewData,
        width: u32,
        height: u32,
        quality: PreviewQuality,
    ) -> Self {
        let progress = if total_steps > 0 {
            (step as f32 / total_steps as f32).min(1.0)
        } else {
            0.0
        };

        Self {
            step,
            total_steps,
            data,
            width,
            height,
            timestamp: Instant::now(),
            progress,
            is_final: step >= total_steps,
            quality,
            decode_time_ms: 0,
        }
    }

    /// Create from latent
    pub fn from_latent(
        step: u32,
        total_steps: u32,
        latent: Vec<f32>,
        quality: PreviewQuality,
    ) -> Self {
        let (width, height) = quality.resolution();
        Self::new(
            step,
            total_steps,
            PreviewData::Latent(latent),
            width,
            height,
            quality,
        )
    }

    /// Mark as final frame
    pub fn finalize(mut self) -> Self {
        self.is_final = true;
        self.progress = 1.0;
        self
    }

    /// Set decode time
    pub fn with_decode_time(mut self, ms: u32) -> Self {
        self.decode_time_ms = ms;
        self
    }

    /// Estimated quality based on step progress
    pub fn estimated_quality_score(&self) -> f32 {
        // Quality improves with denoising steps
        // Using sigmoid-like curve
        let t = self.progress;
        let quality = 1.0 / (1.0 + (-10.0 * (t - 0.5)).exp());
        quality * 100.0
    }
}

/// Preview frame buffer for smooth playback
#[derive(Debug)]
pub struct PreviewBuffer {
    /// Buffered frames
    frames: Vec<PreviewFrame>,
    /// Maximum buffer size
    max_size: usize,
    /// Current playback position
    position: usize,
}

impl PreviewBuffer {
    /// Create a new buffer
    pub fn new(max_size: usize) -> Self {
        Self {
            frames: Vec::with_capacity(max_size),
            max_size,
            position: 0,
        }
    }

    /// Add a frame to the buffer
    pub fn push(&mut self, frame: PreviewFrame) {
        if self.frames.len() >= self.max_size {
            // Remove oldest non-final frame
            if let Some(pos) = self.frames.iter().position(|f| !f.is_final) {
                self.frames.remove(pos);
                if self.position > pos {
                    self.position = self.position.saturating_sub(1);
                }
            }
        }
        self.frames.push(frame);
    }

    /// Get the next frame for playback
    pub fn next_frame(&mut self) -> Option<&PreviewFrame> {
        if self.position < self.frames.len() {
            let frame = &self.frames[self.position];
            self.position += 1;
            Some(frame)
        } else {
            None
        }
    }

    /// Get the latest frame
    pub fn latest(&self) -> Option<&PreviewFrame> {
        self.frames.last()
    }

    /// Get the final frame if available
    pub fn final_frame(&self) -> Option<&PreviewFrame> {
        self.frames.iter().find(|f| f.is_final)
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.frames.clear();
        self.position = 0;
    }

    /// Number of buffered frames
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether buffer is empty
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Frames remaining
    pub fn remaining(&self) -> usize {
        self.frames.len().saturating_sub(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preview_quality() {
        assert_eq!(PreviewQuality::Thumbnail.resolution(), (64, 64));
        assert_eq!(PreviewQuality::Full.resolution(), (1024, 1024));
        assert!(PreviewQuality::Thumbnail.memory_size() < PreviewQuality::Full.memory_size());
    }

    #[test]
    fn test_preview_frame() {
        let frame = PreviewFrame::new(
            5,
            20,
            PreviewData::Raw(vec![0u8; 64 * 64 * 4]),
            64,
            64,
            PreviewQuality::Thumbnail,
        );

        assert_eq!(frame.step, 5);
        assert!((frame.progress - 0.25).abs() < 0.01);
        assert!(!frame.is_final);
    }

    #[test]
    fn test_preview_buffer() {
        let mut buffer = PreviewBuffer::new(5);

        for i in 0..3 {
            buffer.push(PreviewFrame::new(
                i,
                10,
                PreviewData::Raw(vec![]),
                64,
                64,
                PreviewQuality::Thumbnail,
            ));
        }

        assert_eq!(buffer.len(), 3);
        assert!(buffer.next_frame().is_some());
        assert_eq!(buffer.remaining(), 2);
    }

    #[test]
    fn test_quality_score() {
        let early = PreviewFrame::new(
            2,
            20,
            PreviewData::Raw(vec![]),
            64,
            64,
            PreviewQuality::Thumbnail,
        );
        let late = PreviewFrame::new(
            18,
            20,
            PreviewData::Raw(vec![]),
            64,
            64,
            PreviewQuality::Thumbnail,
        );

        assert!(late.estimated_quality_score() > early.estimated_quality_score());
    }

    #[test]
    fn test_preview_frames_generated() {
        // Test generating multiple preview frames across a generation
        let total_steps = 20;
        let mut frames = Vec::new();

        for step in 0..=total_steps {
            let quality = if step < 5 {
                PreviewQuality::Thumbnail
            } else if step < 15 {
                PreviewQuality::Medium
            } else {
                PreviewQuality::Full
            };

            let (width, height) = quality.resolution();
            let pixel_count = (width * height * 4) as usize;
            let data = PreviewData::Raw(vec![128u8; pixel_count]);

            let frame = PreviewFrame::new(step, total_steps, data, width, height, quality);
            frames.push(frame);
        }

        // Verify all frames generated
        assert_eq!(frames.len(), 21);

        // First frame should be step 0
        assert_eq!(frames[0].step, 0);
        assert!(!frames[0].is_final);
        assert!((frames[0].progress - 0.0).abs() < 0.01);

        // Last frame should be final
        assert_eq!(frames[20].step, 20);
        assert!(frames[20].is_final);
        assert!((frames[20].progress - 1.0).abs() < 0.01);

        // Middle frame progress check
        let mid = &frames[10];
        assert_eq!(mid.step, 10);
        assert!((mid.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_preview_quality_curve() {
        // Test that quality estimation follows expected curve
        let total_steps = 100;
        let mut quality_scores = Vec::new();

        for step in 0..=total_steps {
            let frame = PreviewFrame::new(
                step,
                total_steps,
                PreviewData::Raw(vec![]),
                64,
                64,
                PreviewQuality::Thumbnail,
            );
            quality_scores.push((step, frame.estimated_quality_score()));
        }

        // Quality should be monotonically increasing
        for i in 1..quality_scores.len() {
            assert!(
                quality_scores[i].1 >= quality_scores[i - 1].1,
                "Quality should not decrease: step {} ({:.2}) < step {} ({:.2})",
                quality_scores[i].0,
                quality_scores[i].1,
                quality_scores[i - 1].0,
                quality_scores[i - 1].1
            );
        }

        // Early quality should be low (< 20%)
        assert!(
            quality_scores[10].1 < 20.0,
            "Early quality should be low: {:.2}",
            quality_scores[10].1
        );

        // Late quality should be high (> 80%)
        assert!(
            quality_scores[90].1 > 80.0,
            "Late quality should be high: {:.2}",
            quality_scores[90].1
        );

        // Final quality should approach 100
        assert!(
            quality_scores[100].1 > 95.0,
            "Final quality should be near 100: {:.2}",
            quality_scores[100].1
        );
    }

    #[test]
    fn test_preview_data_variants() {
        // Test all PreviewData variants
        let raw_data = PreviewData::Raw(vec![255u8; 1000]);
        assert_eq!(raw_data.size(), 1000);
        assert!(!raw_data.is_compressed());
        assert!(raw_data.as_raw().is_some());

        let jpeg_data = PreviewData::Jpeg(vec![0xFF, 0xD8, 0xFF, 0xE0]);
        assert_eq!(jpeg_data.size(), 4);
        assert!(jpeg_data.is_compressed());
        assert!(jpeg_data.as_raw().is_none());

        let png_data = PreviewData::Png(vec![0x89, 0x50, 0x4E, 0x47]);
        assert_eq!(png_data.size(), 4);
        assert!(png_data.is_compressed());
        assert!(png_data.as_raw().is_none());

        let latent_data = PreviewData::Latent(vec![0.5f32; 100]);
        assert_eq!(latent_data.size(), 400); // 100 floats * 4 bytes
        assert!(!latent_data.is_compressed());
        assert!(latent_data.as_raw().is_none());
    }

    #[test]
    fn test_preview_buffer_operations() {
        let mut buffer = PreviewBuffer::new(5);

        // Add frames
        for i in 0..3 {
            buffer.push(PreviewFrame::new(
                i,
                10,
                PreviewData::Raw(vec![]),
                64,
                64,
                PreviewQuality::Thumbnail,
            ));
        }

        assert_eq!(buffer.len(), 3);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.remaining(), 3);

        // Consume frames
        let f1 = buffer.next_frame();
        assert!(f1.is_some());
        assert_eq!(f1.unwrap().step, 0);
        assert_eq!(buffer.remaining(), 2);

        // Get latest
        let latest = buffer.latest();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().step, 2);

        // Clear
        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_preview_buffer_overflow() {
        let mut buffer = PreviewBuffer::new(3);

        // Add more frames than buffer can hold
        for i in 0..5 {
            buffer.push(PreviewFrame::new(
                i,
                10,
                PreviewData::Raw(vec![]),
                64,
                64,
                PreviewQuality::Thumbnail,
            ));
        }

        // Buffer should have evicted oldest frames
        assert_eq!(buffer.len(), 3);

        // Latest should be step 4
        assert_eq!(buffer.latest().unwrap().step, 4);
    }

    #[test]
    fn test_preview_frame_finalize() {
        let frame = PreviewFrame::new(
            10,
            20,
            PreviewData::Raw(vec![]),
            64,
            64,
            PreviewQuality::Medium,
        );

        assert!(!frame.is_final);
        assert!((frame.progress - 0.5).abs() < 0.01);

        let finalized = frame.finalize();
        assert!(finalized.is_final);
        assert!((finalized.progress - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_preview_frame_with_decode_time() {
        let frame = PreviewFrame::new(
            5,
            20,
            PreviewData::Raw(vec![]),
            64,
            64,
            PreviewQuality::Thumbnail,
        );

        assert_eq!(frame.decode_time_ms, 0);

        let with_time = frame.with_decode_time(42);
        assert_eq!(with_time.decode_time_ms, 42);
    }

    #[test]
    fn test_preview_config_default() {
        let config = PreviewConfig::default();

        assert_eq!(config.quality, PreviewQuality::Medium);
        assert!(config.final_full_res);
        assert_eq!(config.jpeg_quality, 85);
        assert!(config.progressive_jpeg);
        assert_eq!(config.max_latency_ms, 100);
    }
}
