//! Streaming VAE decoder for latent to image conversion

use crate::{PreviewData, PreviewFrame, PreviewQuality, Result, StreamError};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Configuration for the streaming decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Enable tiled decoding for memory efficiency
    pub tiled: bool,
    /// Tile size for tiled decoding
    pub tile_size: u32,
    /// Enable progressive decoding
    pub progressive: bool,
    /// Use half precision (FP16)
    pub half_precision: bool,
    /// Maximum decode time before timeout
    pub timeout_ms: u32,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            tiled: true,
            tile_size: 256,
            progressive: true,
            half_precision: true,
            timeout_ms: 5000,
        }
    }
}

/// A decoded frame ready for display
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// RGBA pixel data
    pub pixels: Vec<u8>,
    /// Width
    pub width: u32,
    /// Height
    pub height: u32,
    /// Source step
    pub step: u32,
    /// Decode time in ms
    pub decode_time_ms: u32,
    /// Whether this was decoded at full quality
    pub full_quality: bool,
}

impl DecodedFrame {
    /// Create from raw pixels
    pub fn new(pixels: Vec<u8>, width: u32, height: u32, step: u32) -> Self {
        Self {
            pixels,
            width,
            height,
            step,
            decode_time_ms: 0,
            full_quality: false,
        }
    }

    /// Get pixel at (x, y)
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        if idx + 4 <= self.pixels.len() {
            Some([
                self.pixels[idx],
                self.pixels[idx + 1],
                self.pixels[idx + 2],
                self.pixels[idx + 3],
            ])
        } else {
            None
        }
    }

    /// Total pixel count
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }
}

/// Streaming decoder for VAE latent -> image conversion
#[derive(Debug)]
pub struct StreamDecoder {
    config: DecoderConfig,
}

impl StreamDecoder {
    /// Create a new decoder
    pub fn new(config: DecoderConfig) -> Self {
        Self { config }
    }

    /// Decode a latent tensor to an image
    ///
    /// In a real implementation, this would:
    /// 1. Run the VAE decoder network
    /// 2. Convert to RGB/RGBA
    /// 3. Apply post-processing
    pub fn decode_latent(
        &self,
        latent: &[f32],
        target_quality: PreviewQuality,
    ) -> Result<DecodedFrame> {
        let start = Instant::now();

        let (width, height) = target_quality.resolution();

        // Simulate decode (real implementation would run VAE)
        let latent_channels = 4;
        let latent_height = 64; // Typical SDXL latent size
        let latent_width = 64;
        let expected_latent_size = latent_channels * latent_height * latent_width;

        if latent.len() < expected_latent_size {
            // Create synthetic preview from available data
            return self.create_synthetic_preview(latent, width, height);
        }

        // Simulate VAE decode: latent -> image
        let pixels = self.simulate_vae_decode(latent, width, height);

        let decode_time = start.elapsed();

        Ok(DecodedFrame {
            pixels,
            width,
            height,
            step: 0,
            decode_time_ms: decode_time.as_millis() as u32,
            full_quality: target_quality == PreviewQuality::Full,
        })
    }

    /// Create synthetic preview from latent (quick visualization)
    fn create_synthetic_preview(
        &self,
        latent: &[f32],
        width: u32,
        height: u32,
    ) -> Result<DecodedFrame> {
        let pixel_count = (width * height) as usize;
        let mut pixels = vec![0u8; pixel_count * 4];

        // Simple visualization: map latent values to colors
        for i in 0..pixel_count {
            let latent_idx = i % latent.len();
            let value = latent[latent_idx];

            // Map to grayscale with some color
            let intensity = ((value + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;

            pixels[i * 4] = intensity; // R
            pixels[i * 4 + 1] = (intensity as f32 * 0.9) as u8; // G
            pixels[i * 4 + 2] = (intensity as f32 * 0.8) as u8; // B
            pixels[i * 4 + 3] = 255; // A
        }

        Ok(DecodedFrame {
            pixels,
            width,
            height,
            step: 0,
            decode_time_ms: 1,
            full_quality: false,
        })
    }

    /// Simulate VAE decode (placeholder for real implementation)
    fn simulate_vae_decode(&self, _latent: &[f32], width: u32, height: u32) -> Vec<u8> {
        // Create gradient image as placeholder
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);

        for y in 0..height {
            for x in 0..width {
                let r = (x as f32 / width as f32 * 255.0) as u8;
                let g = (y as f32 / height as f32 * 255.0) as u8;
                let b = 128u8;
                let a = 255u8;
                pixels.extend_from_slice(&[r, g, b, a]);
            }
        }

        pixels
    }

    /// Decode preview frame
    pub fn decode_frame(&self, frame: &PreviewFrame) -> Result<DecodedFrame> {
        let start = Instant::now();

        let decoded = match &frame.data {
            PreviewData::Raw(data) => DecodedFrame {
                pixels: data.clone(),
                width: frame.width,
                height: frame.height,
                step: frame.step,
                decode_time_ms: 0,
                full_quality: frame.quality == PreviewQuality::Full,
            },
            PreviewData::Jpeg(_data) => {
                // Would decode JPEG here
                self.placeholder_frame(frame)?
            }
            PreviewData::Png(_data) => {
                // Would decode PNG here
                self.placeholder_frame(frame)?
            }
            PreviewData::Latent(latent) => {
                let mut decoded = self.decode_latent(latent, frame.quality)?;
                decoded.step = frame.step;
                decoded
            }
        };

        let decode_time = start.elapsed();
        let mut result = decoded;
        result.decode_time_ms = decode_time.as_millis() as u32;

        Ok(result)
    }

    /// Create placeholder frame (for unsupported formats)
    fn placeholder_frame(&self, frame: &PreviewFrame) -> Result<DecodedFrame> {
        let pixel_count = (frame.width * frame.height) as usize;
        let pixels = vec![128u8; pixel_count * 4]; // Gray placeholder

        Ok(DecodedFrame {
            pixels,
            width: frame.width,
            height: frame.height,
            step: frame.step,
            decode_time_ms: 1,
            full_quality: false,
        })
    }

    /// Upscale a decoded frame
    pub fn upscale(&self, frame: &DecodedFrame, target_width: u32, target_height: u32) -> DecodedFrame {
        if frame.width == target_width && frame.height == target_height {
            return frame.clone();
        }

        // Simple bilinear upscaling
        let mut pixels = Vec::with_capacity((target_width * target_height * 4) as usize);

        for y in 0..target_height {
            for x in 0..target_width {
                // Map to source coordinates
                let src_x = (x as f32 / target_width as f32) * frame.width as f32;
                let src_y = (y as f32 / target_height as f32) * frame.height as f32;

                // Bilinear interpolation
                let pixel = self.bilinear_sample(frame, src_x, src_y);
                pixels.extend_from_slice(&pixel);
            }
        }

        DecodedFrame {
            pixels,
            width: target_width,
            height: target_height,
            step: frame.step,
            decode_time_ms: frame.decode_time_ms,
            full_quality: frame.full_quality,
        }
    }

    /// Bilinear sampling
    fn bilinear_sample(&self, frame: &DecodedFrame, x: f32, y: f32) -> [u8; 4] {
        let x0 = (x.floor() as u32).min(frame.width - 1);
        let x1 = (x.ceil() as u32).min(frame.width - 1);
        let y0 = (y.floor() as u32).min(frame.height - 1);
        let y1 = (y.ceil() as u32).min(frame.height - 1);

        let fx = x - x.floor();
        let fy = y - y.floor();

        let p00 = frame.get_pixel(x0, y0).unwrap_or([0; 4]);
        let p10 = frame.get_pixel(x1, y0).unwrap_or([0; 4]);
        let p01 = frame.get_pixel(x0, y1).unwrap_or([0; 4]);
        let p11 = frame.get_pixel(x1, y1).unwrap_or([0; 4]);

        let mut result = [0u8; 4];
        for i in 0..4 {
            let top = p00[i] as f32 * (1.0 - fx) + p10[i] as f32 * fx;
            let bottom = p01[i] as f32 * (1.0 - fx) + p11[i] as f32 * fx;
            result[i] = (top * (1.0 - fy) + bottom * fy) as u8;
        }

        result
    }

    /// Get configuration
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_latent() {
        let decoder = StreamDecoder::new(DecoderConfig::default());
        let latent: Vec<f32> = (0..16384).map(|i| (i as f32 / 16384.0) - 0.5).collect();

        let frame = decoder.decode_latent(&latent, PreviewQuality::Thumbnail).unwrap();

        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.pixels.len(), 64 * 64 * 4);
    }

    #[test]
    fn test_upscale() {
        let decoder = StreamDecoder::new(DecoderConfig::default());

        let small = DecodedFrame::new(vec![128u8; 64 * 64 * 4], 64, 64, 0);
        let large = decoder.upscale(&small, 256, 256);

        assert_eq!(large.width, 256);
        assert_eq!(large.height, 256);
        assert_eq!(large.pixels.len(), 256 * 256 * 4);
    }

    #[test]
    fn test_get_pixel() {
        let pixels = vec![
            255, 0, 0, 255, // Red
            0, 255, 0, 255, // Green
            0, 0, 255, 255, // Blue
            255, 255, 255, 255, // White
        ];

        let frame = DecodedFrame::new(pixels, 2, 2, 0);

        assert_eq!(frame.get_pixel(0, 0), Some([255, 0, 0, 255])); // Red
        assert_eq!(frame.get_pixel(1, 0), Some([0, 255, 0, 255])); // Green
        assert_eq!(frame.get_pixel(0, 1), Some([0, 0, 255, 255])); // Blue
        assert_eq!(frame.get_pixel(1, 1), Some([255, 255, 255, 255])); // White
        assert_eq!(frame.get_pixel(2, 2), None); // Out of bounds
    }

    #[test]
    fn test_preview_decoder_creation() {
        // Test with default config
        let decoder_default = StreamDecoder::new(DecoderConfig::default());
        assert!(decoder_default.config().tiled);
        assert!(decoder_default.config().progressive);
        assert_eq!(decoder_default.config().tile_size, 256);

        // Test with custom config
        let custom_config = DecoderConfig {
            tiled: false,
            tile_size: 512,
            progressive: false,
            half_precision: false,
            timeout_ms: 10000,
        };
        let decoder_custom = StreamDecoder::new(custom_config);
        assert!(!decoder_custom.config().tiled);
        assert!(!decoder_custom.config().progressive);
        assert_eq!(decoder_custom.config().tile_size, 512);
        assert_eq!(decoder_custom.config().timeout_ms, 10000);
    }

    #[test]
    fn test_preview_vae_integration() {
        let decoder = StreamDecoder::new(DecoderConfig::default());

        // Test VAE decode simulation with full latent
        let latent: Vec<f32> = (0..16384).map(|i| (i as f32 / 16384.0) * 2.0 - 1.0).collect();

        // Decode at different quality levels
        let thumb = decoder.decode_latent(&latent, PreviewQuality::Thumbnail).unwrap();
        assert_eq!(thumb.width, 64);
        assert_eq!(thumb.height, 64);
        assert_eq!(thumb.pixels.len(), 64 * 64 * 4);

        let low = decoder.decode_latent(&latent, PreviewQuality::Low).unwrap();
        assert_eq!(low.width, 256);
        assert_eq!(low.height, 256);
        assert_eq!(low.pixels.len(), 256 * 256 * 4);

        let medium = decoder.decode_latent(&latent, PreviewQuality::Medium).unwrap();
        assert_eq!(medium.width, 512);
        assert_eq!(medium.height, 512);
        assert_eq!(medium.pixels.len(), 512 * 512 * 4);

        let full = decoder.decode_latent(&latent, PreviewQuality::Full).unwrap();
        assert_eq!(full.width, 1024);
        assert_eq!(full.height, 1024);
        assert_eq!(full.pixels.len(), 1024 * 1024 * 4);
        assert!(full.full_quality);
    }

    #[test]
    fn test_preview_resolution_scaling() {
        let decoder = StreamDecoder::new(DecoderConfig::default());

        // Test that different quality levels produce different resolutions
        let small_frame = DecodedFrame::new(vec![128u8; 64 * 64 * 4], 64, 64, 0);

        // Upscale to various sizes
        let to_256 = decoder.upscale(&small_frame, 256, 256);
        assert_eq!(to_256.width, 256);
        assert_eq!(to_256.height, 256);
        assert_eq!(to_256.pixels.len(), 256 * 256 * 4);

        let to_512 = decoder.upscale(&small_frame, 512, 512);
        assert_eq!(to_512.width, 512);
        assert_eq!(to_512.pixels.len(), 512 * 512 * 4);

        // Verify no-op when dimensions match
        let same_size = decoder.upscale(&small_frame, 64, 64);
        assert_eq!(same_size.width, 64);
        assert_eq!(same_size.height, 64);

        // Test non-square upscale
        let non_square = decoder.upscale(&small_frame, 128, 256);
        assert_eq!(non_square.width, 128);
        assert_eq!(non_square.height, 256);
        assert_eq!(non_square.pixels.len(), 128 * 256 * 4);
    }

    #[test]
    fn test_decode_frame_variants() {
        let decoder = StreamDecoder::new(DecoderConfig::default());

        // Test decoding Raw data
        let raw_frame = PreviewFrame::new(
            5,
            20,
            PreviewData::Raw(vec![128u8; 64 * 64 * 4]),
            64,
            64,
            PreviewQuality::Thumbnail,
        );
        let decoded_raw = decoder.decode_frame(&raw_frame).unwrap();
        assert_eq!(decoded_raw.width, 64);
        assert_eq!(decoded_raw.step, 5);

        // Test decoding Latent data
        let latent_data: Vec<f32> = (0..16384).map(|i| (i as f32 / 16384.0) - 0.5).collect();
        let latent_frame = PreviewFrame::from_latent(
            10,
            20,
            latent_data,
            PreviewQuality::Thumbnail,
        );
        let decoded_latent = decoder.decode_frame(&latent_frame).unwrap();
        assert_eq!(decoded_latent.step, 10);
        assert_eq!(decoded_latent.width, 64);

        // Test decoding JPEG placeholder
        let jpeg_frame = PreviewFrame::new(
            15,
            20,
            PreviewData::Jpeg(vec![0xFF, 0xD8, 0xFF]), // JPEG magic bytes
            64,
            64,
            PreviewQuality::Thumbnail,
        );
        let decoded_jpeg = decoder.decode_frame(&jpeg_frame).unwrap();
        assert_eq!(decoded_jpeg.step, 15);
    }

    #[test]
    fn test_synthetic_preview_small_latent() {
        let decoder = StreamDecoder::new(DecoderConfig::default());

        // Test with smaller-than-expected latent (triggers synthetic preview)
        let small_latent: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) - 0.5).collect();
        let result = decoder.decode_latent(&small_latent, PreviewQuality::Thumbnail).unwrap();

        // Should still produce valid frame
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
        assert!(!result.full_quality);

        // Verify pixels have some variation from latent values
        let pixels = &result.pixels;
        assert!(!pixels.iter().all(|&p| p == pixels[0]));
    }
}
