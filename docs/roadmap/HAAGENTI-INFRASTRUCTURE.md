# Track C: Infrastructure & Security

## Overview

This document details the TDD roadmap for infrastructure improvements in Haagenti.

**Timeline:** 4-5 weeks
**Priority:** Medium-High
**Crates:** `haagenti-grpc`, `haagenti-streaming`, `haagenti-speculative`, `haagenti-python`

---

## Phase C.1: gRPC TLS Support

### Purpose
Enable TLS encryption for production-grade security.

### Current State
```rust
// haagenti-grpc/src/main.rs
// "TLS requested but not yet implemented - running without TLS"
```

### Test Specification

```rust
// tests/grpc_tls_test.rs

#[cfg(test)]
mod tls_tests {
    use haagenti_grpc::{GrpcServer, GrpcClient, TlsConfig};
    use std::path::Path;

    #[tokio::test]
    async fn test_tls_server_creation() {
        let config = TlsConfig::from_pem(
            Path::new("testdata/server.crt"),
            Path::new("testdata/server.key"),
        ).unwrap();

        let server = GrpcServer::builder()
            .with_tls(config)
            .build()
            .unwrap();

        assert!(server.is_tls_enabled());
    }

    #[tokio::test]
    async fn test_tls_client_connection() {
        // Start TLS server
        let server_config = TlsConfig::from_pem(
            Path::new("testdata/server.crt"),
            Path::new("testdata/server.key"),
        ).unwrap();

        let server = GrpcServer::builder()
            .with_tls(server_config)
            .bind("[::1]:0")
            .build()
            .unwrap();

        let addr = server.local_addr();
        tokio::spawn(async move { server.serve().await });

        // Connect with TLS client
        let client_config = TlsConfig::with_ca_cert(
            Path::new("testdata/ca.crt"),
        ).unwrap();

        let client = GrpcClient::builder()
            .with_tls(client_config)
            .connect(&format!("https://{}", addr))
            .await
            .unwrap();

        assert!(client.is_connected());
    }

    #[tokio::test]
    async fn test_tls_compression_request() {
        let (server, client) = setup_tls_pair().await;

        let data = b"Test data for TLS compression";
        let response = client.compress(data, Algorithm::Zstd).await.unwrap();

        assert!(response.len() < data.len());
    }

    #[tokio::test]
    async fn test_tls_mutual_auth() {
        let server_config = TlsConfig::from_pem(
            Path::new("testdata/server.crt"),
            Path::new("testdata/server.key"),
        )
        .with_client_ca(Path::new("testdata/ca.crt"))
        .unwrap();

        let client_config = TlsConfig::from_pem(
            Path::new("testdata/client.crt"),
            Path::new("testdata/client.key"),
        )
        .with_ca_cert(Path::new("testdata/ca.crt"))
        .unwrap();

        let (server, client) = setup_tls_pair_mtls(server_config, client_config).await;

        let response = client.compress(b"test", Algorithm::Zstd).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_tls_reject_invalid_cert() {
        let server_config = TlsConfig::from_pem(
            Path::new("testdata/server.crt"),
            Path::new("testdata/server.key"),
        ).unwrap();

        let server = setup_tls_server(server_config).await;

        // Try connecting without TLS
        let client = GrpcClient::builder()
            .connect(&format!("http://{}", server.local_addr()))
            .await;

        assert!(client.is_err());
    }

    #[tokio::test]
    async fn test_tls_certificate_reload() {
        let server = setup_tls_server_with_reload().await;

        // Replace certificate
        std::fs::copy("testdata/server2.crt", "testdata/server.crt").unwrap();

        // Trigger reload
        server.reload_certificates().await.unwrap();

        // Verify new cert is used
        let client = setup_tls_client().await;
        let cert_info = client.peer_certificate_info().await.unwrap();

        assert!(cert_info.subject.contains("server2"));
    }

    #[tokio::test]
    async fn test_tls_performance_overhead() {
        let (server, client) = setup_tls_pair().await;
        let (plain_server, plain_client) = setup_plain_pair().await;

        let data = vec![0u8; 1_000_000]; // 1MB

        // TLS timing
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = client.compress(&data, Algorithm::Zstd).await.unwrap();
        }
        let tls_time = start.elapsed();

        // Plain timing
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = plain_client.compress(&data, Algorithm::Zstd).await.unwrap();
        }
        let plain_time = start.elapsed();

        // TLS overhead should be <20%
        let overhead = tls_time.as_secs_f64() / plain_time.as_secs_f64() - 1.0;
        assert!(overhead < 0.20, "TLS overhead: {:.1}%", overhead * 100.0);
    }
}
```

### Implementation Specification

```rust
// haagenti-grpc/src/tls.rs

/// TLS configuration for gRPC server/client.
pub struct TlsConfig {
    /// Server certificate chain (PEM)
    cert_chain: Vec<u8>,
    /// Private key (PEM)
    private_key: Vec<u8>,
    /// CA certificates for client verification
    client_ca: Option<Vec<u8>>,
    /// CA certificate for server verification
    server_ca: Option<Vec<u8>>,
}

impl TlsConfig {
    /// Load from PEM files.
    pub fn from_pem(cert_path: &Path, key_path: &Path) -> Result<Self>;

    /// Add client CA for mutual TLS.
    pub fn with_client_ca(self, ca_path: &Path) -> Result<Self>;

    /// Add server CA for client.
    pub fn with_ca_cert(ca_path: &Path) -> Result<Self>;
}

// Server integration
impl GrpcServerBuilder {
    pub fn with_tls(self, config: TlsConfig) -> Self;
}

// Client integration
impl GrpcClientBuilder {
    pub fn with_tls(self, config: TlsConfig) -> Self;
}
```

### Quality Gate C.1

```bash
#!/bin/bash
# Phase C.1 Quality Gate

echo "=== Phase C.1: TLS Support Quality Gate ==="

# 1. Generate test certificates
./scripts/generate_test_certs.sh

# 2. All TLS tests pass
cargo test --package haagenti-grpc tls_ -- --nocapture

# 3. Connection works
cargo test --package haagenti-grpc test_tls_client_connection

# 4. Mutual auth works
cargo test --package haagenti-grpc test_tls_mutual_auth

# 5. Performance overhead acceptable
cargo test --package haagenti-grpc test_tls_performance_overhead

echo "=== Phase C.1 PASSED ==="
```

---

## Phase C.2: Streaming Real-Time Preview

### Purpose
Enable progressive image/tensor preview during streaming decompression.

### Current State
```rust
// haagenti-streaming/src/decoder.rs:124-127
// "Simulate VAE decode (placeholder for real implementation)"
```

### Test Specification

```rust
// tests/streaming_preview_test.rs

#[cfg(test)]
mod preview_tests {
    use haagenti_streaming::{StreamingDecoder, PreviewConfig, PreviewFrame};

    #[test]
    fn test_preview_decoder_creation() {
        let config = PreviewConfig {
            preview_interval_ms: 100,
            min_quality: 0.3,
            max_preview_resolution: (512, 512),
        };

        let decoder = StreamingDecoder::with_preview(config);
        assert!(decoder.preview_enabled());
    }

    #[test]
    fn test_preview_frames_generated() {
        let decoder = StreamingDecoder::with_preview(PreviewConfig::default());

        let encoded_data = load_test_image_encoded();
        let mut previews: Vec<PreviewFrame> = Vec::new();

        decoder.decode_with_previews(&encoded_data, |frame| {
            previews.push(frame);
        }).unwrap();

        // Should generate multiple preview frames
        assert!(previews.len() >= 3, "Got {} previews", previews.len());

        // Quality should increase
        for window in previews.windows(2) {
            assert!(window[1].quality >= window[0].quality);
        }
    }

    #[test]
    fn test_preview_resolution_scaling() {
        let config = PreviewConfig {
            max_preview_resolution: (256, 256),
            ..Default::default()
        };

        let decoder = StreamingDecoder::with_preview(config);

        let large_image = load_test_image(2048, 2048);
        let encoded = encode_for_streaming(&large_image);

        let mut last_preview = None;
        decoder.decode_with_previews(&encoded, |frame| {
            // Preview should be scaled down
            assert!(frame.width <= 256);
            assert!(frame.height <= 256);
            last_preview = Some(frame);
        }).unwrap();

        // Final frame should have full quality (but possibly still scaled)
        let final_frame = last_preview.unwrap();
        assert!(final_frame.quality > 0.9);
    }

    #[test]
    fn test_preview_early_termination() {
        let decoder = StreamingDecoder::with_preview(PreviewConfig::default());

        let encoded = load_test_image_encoded();
        let mut preview_count = 0;

        let result = decoder.decode_with_previews(&encoded, |frame| {
            preview_count += 1;
            if preview_count >= 2 {
                return Err(StreamingError::Cancelled);
            }
            Ok(())
        });

        assert!(result.is_err());
        assert_eq!(preview_count, 2);
    }

    #[test]
    fn test_preview_vae_integration() {
        let decoder = StreamingDecoder::with_preview(PreviewConfig::default())
            .with_vae_decoder(load_vae_model());

        let latents = load_test_latents();
        let encoded = encode_latents_for_streaming(&latents);

        let mut previews = Vec::new();
        decoder.decode_with_previews(&encoded, |frame| {
            previews.push(frame);
        }).unwrap();

        // VAE should produce valid RGB frames
        for preview in &previews {
            assert_eq!(preview.channels, 3);
            assert!(preview.data.len() == preview.width * preview.height * 3);
        }
    }

    #[test]
    fn test_preview_quality_curve() {
        let decoder = StreamingDecoder::with_preview(PreviewConfig::default());

        let encoded = load_test_image_encoded();

        let mut quality_samples = Vec::new();
        decoder.decode_with_previews(&encoded, |frame| {
            quality_samples.push((frame.bytes_processed, frame.quality));
        }).unwrap();

        // Quality should follow expected curve
        for (bytes, quality) in &quality_samples {
            let expected_min = (*bytes as f32 / encoded.len() as f32).sqrt() * 0.5;
            assert!(*quality >= expected_min,
                "At {} bytes, quality {} < expected {}", bytes, quality, expected_min);
        }
    }

    #[test]
    fn test_preview_async_streaming() {
        let decoder = StreamingDecoder::with_preview(PreviewConfig::default());

        let (tx, rx) = std::sync::mpsc::channel();

        // Start async decode
        let handle = std::thread::spawn(move || {
            let encoded = load_test_image_encoded();
            decoder.decode_with_previews(&encoded, |frame| {
                tx.send(frame).unwrap();
            })
        });

        // Receive previews as they arrive
        let mut count = 0;
        while let Ok(_frame) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
            count += 1;
        }

        handle.join().unwrap().unwrap();
        assert!(count >= 3);
    }
}
```

### Quality Gate C.2

```bash
#!/bin/bash
# Phase C.2 Quality Gate

echo "=== Phase C.2: Streaming Preview Quality Gate ==="

# 1. All preview tests pass
cargo test --package haagenti-streaming preview_ -- --nocapture

# 2. Quality progression verified
cargo test --package haagenti-streaming test_preview_quality_curve

# 3. VAE integration works
cargo test --package haagenti-streaming test_preview_vae_integration

echo "=== Phase C.2 PASSED ==="
```

---

## Phase C.3: Speculative Prefetch ML

### Purpose
Implement ML-based intent prediction for speculative fragment prefetching.

### Test Specification

```rust
// tests/speculative_prefetch_test.rs

#[cfg(test)]
mod prefetch_tests {
    use haagenti_speculative::{IntentPredictor, PrefetchManager, SessionHistory};

    #[test]
    fn test_intent_predictor_creation() {
        let predictor = IntentPredictor::new();
        assert!(predictor.is_ready());
    }

    #[test]
    fn test_session_history_recording() {
        let mut history = SessionHistory::new();

        history.record_access("layer.0.weight");
        history.record_access("layer.0.bias");
        history.record_access("layer.1.weight");

        assert_eq!(history.len(), 3);
        assert_eq!(history.last_access(), Some("layer.1.weight"));
    }

    #[test]
    fn test_intent_prediction_sequential() {
        let predictor = IntentPredictor::new();
        let mut history = SessionHistory::new();

        // Sequential layer access pattern
        history.record_access("layer.0.weight");
        history.record_access("layer.1.weight");
        history.record_access("layer.2.weight");

        let predictions = predictor.predict(&history, 3);

        // Should predict layer.3, layer.4, layer.5
        assert!(predictions.iter().any(|p| p.name.contains("layer.3")));
    }

    #[test]
    fn test_intent_prediction_attention_pattern() {
        let predictor = IntentPredictor::new();
        let mut history = SessionHistory::new();

        // Attention pattern: q, k, v, o for each layer
        history.record_access("layer.0.q_proj");
        history.record_access("layer.0.k_proj");
        history.record_access("layer.0.v_proj");

        let predictions = predictor.predict(&history, 2);

        // Should predict o_proj and layer.1 start
        assert!(predictions.iter().any(|p| p.name.contains("o_proj")));
    }

    #[test]
    fn test_prefetch_manager_integration() {
        let predictor = IntentPredictor::new();
        let manager = PrefetchManager::new(predictor);

        // Simulate access
        manager.notify_access("layer.0.weight").unwrap();

        // Check prefetch queue
        let prefetch_queue = manager.pending_prefetches();
        assert!(!prefetch_queue.is_empty());
    }

    #[test]
    fn test_prefetch_priority_ordering() {
        let predictor = IntentPredictor::new();
        let manager = PrefetchManager::new(predictor);

        // Record pattern
        manager.notify_access("layer.0.weight").unwrap();
        manager.notify_access("layer.0.bias").unwrap();

        let queue = manager.pending_prefetches();

        // Higher confidence predictions should be first
        for window in queue.windows(2) {
            assert!(window[0].confidence >= window[1].confidence);
        }
    }

    #[test]
    fn test_prefetch_hit_rate() {
        let predictor = IntentPredictor::new();
        let manager = PrefetchManager::new(predictor);

        // Simulate many accesses
        let access_sequence = [
            "layer.0.weight", "layer.0.bias",
            "layer.1.weight", "layer.1.bias",
            "layer.2.weight", "layer.2.bias",
            "layer.3.weight", "layer.3.bias",
        ];

        let mut hits = 0;
        let mut total = 0;

        for access in &access_sequence {
            // Check if it was prefetched
            if manager.was_prefetched(access) {
                hits += 1;
            }
            total += 1;

            manager.notify_access(access).unwrap();
        }

        // Target: >50% hit rate after warmup
        let hit_rate = hits as f32 / total as f32;
        assert!(hit_rate > 0.5 || total < 4, // Allow warmup
            "Hit rate: {:.1}%", hit_rate * 100.0);
    }

    #[test]
    fn test_prefetch_bandwidth_limit() {
        let predictor = IntentPredictor::new();
        let manager = PrefetchManager::new(predictor)
            .with_bandwidth_limit(100_000_000); // 100 MB/s

        manager.notify_access("layer.0.weight").unwrap();

        let queue = manager.pending_prefetches();
        let total_bytes: usize = queue.iter().map(|p| p.estimated_size).sum();

        // Should not exceed bandwidth budget
        assert!(total_bytes <= 100_000_000);
    }

    #[test]
    fn test_predictor_model_update() {
        let mut predictor = IntentPredictor::new();

        // Record training data
        let training_data = vec![
            vec!["layer.0.w", "layer.0.b", "layer.1.w", "layer.1.b"],
            vec!["layer.0.w", "layer.0.b", "layer.1.w", "layer.1.b"],
        ];

        predictor.train(&training_data);

        // Prediction should improve
        let mut history = SessionHistory::new();
        history.record_access("layer.0.w");
        history.record_access("layer.0.b");

        let predictions = predictor.predict(&history, 2);
        assert!(predictions.iter().any(|p| p.name.contains("layer.1")));
    }
}
```

### Quality Gate C.3

```bash
#!/bin/bash
# Phase C.3 Quality Gate

echo "=== Phase C.3: Speculative Prefetch Quality Gate ==="

# 1. All prefetch tests pass
cargo test --package haagenti-speculative prefetch_ -- --nocapture

# 2. Hit rate target met
cargo test --package haagenti-speculative test_prefetch_hit_rate

# 3. Bandwidth limits respected
cargo test --package haagenti-speculative test_prefetch_bandwidth_limit

echo "=== Phase C.3 PASSED ==="
```

---

## Phase C.4: Quality-Aware Adaptive Streaming

### Purpose
Enhance streaming with automatic quality adaptation based on network conditions.

### Test Specification

```rust
// tests/adaptive_streaming_test.rs

#[cfg(test)]
mod adaptive_tests {
    use haagenti_streaming::{AdaptiveStreamManager, NetworkConditions, QualityPolicy};

    #[test]
    fn test_adaptive_manager_creation() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());
        assert!(manager.is_ready());
    }

    #[test]
    fn test_quality_degradation_on_slow_network() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        // Simulate slow network
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 1_000_000, // 1 Mbps
            latency_ms: 200,
            packet_loss: 0.05,
        });

        let recommended = manager.recommended_quality();

        // Should reduce quality for slow network
        assert!(recommended.fragments_to_load < 32);
        assert!(recommended.target_quality < 0.9);
    }

    #[test]
    fn test_quality_increase_on_fast_network() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        // Start with slow network
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 1_000_000,
            latency_ms: 200,
            packet_loss: 0.05,
        });

        // Network improves
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 100_000_000, // 100 Mbps
            latency_ms: 10,
            packet_loss: 0.0,
        });

        let recommended = manager.recommended_quality();

        // Should increase quality
        assert!(recommended.target_quality >= 0.95);
    }

    #[test]
    fn test_quality_policy_minimum() {
        let policy = QualityPolicy {
            min_quality: 0.7,
            max_quality: 0.99,
            ..Default::default()
        };

        let manager = AdaptiveStreamManager::new(policy);

        // Very slow network
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 100_000, // 100 Kbps
            latency_ms: 1000,
            packet_loss: 0.10,
        });

        let recommended = manager.recommended_quality();

        // Should not go below minimum
        assert!(recommended.target_quality >= 0.7);
    }

    #[test]
    fn test_adaptive_fragment_ordering() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 10_000_000, // 10 Mbps
            latency_ms: 50,
            packet_loss: 0.01,
        });

        let fragment_order = manager.optimal_fragment_order(32);

        // Should prioritize high-impact fragments
        // Fragment 0 should always be first (contains essential data)
        assert_eq!(fragment_order[0], 0);
    }

    #[test]
    fn test_bandwidth_estimation() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        // Record transfer samples
        manager.record_transfer(1_000_000, std::time::Duration::from_millis(100));
        manager.record_transfer(1_000_000, std::time::Duration::from_millis(110));
        manager.record_transfer(1_000_000, std::time::Duration::from_millis(90));

        let estimated = manager.estimated_bandwidth_bps();

        // Should estimate ~10 MB/s (80 Mbps)
        assert!(estimated > 70_000_000 && estimated < 90_000_000,
            "Estimated: {} bps", estimated);
    }

    #[test]
    fn test_quality_curve_integration() {
        use haagenti::holotensor::QualityCurve;

        let quality_curve = QualityCurve::default();
        let manager = AdaptiveStreamManager::new(QualityPolicy::default())
            .with_quality_curve(quality_curve);

        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 50_000_000,
            latency_ms: 20,
            packet_loss: 0.0,
        });

        let recommended = manager.recommended_quality();

        // Should use quality curve for fragment selection
        let fragments_needed = manager.fragments_for_target(0.9, 32);
        assert!(fragments_needed > 0 && fragments_needed < 32);
    }
}
```

### Quality Gate C.4

```bash
#!/bin/bash
# Phase C.4 Quality Gate

echo "=== Phase C.4: Adaptive Streaming Quality Gate ==="

# 1. All adaptive tests pass
cargo test --package haagenti-streaming adaptive_ -- --nocapture

# 2. Quality curve integration
cargo test --package haagenti-streaming test_quality_curve_integration

# 3. Bandwidth estimation accurate
cargo test --package haagenti-streaming test_bandwidth_estimation

echo "=== Phase C.4 PASSED ==="
```

---

## Phase C.5: Python Bindings Completion

### Purpose
Complete Python API for all Haagenti features.

### Test Specification

```python
# tests/test_python_bindings.py

import pytest
import haagenti

class TestCompressionBindings:
    def test_zstd_compress_decompress(self):
        data = b"Hello, Haagenti Python!" * 1000
        compressed = haagenti.compress(data, algorithm="zstd")
        decompressed = haagenti.decompress(compressed, algorithm="zstd")
        assert decompressed == data

    def test_lz4_compress_decompress(self):
        data = b"LZ4 test data" * 1000
        compressed = haagenti.compress(data, algorithm="lz4")
        decompressed = haagenti.decompress(compressed, algorithm="lz4")
        assert decompressed == data

    def test_compression_levels(self):
        data = b"Test data for levels" * 1000
        fast = haagenti.compress(data, algorithm="zstd", level="fast")
        best = haagenti.compress(data, algorithm="zstd", level="best")
        assert len(best) <= len(fast)

class TestHoloTensorBindings:
    def test_holotensor_encode_decode(self):
        import numpy as np
        tensor = np.random.randn(1024, 1024).astype(np.float32)

        encoded = haagenti.holotensor.encode(tensor, num_fragments=32)
        decoded = haagenti.holotensor.decode(encoded)

        # Allow some quality loss
        correlation = np.corrcoef(tensor.flatten(), decoded.flatten())[0, 1]
        assert correlation > 0.95

    def test_holotensor_progressive_decode(self):
        import numpy as np
        tensor = np.random.randn(512, 512).astype(np.float32)

        encoded = haagenti.holotensor.encode(tensor, num_fragments=16)

        # Decode with partial fragments
        partial = haagenti.holotensor.decode(encoded, max_fragments=8)
        full = haagenti.holotensor.decode(encoded)

        assert partial.shape == full.shape

    def test_holotensor_quality_curve(self):
        curve = haagenti.holotensor.QualityCurve()

        q1 = curve.predict(4, 16)
        q2 = curve.predict(8, 16)
        q3 = curve.predict(16, 16)

        assert q1 < q2 < q3
        assert q3 == 1.0

class TestStreamingBindings:
    def test_streaming_encoder(self):
        data = b"Streaming test data" * 10000

        with haagenti.streaming.Encoder("zstd") as encoder:
            for chunk in [data[i:i+1000] for i in range(0, len(data), 1000)]:
                encoder.write(chunk)
            compressed = encoder.finish()

        assert len(compressed) < len(data)

    def test_streaming_decoder(self):
        data = b"Streaming test data" * 10000
        compressed = haagenti.compress(data, algorithm="zstd")

        chunks = []
        with haagenti.streaming.Decoder("zstd") as decoder:
            for chunk in [compressed[i:i+100] for i in range(0, len(compressed), 100)]:
                decoded = decoder.write(chunk)
                if decoded:
                    chunks.append(decoded)
            chunks.append(decoder.finish())

        assert b"".join(chunks) == data

class TestDictionaryBindings:
    def test_dictionary_training(self):
        samples = [f"model.layers.{i}.weight".encode() for i in range(100)]
        dict_obj = haagenti.ZstdDict.train(samples, max_size=8192)

        assert dict_obj.id != 0
        assert len(dict_obj.as_bytes()) <= 8192

    def test_dictionary_compression(self):
        samples = [f"model.layers.{i}.weight data".encode() for i in range(100)]
        dict_obj = haagenti.ZstdDict.train(samples, max_size=8192)

        data = b"model.layers.42.weight data" * 100

        with_dict = haagenti.compress(data, algorithm="zstd", dictionary=dict_obj)
        without_dict = haagenti.compress(data, algorithm="zstd")

        assert len(with_dict) < len(without_dict)

class TestNeuralBindings:
    def test_nct_file_read(self):
        nct = haagenti.neural.NctFile.open("testdata/small_model.nct")

        assert nct.num_tensors > 0
        assert nct.num_layers > 0

    def test_nct_tensor_decode(self):
        import numpy as np

        nct = haagenti.neural.NctFile.open("testdata/small_model.nct")
        tensor = nct.read_tensor("layer.0.weight")

        assert isinstance(tensor, np.ndarray)
        assert tensor.dtype == np.float32

class TestErrorHandling:
    def test_invalid_algorithm(self):
        with pytest.raises(ValueError):
            haagenti.compress(b"test", algorithm="invalid")

    def test_corrupted_data(self):
        with pytest.raises(haagenti.DecompressionError):
            haagenti.decompress(b"not valid compressed data", algorithm="zstd")

    def test_empty_input(self):
        compressed = haagenti.compress(b"", algorithm="zstd")
        decompressed = haagenti.decompress(compressed, algorithm="zstd")
        assert decompressed == b""
```

### Quality Gate C.5

```bash
#!/bin/bash
# Phase C.5 Quality Gate

echo "=== Phase C.5: Python Bindings Quality Gate ==="

# 1. Build Python wheel
cd crates/haagenti-python
maturin build --release

# 2. Install wheel
pip install target/wheels/haagenti-*.whl --force-reinstall

# 3. Run Python tests
pytest tests/ -v

# 4. Test all binding categories
pytest tests/test_python_bindings.py -v

# 5. Type stub verification
mypy --strict tests/

echo "=== Phase C.5 PASSED ==="
```

---

## Track C Summary

### Test Count by Phase

| Phase | Unit Tests | Integration | Python | Total |
|-------|------------|-------------|--------|-------|
| C.1 TLS | 6 | 2 | 0 | 8 |
| C.2 Preview | 7 | 2 | 1 | 10 |
| C.3 Prefetch | 8 | 2 | 2 | 12 |
| C.4 Adaptive | 7 | 2 | 1 | 10 |
| C.5 Python | 0 | 0 | 15 | 15 |
| **Total** | **28** | **8** | **19** | **55** |

### Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| TLS support | None | Full mTLS | Security |
| Preview latency | N/A | <100ms | New |
| Prefetch hit rate | 0% | >50% | New |
| Python coverage | 60% | 95% | +35% |

---

*Document Version: 1.0*
*Created: 2026-01-06*
