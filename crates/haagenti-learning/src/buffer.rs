//! Experience replay buffer for online learning

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Buffer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Maximum buffer size
    pub max_size: usize,
    /// Batch size for sampling
    pub batch_size: usize,
    /// Prioritized replay
    pub prioritized: bool,
    /// Priority exponent (alpha)
    pub priority_alpha: f32,
    /// Importance sampling exponent (beta)
    pub importance_beta: f32,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            batch_size: 32,
            prioritized: false,
            priority_alpha: 0.6,
            importance_beta: 0.4,
            seed: None,
        }
    }
}

/// Single experience entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Input data
    pub input: Vec<f32>,
    /// Target/label
    pub target: Vec<f32>,
    /// Loss value (for prioritized replay)
    pub loss: f32,
    /// Priority (derived from loss)
    pub priority: f32,
    /// Timestamp (for recency weighting)
    pub timestamp: u64,
    /// Custom metadata
    pub metadata: Option<String>,
}

impl Experience {
    /// Create new experience
    pub fn new(input: Vec<f32>, target: Vec<f32>) -> Self {
        Self {
            input,
            target,
            loss: 1.0,
            priority: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            metadata: None,
        }
    }

    /// Update loss and priority
    pub fn update_loss(&mut self, loss: f32, alpha: f32) {
        self.loss = loss;
        self.priority = (loss + 1e-6).powf(alpha);
    }
}

/// Experience replay buffer
#[derive(Debug)]
pub struct ReplayBuffer {
    /// Configuration
    config: BufferConfig,
    /// Buffer storage
    buffer: VecDeque<Experience>,
    /// Random number generator
    rng: StdRng,
    /// Total priority (for prioritized sampling)
    total_priority: f32,
    /// Max priority seen
    max_priority: f32,
}

impl ReplayBuffer {
    /// Create new replay buffer
    pub fn new(config: BufferConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            config,
            buffer: VecDeque::new(),
            rng,
            total_priority: 0.0,
            max_priority: 1.0,
        }
    }

    /// Add experience to buffer
    pub fn add(&mut self, mut experience: Experience) {
        // Set initial priority to max
        experience.priority = self.max_priority;

        // Remove oldest if full
        if self.buffer.len() >= self.config.max_size {
            if let Some(old) = self.buffer.pop_front() {
                self.total_priority -= old.priority;
            }
        }

        self.total_priority += experience.priority;
        self.buffer.push_back(experience);
    }

    /// Add batch of experiences
    pub fn add_batch(&mut self, experiences: Vec<Experience>) {
        for exp in experiences {
            self.add(exp);
        }
    }

    /// Sample a batch of experiences
    pub fn sample(&mut self) -> Vec<(usize, &Experience, f32)> {
        let batch_size = self.config.batch_size.min(self.buffer.len());

        if batch_size == 0 {
            return Vec::new();
        }

        if self.config.prioritized {
            self.sample_prioritized(batch_size)
        } else {
            self.sample_uniform(batch_size)
        }
    }

    /// Uniform random sampling
    fn sample_uniform(&mut self, batch_size: usize) -> Vec<(usize, &Experience, f32)> {
        let n = self.buffer.len();
        let mut indices: Vec<usize> = (0..n).collect();

        // Fisher-Yates shuffle for first batch_size elements
        for i in 0..batch_size {
            let j = self.rng.gen_range(i..n);
            indices.swap(i, j);
        }

        indices
            .iter()
            .take(batch_size)
            .map(|&i| (i, &self.buffer[i], 1.0))
            .collect()
    }

    /// Prioritized sampling
    fn sample_prioritized(&mut self, batch_size: usize) -> Vec<(usize, &Experience, f32)> {
        let n = self.buffer.len();
        let segment_size = self.total_priority / batch_size as f32;

        let mut result = Vec::with_capacity(batch_size);
        let mut sampled = std::collections::HashSet::new();

        for seg in 0..batch_size {
            let target = self.rng.gen::<f32>() * segment_size + seg as f32 * segment_size;

            let mut cumsum = 0.0;
            for (idx, exp) in self.buffer.iter().enumerate() {
                cumsum += exp.priority;
                if cumsum >= target && !sampled.contains(&idx) {
                    // Importance sampling weight
                    let prob = exp.priority / self.total_priority;
                    let weight = (n as f32 * prob).powf(-self.config.importance_beta);

                    result.push((idx, exp, weight));
                    sampled.insert(idx);
                    break;
                }
            }
        }

        // Normalize weights
        if let Some(max_weight) = result.iter().map(|(_, _, w)| *w).reduce(f32::max) {
            result
                .iter()
                .map(|(i, e, w)| (*i, *e, w / max_weight))
                .collect()
        } else {
            result
        }
    }

    /// Update priorities for sampled experiences
    pub fn update_priorities(&mut self, indices: &[usize], losses: &[f32]) {
        for (&idx, &loss) in indices.iter().zip(losses) {
            if idx < self.buffer.len() {
                let old_priority = self.buffer[idx].priority;
                self.buffer[idx].update_loss(loss, self.config.priority_alpha);
                let new_priority = self.buffer[idx].priority;

                self.total_priority += new_priority - old_priority;
                self.max_priority = self.max_priority.max(new_priority);
            }
        }
    }

    /// Current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Is buffer empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Is buffer full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.config.max_size
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.total_priority = 0.0;
        self.max_priority = 1.0;
    }

    /// Get configuration
    pub fn config(&self) -> &BufferConfig {
        &self.config
    }

    /// Get all experiences (for checkpointing)
    pub fn all(&self) -> &VecDeque<Experience> {
        &self.buffer
    }
}

/// Reservoir sampling buffer for streaming data
#[derive(Debug)]
pub struct ReservoirBuffer {
    /// Maximum size
    max_size: usize,
    /// Buffer storage
    buffer: Vec<Experience>,
    /// Total items seen
    seen_count: usize,
    /// RNG
    rng: StdRng,
}

impl ReservoirBuffer {
    /// Create new reservoir buffer
    pub fn new(max_size: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            max_size,
            buffer: Vec::with_capacity(max_size),
            seen_count: 0,
            rng,
        }
    }

    /// Add experience with reservoir sampling
    pub fn add(&mut self, experience: Experience) {
        self.seen_count += 1;

        if self.buffer.len() < self.max_size {
            self.buffer.push(experience);
        } else {
            // Reservoir sampling: replace with probability max_size/seen_count
            let idx = self.rng.gen_range(0..self.seen_count);
            if idx < self.max_size {
                self.buffer[idx] = experience;
            }
        }
    }

    /// Get buffer
    pub fn buffer(&self) -> &[Experience] {
        &self.buffer
    }

    /// Total items seen
    pub fn seen_count(&self) -> usize {
        self.seen_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let config = BufferConfig::default();
        let buffer = ReplayBuffer::new(config);

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_buffer_add() {
        let config = BufferConfig {
            max_size: 5,
            ..Default::default()
        };
        let mut buffer = ReplayBuffer::new(config);

        for i in 0..10 {
            buffer.add(Experience::new(vec![i as f32], vec![0.0]));
        }

        // Should cap at max_size
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_uniform_sampling() {
        let config = BufferConfig {
            max_size: 100,
            batch_size: 10,
            prioritized: false,
            seed: Some(42),
            ..Default::default()
        };
        let mut buffer = ReplayBuffer::new(config);

        for i in 0..100 {
            buffer.add(Experience::new(vec![i as f32], vec![0.0]));
        }

        let samples = buffer.sample();
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_prioritized_sampling() {
        let config = BufferConfig {
            max_size: 100,
            batch_size: 10,
            prioritized: true,
            seed: Some(42),
            ..Default::default()
        };
        let mut buffer = ReplayBuffer::new(config);

        for i in 0..100 {
            let mut exp = Experience::new(vec![i as f32], vec![0.0]);
            exp.update_loss(i as f32, 0.6);
            buffer.add(exp);
        }

        let samples = buffer.sample();
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_reservoir_sampling() {
        let mut reservoir = ReservoirBuffer::new(10, Some(42));

        for i in 0..1000 {
            reservoir.add(Experience::new(vec![i as f32], vec![0.0]));
        }

        assert_eq!(reservoir.buffer().len(), 10);
        assert_eq!(reservoir.seen_count(), 1000);
    }
}
