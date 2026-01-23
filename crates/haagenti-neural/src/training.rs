//! Codebook training for neural compression

use crate::{
    Codebook, CodebookConfig, CodebookStats, LayerCodebook, LayerType, NeuralError, Result,
};
use serde::{Deserialize, Serialize};

/// Configuration for codebook training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum training iterations
    pub max_iterations: usize,
    /// Convergence threshold (change in MSE)
    pub convergence_threshold: f32,
    /// Mini-batch size
    pub batch_size: usize,
    /// Learning rate for centroid updates
    pub learning_rate: f32,
    /// K-means++ initialization
    pub use_kmeans_init: bool,
    /// Number of random restarts
    pub num_restarts: usize,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-5,
            batch_size: 4096,
            learning_rate: 0.01,
            use_kmeans_init: true,
            num_restarts: 3,
            seed: None,
        }
    }
}

/// Codebook trainer using K-means clustering
pub struct CodebookTrainer {
    config: TrainingConfig,
}

impl CodebookTrainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Train a codebook on data
    pub fn train(&self, codebook_config: CodebookConfig, data: &[f32]) -> Result<Codebook> {
        let dim = codebook_config.centroid_dim;
        let num_vectors = data.len() / dim;

        if !data.len().is_multiple_of(dim) {
            return Err(NeuralError::TrainingError(format!(
                "Data length {} not divisible by centroid dim {}",
                data.len(),
                dim
            )));
        }

        if num_vectors < codebook_config.num_centroids {
            return Err(NeuralError::TrainingError(format!(
                "Need at least {} vectors, got {}",
                codebook_config.num_centroids, num_vectors
            )));
        }

        let mut best_codebook = None;
        let mut best_mse = f32::INFINITY;

        for restart in 0..self.config.num_restarts {
            let seed = self.config.seed.map(|s| s + restart as u64);
            let result = self.train_single(&codebook_config, data, seed)?;

            if result.1 < best_mse {
                best_mse = result.1;
                best_codebook = Some(result.0);
            }
        }

        best_codebook.ok_or_else(|| NeuralError::TrainingError("Training failed".into()))
    }

    /// Single training run
    fn train_single(
        &self,
        config: &CodebookConfig,
        data: &[f32],
        seed: Option<u64>,
    ) -> Result<(Codebook, f32)> {
        let dim = config.centroid_dim;
        let num_vectors = data.len() / dim;

        // Initialize centroids
        let mut centroids = if self.config.use_kmeans_init {
            self.kmeans_pp_init(config.num_centroids, data, dim, seed)
        } else {
            self.random_init(config.num_centroids, data, dim, seed)
        };

        let mut prev_mse = f32::INFINITY;
        let mut iterations = 0;

        // K-means iterations
        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Assignment step
            let assignments = self.assign_clusters(&centroids, data, dim);

            // Update step
            let new_centroids =
                self.update_centroids(&assignments, data, config.num_centroids, dim);
            centroids = new_centroids;

            // Compute MSE
            let mse = self.compute_mse(&centroids, &assignments, data, dim);

            // Check convergence
            if (prev_mse - mse).abs() < self.config.convergence_threshold {
                break;
            }
            prev_mse = mse;
        }

        // Compute final statistics
        let assignments = self.assign_clusters(&centroids, data, dim);
        let final_mse = self.compute_mse(&centroids, &assignments, data, dim);
        let usage = self.compute_usage(&assignments, config.num_centroids);

        let stats = CodebookStats {
            mse: final_mse,
            iterations,
            usage_distribution: usage,
            samples_seen: num_vectors,
        };

        let mut codebook = Codebook::from_centroids(config.clone(), centroids, "trained")?;
        codebook.stats = stats;

        Ok((codebook, final_mse))
    }

    /// K-means++ initialization
    fn kmeans_pp_init(&self, k: usize, data: &[f32], dim: usize, seed: Option<u64>) -> Vec<f32> {
        use rand::prelude::*;

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let num_vectors = data.len() / dim;
        let mut centroids = Vec::with_capacity(k * dim);
        let mut distances = vec![f32::INFINITY; num_vectors];

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..num_vectors);
        centroids.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

        // Choose remaining centroids
        for _ in 1..k {
            // Update distances to nearest centroid
            for i in 0..num_vectors {
                let vec = &data[i * dim..(i + 1) * dim];
                let last_centroid = &centroids[centroids.len() - dim..];
                let dist = self.squared_distance(vec, last_centroid);
                distances[i] = distances[i].min(dist);
            }

            // Sample proportional to squared distance
            let total: f32 = distances.iter().sum();
            let threshold = rng.gen::<f32>() * total;

            let mut cumsum = 0.0;
            let mut chosen = 0;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }

            centroids.extend_from_slice(&data[chosen * dim..(chosen + 1) * dim]);
        }

        centroids
    }

    /// Random initialization
    fn random_init(&self, k: usize, data: &[f32], dim: usize, seed: Option<u64>) -> Vec<f32> {
        use rand::prelude::*;

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let num_vectors = data.len() / dim;
        let mut centroids = Vec::with_capacity(k * dim);

        let indices: Vec<usize> = (0..num_vectors).choose_multiple(&mut rng, k);
        for idx in indices {
            centroids.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
        }

        centroids
    }

    /// Assign each vector to nearest centroid
    fn assign_clusters(&self, centroids: &[f32], data: &[f32], dim: usize) -> Vec<usize> {
        let num_vectors = data.len() / dim;
        let num_centroids = centroids.len() / dim;

        (0..num_vectors)
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
                let mut best_idx = 0;
                let mut best_dist = f32::INFINITY;

                for j in 0..num_centroids {
                    let centroid = &centroids[j * dim..(j + 1) * dim];
                    let dist = self.squared_distance(vec, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = j;
                    }
                }

                best_idx
            })
            .collect()
    }

    /// Update centroids based on assignments
    fn update_centroids(
        &self,
        assignments: &[usize],
        data: &[f32],
        k: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];

        for (i, &cluster) in assignments.iter().enumerate() {
            counts[cluster] += 1;
            for d in 0..dim {
                sums[cluster * dim + d] += data[i * dim + d];
            }
        }

        // Compute means
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    sums[j * dim + d] /= counts[j] as f32;
                }
            }
        }

        sums
    }

    /// Compute MSE
    fn compute_mse(
        &self,
        centroids: &[f32],
        assignments: &[usize],
        data: &[f32],
        dim: usize,
    ) -> f32 {
        let total_dist: f32 = assignments
            .iter()
            .enumerate()
            .map(|(i, &cluster)| {
                let vec = &data[i * dim..(i + 1) * dim];
                let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
                self.squared_distance(vec, centroid)
            })
            .sum();

        total_dist / assignments.len() as f32
    }

    /// Compute usage distribution
    fn compute_usage(&self, assignments: &[usize], k: usize) -> (f32, f32, f32) {
        let mut counts = vec![0usize; k];
        for &cluster in assignments {
            counts[cluster] += 1;
        }

        let min = *counts.iter().min().unwrap_or(&0) as f32;
        let max = *counts.iter().max().unwrap_or(&0) as f32;
        let mean = counts.iter().sum::<usize>() as f32 / k as f32;

        (min, max, mean)
    }

    /// Squared L2 distance
    fn squared_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// Train codebooks for all layer types
    pub fn train_all(
        &self,
        model_id: &str,
        data_by_layer: &[(LayerType, Vec<f32>)],
    ) -> Result<LayerCodebook> {
        let mut layer_codebook = LayerCodebook::new(model_id);

        for (layer_type, data) in data_by_layer {
            let config = layer_type.default_config();
            let codebook = self.train(config, data)?;
            layer_codebook.add(*layer_type, codebook);
        }

        Ok(layer_codebook)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer() {
        let config = TrainingConfig {
            max_iterations: 10,
            num_restarts: 1,
            ..Default::default()
        };

        let trainer = CodebookTrainer::new(config);

        // Create simple test data
        let codebook_config = CodebookConfig {
            num_centroids: 4,
            centroid_dim: 2,
            index_bits: 2,
            product_quantization: false,
            pq_subspaces: 1,
        };

        // 4 clusters of 2D points
        let data: Vec<f32> = vec![
            // Cluster 0
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, // Cluster 1
            1.0, 0.0, 1.1, 0.1, 0.9, -0.1, // Cluster 2
            0.0, 1.0, 0.1, 1.1, -0.1, 0.9, // Cluster 3
            1.0, 1.0, 1.1, 1.1, 0.9, 0.9,
        ];

        let codebook = trainer.train(codebook_config, &data).unwrap();

        assert_eq!(codebook.config.num_centroids, 4);
        assert!(codebook.stats.mse < 0.1);
    }

    #[test]
    fn test_kmeans_pp() {
        let trainer = CodebookTrainer::new(TrainingConfig::default());

        let data: Vec<f32> = (0..200).map(|i| (i as f32 / 10.0).sin()).collect();

        let centroids = trainer.kmeans_pp_init(10, &data, 2, Some(42));
        assert_eq!(centroids.len(), 20); // 10 centroids × 2 dimensions
    }
}
