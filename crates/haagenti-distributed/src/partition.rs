//! Tensor and model partitioning strategies

use serde::{Deserialize, Serialize};

/// Partition strategy for distributed inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Tensor parallelism (split tensors across devices)
    TensorParallel {
        /// Number of devices
        world_size: usize,
    },
    /// Pipeline parallelism (split layers across devices)
    PipelineParallel {
        /// Number of pipeline stages
        num_stages: usize,
        /// Micro-batch size
        micro_batch_size: usize,
    },
    /// Expert parallelism (for MoE models)
    ExpertParallel {
        /// Number of expert groups
        num_expert_groups: usize,
        /// Experts per group
        experts_per_group: usize,
    },
    /// Hybrid parallelism
    Hybrid {
        /// Tensor parallel size
        tp_size: usize,
        /// Pipeline parallel size
        pp_size: usize,
    },
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        Self::TensorParallel { world_size: 1 }
    }
}

/// Tensor partition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorPartition {
    /// Partition ID
    pub id: usize,
    /// Total partitions
    pub world_size: usize,
    /// Dimension to partition
    pub dim: usize,
    /// Start index in dimension
    pub start: usize,
    /// End index in dimension
    pub end: usize,
}

impl TensorPartition {
    /// Create partitions for a tensor dimension
    pub fn create(world_size: usize, dim_size: usize, dim: usize) -> Vec<Self> {
        let chunk_size = dim_size.div_ceil(world_size);

        (0..world_size)
            .map(|i| {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(dim_size);
                Self {
                    id: i,
                    world_size,
                    dim,
                    start,
                    end,
                }
            })
            .collect()
    }

    /// Size of this partition
    pub fn size(&self) -> usize {
        self.end - self.start
    }
}

/// Model partition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPartition {
    /// Strategy used
    pub strategy: PartitionStrategy,
    /// Shard definitions (layer_start, layer_end)
    pub shards: Vec<(usize, usize)>,
    /// Total layers
    pub total_layers: usize,
}

impl ModelPartition {
    /// Create model partition based on strategy
    pub fn create(strategy: &PartitionStrategy, num_workers: usize) -> Self {
        // Default to 32 layers (common transformer size)
        Self::create_for_layers(strategy, num_workers, 32)
    }

    /// Create model partition for specific layer count
    pub fn create_for_layers(
        strategy: &PartitionStrategy,
        num_workers: usize,
        total_layers: usize,
    ) -> Self {
        let shards = match strategy {
            PartitionStrategy::TensorParallel { world_size } => {
                // In tensor parallel, all workers have all layers
                // but tensors are partitioned
                (0..*world_size.min(&num_workers))
                    .map(|_| (0, total_layers))
                    .collect()
            }
            PartitionStrategy::PipelineParallel { num_stages, .. } => {
                let stages = *num_stages.min(&num_workers);
                let layers_per_stage = total_layers.div_ceil(stages);

                (0..stages)
                    .map(|i| {
                        let start = i * layers_per_stage;
                        let end = ((i + 1) * layers_per_stage).min(total_layers);
                        (start, end)
                    })
                    .collect()
            }
            PartitionStrategy::ExpertParallel {
                num_expert_groups,
                experts_per_group,
            } => {
                // Expert parallel: each group handles specific experts
                let total_experts = num_expert_groups * experts_per_group;
                let experts_per_worker = total_experts.div_ceil(num_workers);

                (0..num_workers.min(total_experts))
                    .map(|i| {
                        let start = i * experts_per_worker;
                        let end = ((i + 1) * experts_per_worker).min(total_experts);
                        (start, end)
                    })
                    .collect()
            }
            PartitionStrategy::Hybrid { tp_size, pp_size } => {
                // Hybrid: combine tensor and pipeline parallelism
                let total_workers = tp_size * pp_size;
                let layers_per_stage = total_layers.div_ceil(*pp_size);

                (0..total_workers.min(num_workers))
                    .map(|i| {
                        let pp_rank = i / tp_size;
                        let start = pp_rank * layers_per_stage;
                        let end = ((pp_rank + 1) * layers_per_stage).min(total_layers);
                        (start, end)
                    })
                    .collect()
            }
        };

        Self {
            strategy: strategy.clone(),
            shards,
            total_layers,
        }
    }

    /// Get shard for worker rank
    pub fn shard_for_rank(&self, rank: usize) -> Option<(usize, usize)> {
        self.shards.get(rank).copied()
    }

    /// Number of shards
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}

/// Tensor parallel utilities
pub struct TensorParallel;

impl TensorParallel {
    /// Split weight matrix for column parallelism
    pub fn column_parallel_split<T: Clone>(
        weights: &[T],
        rows: usize,
        cols: usize,
        world_size: usize,
        rank: usize,
    ) -> Vec<T> {
        let cols_per_rank = cols.div_ceil(world_size);
        let start_col = rank * cols_per_rank;
        let end_col = ((rank + 1) * cols_per_rank).min(cols);
        let local_cols = end_col - start_col;

        let mut result = Vec::with_capacity(rows * local_cols);

        for row in 0..rows {
            for col in start_col..end_col {
                result.push(weights[row * cols + col].clone());
            }
        }

        result
    }

    /// Split weight matrix for row parallelism
    pub fn row_parallel_split<T: Clone>(
        weights: &[T],
        rows: usize,
        cols: usize,
        world_size: usize,
        rank: usize,
    ) -> Vec<T> {
        let rows_per_rank = rows.div_ceil(world_size);
        let start_row = rank * rows_per_rank;
        let end_row = ((rank + 1) * rows_per_rank).min(rows);

        let mut result = Vec::with_capacity((end_row - start_row) * cols);

        for row in start_row..end_row {
            for col in 0..cols {
                result.push(weights[row * cols + col].clone());
            }
        }

        result
    }
}

/// Pipeline parallel utilities
pub struct PipelineParallel;

impl PipelineParallel {
    /// Calculate number of micro-batches needed
    pub fn num_micro_batches(batch_size: usize, micro_batch_size: usize) -> usize {
        batch_size.div_ceil(micro_batch_size)
    }

    /// Calculate pipeline bubble overhead
    pub fn bubble_overhead(num_stages: usize, num_micro_batches: usize) -> f64 {
        if num_micro_batches == 0 {
            return 1.0;
        }
        (num_stages - 1) as f64 / num_micro_batches as f64
    }

    /// Calculate optimal micro-batch size to minimize bubble
    pub fn optimal_micro_batch_size(batch_size: usize, num_stages: usize) -> usize {
        // Rule of thumb: micro_batches >= 4 * num_stages for < 25% bubble
        let target_micro_batches = 4 * num_stages;
        let micro_batch_size = batch_size.div_ceil(target_micro_batches);
        micro_batch_size.max(1)
    }
}

/// Expert parallel utilities
pub struct ExpertParallel;

impl ExpertParallel {
    /// Calculate load balance factor (1.0 = perfect balance)
    pub fn load_balance_factor(tokens_per_expert: &[usize]) -> f64 {
        if tokens_per_expert.is_empty() {
            return 1.0;
        }

        let total: usize = tokens_per_expert.iter().sum();
        let avg = total as f64 / tokens_per_expert.len() as f64;
        let max = *tokens_per_expert.iter().max().unwrap_or(&0) as f64;

        if max == 0.0 {
            1.0
        } else {
            avg / max
        }
    }

    /// Calculate capacity factor needed for given imbalance
    pub fn required_capacity_factor(load_balance: f64) -> f64 {
        if load_balance <= 0.0 {
            return 2.0;
        }
        (1.0 / load_balance).clamp(1.0, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_partition() {
        let partitions = TensorPartition::create(4, 1024, 1);

        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].start, 0);
        assert_eq!(partitions[0].end, 256);
        assert_eq!(partitions[3].start, 768);
        assert_eq!(partitions[3].end, 1024);
    }

    #[test]
    fn test_model_partition_tensor_parallel() {
        let strategy = PartitionStrategy::TensorParallel { world_size: 4 };
        let partition = ModelPartition::create_for_layers(&strategy, 4, 32);

        assert_eq!(partition.num_shards(), 4);
        // All shards have all layers in tensor parallel
        for shard in &partition.shards {
            assert_eq!(*shard, (0, 32));
        }
    }

    #[test]
    fn test_model_partition_pipeline_parallel() {
        let strategy = PartitionStrategy::PipelineParallel {
            num_stages: 4,
            micro_batch_size: 2,
        };
        let partition = ModelPartition::create_for_layers(&strategy, 4, 32);

        assert_eq!(partition.num_shards(), 4);
        assert_eq!(partition.shards[0], (0, 8));
        assert_eq!(partition.shards[1], (8, 16));
        assert_eq!(partition.shards[2], (16, 24));
        assert_eq!(partition.shards[3], (24, 32));
    }

    #[test]
    fn test_column_parallel_split() {
        let weights: Vec<f32> = (0..12).map(|i| i as f32).collect(); // 3x4 matrix
        let split = TensorParallel::column_parallel_split(&weights, 3, 4, 2, 0);

        // First half of columns: [0,1], [4,5], [8,9]
        assert_eq!(split.len(), 6);
        assert_eq!(split[0], 0.0);
        assert_eq!(split[1], 1.0);
        assert_eq!(split[2], 4.0);
    }

    #[test]
    fn test_pipeline_bubble() {
        // 4 stages, 16 micro-batches
        let overhead = PipelineParallel::bubble_overhead(4, 16);
        assert!((overhead - 0.1875).abs() < 0.001);
    }

    #[test]
    fn test_expert_load_balance() {
        let balanced = vec![100, 100, 100, 100];
        assert_eq!(ExpertParallel::load_balance_factor(&balanced), 1.0);

        let imbalanced = vec![100, 200, 100, 100];
        assert!((ExpertParallel::load_balance_factor(&imbalanced) - 0.625).abs() < 0.001);
    }
}
