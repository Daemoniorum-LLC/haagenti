//! Job coordination for distributed inference

use crate::{
    node::{Node, NodeRegistry, NodeStatus, ShardAssignment},
    partition::{ModelPartition, PartitionStrategy},
    topology::Topology,
    DistributedError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,
    /// Job timeout
    pub job_timeout: Duration,
    /// Retry count for failed shards
    pub retry_count: u32,
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_jobs: 10,
            job_timeout: Duration::from_secs(300),
            retry_count: 3,
            auto_failover: true,
            health_check_interval: Duration::from_secs(10),
        }
    }
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is queued
    Queued,
    /// Job is partitioning
    Partitioning,
    /// Job is running
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
    /// Job was cancelled
    Cancelled,
}

/// Inference job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Job ID
    pub id: String,
    /// Model name
    pub model_name: String,
    /// Status
    pub status: JobStatus,
    /// Partition strategy
    pub strategy: PartitionStrategy,
    /// Created time (unix ms)
    pub created_at: u64,
    /// Started time (unix ms)
    pub started_at: Option<u64>,
    /// Completed time (unix ms)
    pub completed_at: Option<u64>,
    /// Shard assignments
    pub shards: Vec<ShardStatus>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Shard execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardStatus {
    /// Shard ID
    pub shard_id: String,
    /// Assigned node
    pub node_id: String,
    /// Layer range
    pub layer_range: (usize, usize),
    /// Status
    pub status: ShardExecutionStatus,
    /// Retry count
    pub retries: u32,
}

/// Shard execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardExecutionStatus {
    /// Pending assignment
    Pending,
    /// Assigned to node
    Assigned,
    /// Running on node
    Running,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
}

impl Job {
    /// Create new job
    pub fn new(id: impl Into<String>, model_name: impl Into<String>, strategy: PartitionStrategy) -> Self {
        Self {
            id: id.into(),
            model_name: model_name.into(),
            status: JobStatus::Queued,
            strategy,
            created_at: now_ms(),
            started_at: None,
            completed_at: None,
            shards: Vec::new(),
            error: None,
        }
    }

    /// Check if job is terminal
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        )
    }

    /// Get completion percentage
    pub fn progress(&self) -> f64 {
        if self.shards.is_empty() {
            return 0.0;
        }

        let completed = self
            .shards
            .iter()
            .filter(|s| s.status == ShardExecutionStatus::Completed)
            .count();

        completed as f64 / self.shards.len() as f64
    }

    /// Duration in ms
    pub fn duration_ms(&self) -> Option<u64> {
        let start = self.started_at?;
        let end = self.completed_at.unwrap_or_else(now_ms);
        Some(end.saturating_sub(start))
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Distributed inference coordinator
#[derive(Debug)]
pub struct Coordinator {
    /// Configuration
    config: CoordinatorConfig,
    /// Node registry
    nodes: Arc<RwLock<NodeRegistry>>,
    /// Active jobs
    jobs: HashMap<String, Job>,
    /// Job queue
    queue: Vec<String>,
    /// Topology
    topology: Option<Topology>,
}

impl Coordinator {
    /// Create new coordinator
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            config,
            nodes: Arc::new(RwLock::new(NodeRegistry::new())),
            jobs: HashMap::new(),
            queue: Vec::new(),
            topology: None,
        }
    }

    /// Get node registry
    pub fn nodes(&self) -> Arc<RwLock<NodeRegistry>> {
        Arc::clone(&self.nodes)
    }

    /// Set topology
    pub fn set_topology(&mut self, topology: Topology) {
        self.topology = Some(topology);
    }

    /// Submit a new job
    pub async fn submit_job(
        &mut self,
        model_name: impl Into<String>,
        strategy: PartitionStrategy,
    ) -> Result<String> {
        let job_id = format!("job-{}", now_ms());
        let job = Job::new(&job_id, model_name, strategy);

        self.jobs.insert(job_id.clone(), job);
        self.queue.push(job_id.clone());

        // Try to schedule immediately
        self.schedule_pending().await?;

        Ok(job_id)
    }

    /// Schedule pending jobs
    async fn schedule_pending(&mut self) -> Result<()> {
        let running_count = self
            .jobs
            .values()
            .filter(|j| j.status == JobStatus::Running)
            .count();

        if running_count >= self.config.max_concurrent_jobs {
            return Ok(());
        }

        while let Some(job_id) = self.queue.first().cloned() {
            if let Some(job) = self.jobs.get_mut(&job_id) {
                if job.status == JobStatus::Queued {
                    self.start_job(&job_id).await?;
                    self.queue.remove(0);
                }
            }

            if self.jobs.values().filter(|j| j.status == JobStatus::Running).count()
                >= self.config.max_concurrent_jobs
            {
                break;
            }
        }

        Ok(())
    }

    /// Start a job
    async fn start_job(&mut self, job_id: &str) -> Result<()> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| DistributedError::JobFailed {
                job_id: job_id.into(),
                reason: "Job not found".into(),
            })?;

        job.status = JobStatus::Partitioning;
        job.started_at = Some(now_ms());

        // Get available workers
        let nodes = self.nodes.read().await;
        let workers = nodes.ready_workers();

        if workers.is_empty() {
            job.status = JobStatus::Failed;
            job.error = Some("No available workers".into());
            return Err(DistributedError::InsufficientNodes {
                required: 1,
                available: 0,
            });
        }

        // Create partitions based on strategy
        let partition = ModelPartition::create(&job.strategy, workers.len());

        // Assign shards to workers
        for (idx, layer_range) in partition.shards.iter().enumerate() {
            let worker = &workers[idx % workers.len()];
            job.shards.push(ShardStatus {
                shard_id: format!("{}-shard-{}", job_id, idx),
                node_id: worker.id().to_string(),
                layer_range: *layer_range,
                status: ShardExecutionStatus::Assigned,
                retries: 0,
            });
        }

        drop(nodes);

        job.status = JobStatus::Running;
        Ok(())
    }

    /// Get job status
    pub fn get_job(&self, job_id: &str) -> Option<&Job> {
        self.jobs.get(job_id)
    }

    /// Cancel a job
    pub fn cancel_job(&mut self, job_id: &str) -> Result<()> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| DistributedError::JobFailed {
                job_id: job_id.into(),
                reason: "Job not found".into(),
            })?;

        if job.is_terminal() {
            return Err(DistributedError::JobFailed {
                job_id: job_id.into(),
                reason: "Job already terminal".into(),
            });
        }

        job.status = JobStatus::Cancelled;
        job.completed_at = Some(now_ms());

        // Remove from queue
        self.queue.retain(|id| id != job_id);

        Ok(())
    }

    /// Update shard status
    pub fn update_shard(
        &mut self,
        job_id: &str,
        shard_id: &str,
        status: ShardExecutionStatus,
    ) -> Result<()> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| DistributedError::JobFailed {
                job_id: job_id.into(),
                reason: "Job not found".into(),
            })?;

        if let Some(shard) = job.shards.iter_mut().find(|s| s.shard_id == shard_id) {
            shard.status = status;
        }

        // Check if all shards completed
        let all_completed = job
            .shards
            .iter()
            .all(|s| s.status == ShardExecutionStatus::Completed);

        let any_failed = job
            .shards
            .iter()
            .any(|s| s.status == ShardExecutionStatus::Failed && s.retries >= self.config.retry_count);

        if all_completed {
            job.status = JobStatus::Completed;
            job.completed_at = Some(now_ms());
        } else if any_failed {
            job.status = JobStatus::Failed;
            job.completed_at = Some(now_ms());
            job.error = Some("Shard execution failed".into());
        }

        Ok(())
    }

    /// Handle node failure
    pub async fn handle_node_failure(&mut self, node_id: &str) -> Result<()> {
        if !self.config.auto_failover {
            return Ok(());
        }

        // Find affected jobs
        let affected_jobs: Vec<String> = self
            .jobs
            .iter()
            .filter(|(_, job)| {
                job.status == JobStatus::Running
                    && job.shards.iter().any(|s| s.node_id == node_id)
            })
            .map(|(id, _)| id.clone())
            .collect();

        // Reassign shards
        for job_id in affected_jobs {
            self.reassign_shards(&job_id, node_id).await?;
        }

        Ok(())
    }

    /// Reassign shards from failed node
    async fn reassign_shards(&mut self, job_id: &str, failed_node_id: &str) -> Result<()> {
        let nodes = self.nodes.read().await;
        let workers = nodes.ready_workers();

        if workers.is_empty() {
            return Err(DistributedError::InsufficientNodes {
                required: 1,
                available: 0,
            });
        }

        drop(nodes);

        if let Some(job) = self.jobs.get_mut(job_id) {
            for shard in &mut job.shards {
                if shard.node_id == failed_node_id
                    && shard.status != ShardExecutionStatus::Completed
                {
                    if shard.retries < self.config.retry_count {
                        // Find new node (round-robin for simplicity)
                        let nodes = self.nodes.read().await;
                        let workers = nodes.ready_workers();
                        if let Some(new_worker) = workers.first() {
                            shard.node_id = new_worker.id().to_string();
                            shard.status = ShardExecutionStatus::Assigned;
                            shard.retries += 1;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get all jobs
    pub fn all_jobs(&self) -> Vec<&Job> {
        self.jobs.values().collect()
    }

    /// Get running jobs
    pub fn running_jobs(&self) -> Vec<&Job> {
        self.jobs
            .values()
            .filter(|j| j.status == JobStatus::Running)
            .collect()
    }

    /// Cleanup completed jobs older than duration
    pub fn cleanup(&mut self, older_than: Duration) {
        let now = now_ms();
        let threshold_ms = older_than.as_millis() as u64;

        self.jobs.retain(|_, job| {
            if let Some(completed_at) = job.completed_at {
                now - completed_at < threshold_ms
            } else {
                true
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.max_concurrent_jobs, 10);
        assert!(config.auto_failover);
    }

    #[test]
    fn test_job_creation() {
        let job = Job::new("job-1", "llama-7b", PartitionStrategy::TensorParallel { world_size: 4 });

        assert_eq!(job.id, "job-1");
        assert_eq!(job.status, JobStatus::Queued);
        assert!(!job.is_terminal());
    }

    #[test]
    fn test_job_progress() {
        let mut job = Job::new("job-1", "model", PartitionStrategy::TensorParallel { world_size: 2 });

        job.shards.push(ShardStatus {
            shard_id: "s1".into(),
            node_id: "n1".into(),
            layer_range: (0, 16),
            status: ShardExecutionStatus::Completed,
            retries: 0,
        });

        job.shards.push(ShardStatus {
            shard_id: "s2".into(),
            node_id: "n2".into(),
            layer_range: (16, 32),
            status: ShardExecutionStatus::Running,
            retries: 0,
        });

        assert_eq!(job.progress(), 0.5);
    }

    #[test]
    fn test_coordinator_creation() {
        let config = CoordinatorConfig::default();
        let coordinator = Coordinator::new(config);

        assert!(coordinator.all_jobs().is_empty());
    }
}
