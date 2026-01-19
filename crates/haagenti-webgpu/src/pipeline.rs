//! Compute pipeline management for WebGPU

use crate::{Result, WebGpuError};
use crate::buffer::{BufferUsage, GpuBuffer};
use crate::shader::{ShaderModule, WgslSource};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline label
    pub label: String,
    /// Workgroup size (x, y, z)
    pub workgroup_size: (u32, u32, u32),
    /// Number of bind groups
    pub bind_group_count: u32,
    /// Enable async compilation
    pub async_compile: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            label: "compute_pipeline".into(),
            workgroup_size: (256, 1, 1),
            bind_group_count: 1,
            async_compile: true,
        }
    }
}

/// Binding layout for a buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingLayout {
    /// Binding index
    pub binding: u32,
    /// Buffer type
    pub buffer_type: BufferBindingType,
    /// Visibility (compute, vertex, fragment)
    pub visibility: ShaderStage,
    /// Minimum buffer size
    pub min_size: Option<u64>,
}

/// Buffer binding type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferBindingType {
    /// Uniform buffer
    Uniform,
    /// Read-only storage
    ReadOnlyStorage,
    /// Read-write storage
    Storage,
}

/// Shader stage visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShaderStage {
    /// Vertex shader
    Vertex,
    /// Fragment shader
    Fragment,
    /// Compute shader
    Compute,
    /// All stages
    All,
}

impl ShaderStage {
    /// Convert to WebGPU visibility flags
    pub fn to_webgpu(&self) -> u32 {
        match self {
            ShaderStage::Vertex => 0x1,
            ShaderStage::Fragment => 0x2,
            ShaderStage::Compute => 0x4,
            ShaderStage::All => 0x7,
        }
    }
}

/// Bind group layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindGroupLayout {
    /// Group index
    pub group: u32,
    /// Bindings in this group
    pub bindings: Vec<BindingLayout>,
}

impl BindGroupLayout {
    /// Create a new bind group layout
    pub fn new(group: u32) -> Self {
        Self {
            group,
            bindings: Vec::new(),
        }
    }

    /// Add uniform buffer binding
    pub fn add_uniform(&mut self, binding: u32) -> &mut Self {
        self.bindings.push(BindingLayout {
            binding,
            buffer_type: BufferBindingType::Uniform,
            visibility: ShaderStage::Compute,
            min_size: None,
        });
        self
    }

    /// Add read-only storage buffer binding
    pub fn add_read_storage(&mut self, binding: u32) -> &mut Self {
        self.bindings.push(BindingLayout {
            binding,
            buffer_type: BufferBindingType::ReadOnlyStorage,
            visibility: ShaderStage::Compute,
            min_size: None,
        });
        self
    }

    /// Add read-write storage buffer binding
    pub fn add_storage(&mut self, binding: u32) -> &mut Self {
        self.bindings.push(BindingLayout {
            binding,
            buffer_type: BufferBindingType::Storage,
            visibility: ShaderStage::Compute,
            min_size: None,
        });
        self
    }
}

/// Compute pipeline
#[derive(Debug)]
pub struct ComputePipeline {
    /// Configuration
    pub config: PipelineConfig,
    /// Shader module
    pub shader: ShaderModule,
    /// Entry point name
    pub entry_point: String,
    /// Bind group layouts
    pub bind_group_layouts: Vec<BindGroupLayout>,
    /// Whether pipeline is compiled
    compiled: bool,
    /// Pipeline ID (for caching)
    id: u64,
}

impl ComputePipeline {
    /// Create a new compute pipeline
    pub fn new(
        config: PipelineConfig,
        source: WgslSource,
        entry_point: impl Into<String>,
    ) -> Result<Self> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        let shader = ShaderModule::from_source(source)?;

        Ok(Self {
            config,
            shader,
            entry_point: entry_point.into(),
            bind_group_layouts: Vec::new(),
            compiled: false,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        })
    }

    /// Add a bind group layout
    pub fn with_bind_group(mut self, layout: BindGroupLayout) -> Self {
        self.bind_group_layouts.push(layout);
        self
    }

    /// Get pipeline ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Check if compiled
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Mark as compiled
    pub fn mark_compiled(&mut self) {
        self.compiled = true;
    }

    /// Calculate dispatch dimensions for a given total work size
    pub fn dispatch_size(&self, total_x: u32, total_y: u32, total_z: u32) -> (u32, u32, u32) {
        let (wg_x, wg_y, wg_z) = self.config.workgroup_size;

        let dispatch_x = (total_x + wg_x - 1) / wg_x;
        let dispatch_y = (total_y + wg_y - 1) / wg_y;
        let dispatch_z = (total_z + wg_z - 1) / wg_z;

        (dispatch_x, dispatch_y, dispatch_z)
    }
}

/// Standard pipeline builders for common operations
pub mod builders {
    use super::*;
    use crate::shader::templates;

    /// Build matrix multiplication pipeline
    pub fn matmul() -> Result<ComputePipeline> {
        let config = PipelineConfig {
            label: "matmul".into(),
            workgroup_size: (16, 16, 1),
            bind_group_count: 1,
            async_compile: true,
        };

        let mut layout = BindGroupLayout::new(0);
        layout
            .add_uniform(0)      // Uniforms (M, N, K)
            .add_read_storage(1) // Matrix A
            .add_read_storage(2) // Matrix B
            .add_storage(3);     // Matrix C (output)

        ComputePipeline::new(config, templates::matmul(), "matmul_main")
            .map(|p| p.with_bind_group(layout))
    }

    /// Build element-wise addition pipeline
    pub fn add() -> Result<ComputePipeline> {
        let config = PipelineConfig {
            label: "add".into(),
            workgroup_size: (256, 1, 1),
            bind_group_count: 1,
            async_compile: true,
        };

        let mut layout = BindGroupLayout::new(0);
        layout
            .add_uniform(0)      // Uniforms (size)
            .add_read_storage(1) // Input A
            .add_read_storage(2) // Input B
            .add_storage(3);     // Output C

        ComputePipeline::new(config, templates::add(), "add_main")
            .map(|p| p.with_bind_group(layout))
    }

    /// Build GELU activation pipeline
    pub fn gelu() -> Result<ComputePipeline> {
        let config = PipelineConfig {
            label: "gelu".into(),
            workgroup_size: (256, 1, 1),
            bind_group_count: 1,
            async_compile: true,
        };

        let mut layout = BindGroupLayout::new(0);
        layout
            .add_uniform(0)      // Uniforms (size)
            .add_read_storage(1) // Input
            .add_storage(2);     // Output

        ComputePipeline::new(config, templates::gelu(), "gelu_main")
            .map(|p| p.with_bind_group(layout))
    }

    /// Build softmax pipeline
    pub fn softmax() -> Result<ComputePipeline> {
        let config = PipelineConfig {
            label: "softmax".into(),
            workgroup_size: (256, 1, 1),
            bind_group_count: 1,
            async_compile: true,
        };

        let mut layout = BindGroupLayout::new(0);
        layout
            .add_uniform(0)      // Uniforms (batch_size, seq_len)
            .add_read_storage(1) // Input
            .add_storage(2);     // Output

        ComputePipeline::new(config, templates::softmax(), "softmax_main")
            .map(|p| p.with_bind_group(layout))
    }

    /// Build layer normalization pipeline
    pub fn layer_norm() -> Result<ComputePipeline> {
        let config = PipelineConfig {
            label: "layer_norm".into(),
            workgroup_size: (256, 1, 1),
            bind_group_count: 1,
            async_compile: true,
        };

        let mut layout = BindGroupLayout::new(0);
        layout
            .add_uniform(0)      // Uniforms (batch_size, hidden_size, epsilon)
            .add_read_storage(1) // Input
            .add_read_storage(2) // Gamma
            .add_read_storage(3) // Beta
            .add_storage(4);     // Output

        ComputePipeline::new(config, templates::layer_norm(), "layer_norm_main")
            .map(|p| p.with_bind_group(layout))
    }

    /// Build INT4 dequantization pipeline
    pub fn dequantize_int4() -> Result<ComputePipeline> {
        let config = PipelineConfig {
            label: "dequantize_int4".into(),
            workgroup_size: (256, 1, 1),
            bind_group_count: 1,
            async_compile: true,
        };

        let mut layout = BindGroupLayout::new(0);
        layout
            .add_uniform(0)      // Uniforms (size)
            .add_read_storage(1) // Quantized data
            .add_read_storage(2) // Scales
            .add_storage(3);     // Output

        ComputePipeline::new(config, templates::dequantize_int4(), "dequantize_main")
            .map(|p| p.with_bind_group(layout))
    }
}

/// Pipeline cache for reusing compiled pipelines
#[derive(Debug, Default)]
pub struct PipelineCache {
    /// Cached pipelines by ID
    pipelines: HashMap<u64, ComputePipeline>,
    /// Name to ID mapping
    name_to_id: HashMap<String, u64>,
}

impl PipelineCache {
    /// Create new empty cache
    pub fn new() -> Self {
        Self::default()
    }

    /// Create cache with standard pipelines
    pub fn with_standard() -> Result<Self> {
        let mut cache = Self::new();

        cache.add(builders::matmul()?)?;
        cache.add(builders::add()?)?;
        cache.add(builders::gelu()?)?;
        cache.add(builders::softmax()?)?;
        cache.add(builders::layer_norm()?)?;
        cache.add(builders::dequantize_int4()?)?;

        Ok(cache)
    }

    /// Add a pipeline to the cache
    pub fn add(&mut self, pipeline: ComputePipeline) -> Result<u64> {
        let id = pipeline.id();
        let name = pipeline.config.label.clone();

        self.name_to_id.insert(name, id);
        self.pipelines.insert(id, pipeline);

        Ok(id)
    }

    /// Get pipeline by ID
    pub fn get(&self, id: u64) -> Option<&ComputePipeline> {
        self.pipelines.get(&id)
    }

    /// Get mutable pipeline by ID
    pub fn get_mut(&mut self, id: u64) -> Option<&mut ComputePipeline> {
        self.pipelines.get_mut(&id)
    }

    /// Get pipeline by name
    pub fn get_by_name(&self, name: &str) -> Option<&ComputePipeline> {
        self.name_to_id
            .get(name)
            .and_then(|id| self.pipelines.get(id))
    }

    /// List all pipeline names
    pub fn list_names(&self) -> Vec<&str> {
        self.name_to_id.keys().map(|s| s.as_str()).collect()
    }

    /// Number of cached pipelines
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }
}

/// Execution context for running compute pipelines
#[derive(Debug)]
pub struct ExecutionContext {
    /// Buffer bindings
    bindings: Vec<(u32, u32, u64)>, // (group, binding, buffer_id)
    /// Dispatch dimensions
    dispatch: (u32, u32, u32),
}

impl ExecutionContext {
    /// Create new execution context
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            dispatch: (1, 1, 1),
        }
    }

    /// Bind a buffer
    pub fn bind(&mut self, group: u32, binding: u32, buffer: &GpuBuffer) -> &mut Self {
        self.bindings.push((group, binding, buffer.id));
        self
    }

    /// Set dispatch dimensions
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.dispatch = (x, y, z);
        self
    }

    /// Get bindings
    pub fn bindings(&self) -> &[(u32, u32, u64)] {
        &self.bindings
    }

    /// Get dispatch dimensions
    pub fn dispatch_dims(&self) -> (u32, u32, u32) {
        self.dispatch
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = builders::matmul().unwrap();
        assert_eq!(pipeline.config.label, "matmul");
        assert_eq!(pipeline.config.workgroup_size, (16, 16, 1));
        assert!(!pipeline.is_compiled());
    }

    #[test]
    fn test_dispatch_calculation() {
        let pipeline = builders::matmul().unwrap();

        // 1024x1024 matrix with 16x16 workgroup
        let (dx, dy, dz) = pipeline.dispatch_size(1024, 1024, 1);
        assert_eq!(dx, 64);
        assert_eq!(dy, 64);
        assert_eq!(dz, 1);

        // Non-aligned size
        let (dx, dy, dz) = pipeline.dispatch_size(1000, 1000, 1);
        assert_eq!(dx, 63); // ceil(1000/16)
        assert_eq!(dy, 63);
        assert_eq!(dz, 1);
    }

    #[test]
    fn test_pipeline_cache() {
        let cache = PipelineCache::with_standard().unwrap();

        assert!(cache.get_by_name("matmul").is_some());
        assert!(cache.get_by_name("gelu").is_some());
        assert!(cache.get_by_name("softmax").is_some());
        assert!(cache.get_by_name("nonexistent").is_none());

        assert_eq!(cache.len(), 6);
    }

    #[test]
    fn test_bind_group_layout() {
        let mut layout = BindGroupLayout::new(0);
        layout
            .add_uniform(0)
            .add_read_storage(1)
            .add_storage(2);

        assert_eq!(layout.bindings.len(), 3);
        assert_eq!(layout.bindings[0].buffer_type, BufferBindingType::Uniform);
        assert_eq!(layout.bindings[1].buffer_type, BufferBindingType::ReadOnlyStorage);
        assert_eq!(layout.bindings[2].buffer_type, BufferBindingType::Storage);
    }

    #[test]
    fn test_execution_context() {
        let buffer = GpuBuffer::storage(1024, "test");
        let mut ctx = ExecutionContext::new();

        ctx.bind(0, 0, &buffer)
           .dispatch(64, 64, 1);

        assert_eq!(ctx.bindings().len(), 1);
        assert_eq!(ctx.dispatch_dims(), (64, 64, 1));
    }
}
