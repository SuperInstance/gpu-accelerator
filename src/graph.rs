//! CUDA Graph abstraction for constant-time kernel launch

use crate::error::{GPUError, Result};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use std::usize;

/// CUDA Graph for constant-time kernel launch
#[derive(Debug, Clone)]
pub struct CudaGraph {
    name: String,
    kernels: Vec<Kernel>,
    captured: bool,
    parameters: HashMap<String, f32>,
    metadata: HashMap<String, String>,
    timeout: Option<Duration>,
    graph_id: usize,
}

static NEXT_GRAPH_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Kernel in CUDA Graph
#[derive(Debug, Clone)]
pub struct Kernel {
    name: String,
    block_size: u32,
    grid_size: u32,
    dependencies: HashSet<String>,
}

impl CudaGraph {
    /// Create new CUDA Graph
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            kernels: Vec::new(),
            captured: false,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            timeout: None,
            graph_id: NEXT_GRAPH_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        }
    }

    /// Get graph name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get graph ID
    pub fn id(&self) -> usize {
        self.graph_id
    }

    /// Check if graph is captured
    pub fn is_captured(&self) -> bool {
        self.captured
    }

    /// Get kernel count
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Check if has kernel
    pub fn has_kernel(&self, name: &str) -> bool {
        self.kernels.iter().any(|k| k.name == name)
    }

    /// Add kernel to graph
    pub fn add_kernel(&mut self, name: &str, block_size: u32, grid_size: u32) {
        self.kernels.push(Kernel {
            name: name.to_string(),
            block_size,
            grid_size,
            dependencies: HashSet::new(),
        });
    }

    /// Add dependency between kernels
    pub fn add_dependency(&mut self, kernel: &str, depends_on: &str) {
        if let Some(k) = self.kernels.iter_mut().find(|k| k.name == kernel) {
            k.dependencies.insert(depends_on.to_string());
        }
    }

    /// Add parameter
    pub fn add_parameter(&mut self, name: &str, value: f32) {
        self.parameters.insert(name.to_string(), value);
    }

    /// Get parameter
    pub fn get_parameter<T: From<f32>>(&self, name: &str) -> Option<T> {
        self.parameters.get(name).map(|&v| v.into())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<String> {
        self.metadata.get(key).cloned()
    }

    /// Set timeout
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = Some(timeout);
    }

    /// Get timeout
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Validate graph structure
    pub fn validate(&self) -> Result<()> {
        if self.kernels.is_empty() {
            return Err(GPUError::InvalidGraph("empty graph".to_string()));
        }

        // Check for circular dependencies
        for kernel in &self.kernels {
            self.check_circular(&kernel.name, &mut HashSet::new())?;
        }

        Ok(())
    }

    fn check_circular(&self, name: &str, visited: &mut HashSet<String>) -> Result<()> {
        if visited.contains(name) {
            return Err(GPUError::InvalidGraph(format!("circular dependency at {}", name)));
        }

        visited.insert(name.to_string());

        if let Some(kernel) = self.kernels.iter().find(|k| k.name == name) {
            for dep in &kernel.dependencies {
                self.check_circular(dep, visited)?;
            }
        }

        visited.remove(name);
        Ok(())
    }

    /// Get topological order of kernels
    pub fn topological_order(&self) -> Result<Vec<String>> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();

        for kernel in &self.kernels {
            self.topological_visit(&kernel.name, &mut visited, &mut order)?;
        }

        Ok(order)
    }

    fn topological_visit(
        &self,
        name: &str,
        visited: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) -> Result<()> {
        if visited.contains(name) {
            return Ok(());
        }

        visited.insert(name.to_string());

        if let Some(kernel) = self.kernels.iter().find(|k| k.name == name) {
            for dep in &kernel.dependencies {
                self.topological_visit(dep, visited, order)?;
            }
        }

        order.push(name.to_string());
        Ok(())
    }

    /// Capture graph (mock)
    pub fn capture_mock(&mut self) -> Result<()> {
        self.validate()?;
        self.captured = true;
        Ok(())
    }

    /// Launch graph (mock)
    pub fn launch(&self, _input: &[f32]) -> Result<Vec<f32>> {
        if !self.captured {
            return Err(GPUError::InvalidGraph("graph not captured".to_string()));
        }

        // Mock execution: return dummy result
        Ok(vec![0.0f32; 1024])
    }

    /// Replay captured graph (mock)
    pub fn replay_mock(&self) -> Result<()> {
        if !self.captured {
            return Err(GPUError::InvalidGraph("graph not captured".to_string()));
        }

        Ok(())
    }

    /// Execute with input (mock)
    pub fn execute_with_input(&self, _input: &[f32]) -> Result<Vec<f32>> {
        self.launch(_input)
    }

    /// Execute graph (mock)
    pub fn execute(&self) -> Result<()> {
        if self.kernels.is_empty() {
            return Err(GPUError::InvalidGraph("no kernels".to_string()));
        }

        Ok(())
    }

    /// Launch kernel (mock)
    pub fn launch_kernel(&self, name: &str, _args: &[f32]) -> Result<()> {
        if !self.has_kernel(name) {
            return Err(GPUError::KernelNotFound(name.to_string()));
        }

        Ok(())
    }

    /// Launch kernels individually (mock)
    pub fn launch_kernels_individually(&self, _args: &[f32]) -> Result<()> {
        for kernel in &self.kernels {
            // Simulate kernel launch overhead
            std::thread::sleep(std::time::Duration::from_micros(10));
        }

        Ok(())
    }

    /// Launch with parameters (mock)
    pub fn launch_with_params_mock(&self, _params: &[f32]) -> Result<()> {
        Ok(())
    }
}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn grid_size(&self) -> u32 {
        self.grid_size
    }

    pub fn dependencies(&self) -> &HashSet<String> {
        &self.dependencies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = CudaGraph::new("test_graph");
        assert_eq!(graph.name(), "test_graph");
        assert!(!graph.is_captured());
        assert_eq!(graph.kernel_count(), 0);
    }

    #[test]
    fn test_kernel_addition() {
        let mut graph = CudaGraph::new("test_graph");
        graph.add_kernel("vad_kernel", 256, 512);

        assert_eq!(graph.kernel_count(), 1);
        assert!(graph.has_kernel("vad_kernel"));
    }

    #[test]
    fn test_graph_validation() {
        let mut graph = CudaGraph::new("test_graph");

        assert!(!graph.validate().is_ok());

        graph.add_kernel("test_kernel", 256, 512);

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_circular_dependency() {
        let mut graph = CudaGraph::new("test_graph");
        graph.add_kernel("a", 256, 512);
        graph.add_kernel("b", 256, 512);
        graph.add_kernel("c", 256, 512);

        graph.add_dependency("a", "b");
        graph.add_dependency("b", "c");
        graph.add_dependency("c", "a");

        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_topological_order() {
        let mut graph = CudaGraph::new("test_graph");
        graph.add_kernel("a", 256, 512);
        graph.add_kernel("b", 256, 512);
        graph.add_kernel("c", 256, 512);

        graph.add_dependency("b", "a");
        graph.add_dependency("c", "b");

        let order = graph.topological_order().unwrap();

        assert_eq!(order[0], "a");
        assert_eq!(order[1], "b");
        assert_eq!(order[2], "c");
    }
}
