//! Error types for gpu-accelerator

use std::time::Duration;
use thiserror::Error;

/// GPU accelerator errors
#[derive(Error, Debug)]
pub enum GPUError {
    /// Out of memory error
    #[error("Out of memory: requested {0} bytes")]
    OutOfMemory(usize),

    /// Invalid graph structure
    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    /// Kernel not found
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Kernel launch failed
    #[error("Kernel launch failed: {0}")]
    LaunchFailed(String),

    /// DPX not available on this GPU
    #[error("DPX instructions not available on this GPU")]
    DPXNotAvailable,

    /// DPX initialization failed
    #[error("DPX initialization failed")]
    DPXInitializationFailed,

    /// DPX execution failed
    #[error("DPX execution failed: {0}")]
    DPXExecutionFailed(String),

    /// Memory allocation failed
    #[error("Memory allocation failed")]
    MemoryAllocationFailed,

    /// Memory transfer failed
    #[error("Memory transfer failed")]
    MemoryTransferFailed,

    /// Memory alignment failed
    #[error("Memory alignment failed: requested {0}, got {1}")]
    MemoryAlignmentFailed(usize, usize),

    /// CUDA driver error
    #[error("CUDA driver error: {0}")]
    DriverError(i32),

    /// CUDA runtime error
    #[error("CUDA runtime error: {0}")]
    RuntimeError(String),

    /// Operation timed out
    #[error("Operation timed out after {:?}", _0)]
    Timeout(Duration),

    /// IO error
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for gpu-accelerator operations
pub type Result<T> = std::result::Result<T, GPUError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GPUError::OutOfMemory(1024);
        assert!(err.to_string().contains("1024"));

        let err = GPUError::InvalidGraph("empty graph".to_string());
        assert!(err.to_string().contains("empty graph"));

        let err = GPUError::DPXNotAvailable;
        assert!(err.to_string().contains("DPX"));
    }

    #[test]
    fn test_error_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let gpu_err = GPUError::IO(io_err);

        assert!(gpu_err.source().is_some());
    }
}
