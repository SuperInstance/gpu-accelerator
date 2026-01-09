//! gpu-accelerator: CUDA Graph and DPX instruction wrappers for GPU acceleration
//!
//! This library provides high-performance GPU acceleration for the equilibrium-tokens
//! ecosystem, enabling:
//!
//! - **Constant-time CUDA Graph launch**: 50-90% reduction in kernel launch overhead
//! - **DPX instructions**: 40× acceleration for dynamic programming on H100/H200
//! - **Efficient memory management**: HtoD/DtoH transfers, pooling, zero-copy
//! - **Sub-millisecond latency**: Target < 2ms jitter for rate equilibrium
//!
//! # Example
//!
//! ```rust
//! use gpu_accelerator::{GPUEngine, CudaGraph};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let engine = GPUEngine::new()?;
//!
//!     // Run sentiment inference
//!     let audio = vec![0.1f32, 0.2, 0.3, 0.4];
//!     let sentiment = engine.run_sentiment_graph(&audio).await?;
//!
//!     println!("Valence: {}", sentiment.valence);
//!     println!("Arousal: {}", sentiment.arousal);
//!
//!     Ok(())
//! }
//! ```

pub mod graph;
pub mod dpx;
pub mod memory;
pub mod engine;
pub mod types;
pub mod error;

// Re-exports for convenience
pub use error::{GPUError, Result};
pub use graph::CudaGraph;
pub use dpx::{DPXContext, DPXInstruction};
pub use memory::{GPUBuffer, MemoryPool};
pub use engine::GPUEngine;
pub use types::{AudioBuffer, SentimentResult, VADResult};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum CUDA version required
pub const MIN_CUDA_VERSION: &str = "11.0";

/// Recommended CUDA version
pub const RECOMMENDED_CUDA_VERSION: &str = "12.3";

/// CUDA 13.1 for constant-time launch
pub const CUDA_13_VERSION: &str = "13.1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_cuda_version_constants() {
        assert_eq!(MIN_CUDA_VERSION, "11.0");
        assert_eq!(RECOMMENDED_CUDA_VERSION, "12.3");
        assert_eq!(CUDA_13_VERSION, "13.1");
    }
}
