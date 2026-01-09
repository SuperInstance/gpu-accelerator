# gpu-accelerator Architecture

## Overview

**gpu-accelerator** is a Rust library providing high-performance GPU acceleration for the equilibrium-tokens ecosystem. It wraps CUDA Graphs and DPX instructions to enable sub-millisecond latency for real-time conversational AI.

## Core Principles

1. **Constant-Time Launch**: CUDA Graphs eliminate kernel launch overhead
2. **Hardware Acceleration**: DPX instructions provide 40× speedup for dynamic programming
3. **Type Safety**: Rust's type system prevents GPU memory errors
4. **Testability**: Mock GPU enables CI without hardware

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     equilibrium-tokens                        │
│                  (Constraint Grammar Engine)                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ uses
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      gpu-accelerator                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ CUDA Graphs  │  │ DPX Instrs   │  │ Memory Mgmt  │      │
│  │              │  │              │  │              │      │
│  │ • Capture    │  │ • Min/Max    │  │ • HtoD       │      │
│  │ • Replay     │  │ • Compare    │  │ • DtoH       │      │
│  │ • Constant   │  │ • Path Opt   │  │ • Pooling    │      │
│  │   Time Launch│  │              │  │ • Zero-Copy  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            GPU Engine (High-Level API)              │    │
│  │                                                     │    │
│  │ • run_vad_graph()                                   │    │
│  │ • run_sentiment_graph()                             │    │
│  │ • compute_embeddings()                              │    │
│  │ • compose_equilibrium()                             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ wraps
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    NVIDIA CUDA / cuDNN                       │
│                    (Driver & Runtime)                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ runs on
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU Hardware                              │
│                  (H100, H200, B200)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Structure

### 1. CUDA Graph Abstraction (`src/graph.rs`)

**Purpose**: Wrap CUDA Graphs for constant-time kernel launch

**Key Types**:
```rust
pub struct CudaGraph {
    name: String,
    kernels: Vec<Kernel>,
    captured: bool,
    parameters: HashMap<String, Parameter>,
    metadata: HashMap<String, String>,
}

pub struct Kernel {
    name: String,
    block_size: u32,
    grid_size: u32,
    dependencies: Vec<String>,
}
```

**Key Operations**:
- `capture()`: Capture kernel sequence as graph
- `launch()`: Launch graph with constant-time overhead
- `replay()`: Replay captured graph (faster than rebuild)
- `add_kernel()`: Add kernel to graph
- `add_dependency()`: Specify kernel execution order

**CUDA 13.1 Integration**:
```rust
// Use CUDA 13.1 constant-time launch
#[cfg(feature = "cuda_13")]
impl CudaGraph {
    pub fn launch_constant_time(&self) -> Result<()> {
        // CUDA 13.1 provides constant-time launch for straight-line graphs
        unsafe { cuda_graph_launch_constant_time(self.graph) }
    }
}
```

**Example Usage**:
```rust
let mut graph = CudaGraph::new("sentiment_pipeline");

// Add kernels
graph.add_kernel("vad_kernel", 256, 512);
graph.add_kernel("sentiment_kernel", 256, 512);
graph.add_kernel("dominance_kernel", 128, 256);

// Define dependencies
graph.add_dependency("sentiment_kernel", "vad_kernel");
graph.add_dependency("dominance_kernel", "sentiment_kernel");

// Capture and launch
graph.capture()?;
graph.launch(&input_data)?;

// Replay (faster)
for _ in 0..1000 {
    graph.replay()?;
}
```

---

### 2. DPX Instruction Wrappers (`src/dpx.rs`)

**Purpose**: Hardware-accelerated dynamic programming on H100/H200

**Key Types**:
```rust
pub struct DPXContext {
    device_id: i32,
    initialized: bool,
    dpx_supported: bool,
}

pub enum DPXInstruction {
    Min(i32, i32),
    Max(i32, i32),
    Compare(i32, i32),
    MinMax(Vec<i32>),
}
```

**Key Operations**:
- `min_max()`: Fast min/max using DPX
- `compute_optimal_path()`: DP-accelerated path optimization
- `aggregate_vad_scores()`: VAD score aggregation with rolling window
- `rolling_window_min_max()`: Sliding window min/max

**Performance Characteristics**:
- **Min/Max**: 7× speedup on H100 vs Ampere
- **Path Optimization**: Up to 40× speedup for DP workloads
- **VAD Aggregation**: 10-20× speedup for rolling window

**Example Usage**:
```rust
let context = DPXContext::new()?;
context.initialize()?;

// Fast min/max
let data = vec![1, 5, 2, 8, 3, 9, 4];
let (min, max) = context.min_max(&data)?;

// Optimal path through cost matrix
let cost_matrix = vec![
    vec![1.0, 3.0, 1.0, 5.0],
    vec![2.0, 1.0, 4.0, 3.0],
];
let path = context.compute_optimal_path(&cost_matrix)?;

// VAD score aggregation
let vad_scores = vec![0.8, 0.9, 0.7, 0.95, 0.85];
let aggregated = context.aggregate_vad_scores(&vad_scores)?;
```

---

### 3. Memory Management (`src/memory.rs`)

**Purpose**: GPU memory allocation, transfer, pooling

**Key Types**:
```rust
pub struct MemoryPool {
    total_size: usize,
    used_size: usize,
    buffers: Vec<GPUBuffer>,
    free_list: Vec<GPUBuffer>,
}

pub struct GPUBuffer {
    ptr: CUdeviceptr,
    size: usize,
    alignment: usize,
    is_pinned: bool,
}

pub struct TransferInfo {
    src_size: usize,
    dst_size: usize,
    bytes_to_copy: usize,
    bandwidth_estimate: f64,
}
```

**Key Operations**:
- `allocate()`: Allocate GPU memory
- `allocate_aligned()`: Allocate with alignment
- `htod_transfer()`: Host to Device transfer
- `dtoh_transfer()`: Device to Host transfer
- `compact()`: Defragment memory pool

**Memory Strategies**:
1. **Pooling**: Pre-allocated pool reduces allocation overhead
2. **Pinned Memory**: Page-locked memory for faster transfers
3. **Zero-Copy**: Direct access on supported GPUs (Grace Hopper)
4. **Async Transfers**: Overlap compute and transfer

**Example Usage**:
```rust
// Create memory pool
let pool = MemoryPool::new(1024 * 1024 * 1024)?; // 1GB

// Allocate buffer
let buffer = pool.allocate(1024 * 1024)?;

// HtoD transfer
let cpu_data = vec![1.0f32; 1024 * 1024];
let gpu_buffer = engine.htod_transfer(&cpu_data)?;

// DtoH transfer
let cpu_result = engine.dtoh_transfer(&gpu_buffer)?;

// Zero-copy (Grace Hopper)
let zero_copy_buffer = pool.allocate_zero_copy(1024 * 1024)?;
```

---

### 4. GPU Engine (`src/engine.rs`)

**Purpose**: High-level API for equilibrium-tokens integration

**Key Types**:
```rust
pub struct GPUEngine {
    device_id: i32,
    stream: CUDAStream,
    memory_pool: MemoryPool,
    graph_cache: HashMap<String, CudaGraph>,
}

pub struct SentimentResult {
    pub valence: f32,      // -1.0 (negative) to 1.0 (positive)
    pub arousal: f32,      // 0.0 (calm) to 1.0 (excited)
    pub dominance: f32,    // 0.0 (submissive) to 1.0 (dominant)
}

pub struct VADResult {
    pub speech_probability: f32,  // 0.0 to 1.0
    pub energy_level: f32,        // 0.0 to 1.0
    pub silence_detected: bool,
}
```

**Key Operations**:

#### Sentiment Inference
```rust
pub async fn run_sentiment_graph(&self, audio: &AudioBuffer) -> Result<SentimentResult> {
    // Upload to GPU
    let gpu_audio = self.upload_to_gpu(audio).await?;

    // Execute VAD graph
    let vad = self.run_vad_graph(&gpu_audio).await?;

    // Execute sentiment graph
    let sentiment = self.run_sentiment_graph_internal(&gpu_audio).await?;

    // Execute dominance graph
    let dominance = self.run_dominance_graph(&gpu_audio).await?;

    Ok(SentimentResult {
        valence: sentiment.valence,
        arousal: sentiment.arousal,
        dominance: dominance.dominance,
    })
}
```

#### Embedding Computation
```rust
pub async fn compute_embeddings(&self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
    let gpu_tokens = self.upload_tokens(tokens).await?;

    // Execute embedding kernel
    let gpu_embeddings = self.run_embedding_kernel(&gpu_tokens).await?;

    // Download results
    let cpu_embeddings = self.download_embeddings(&gpu_embeddings).await?;

    Ok(cpu_embeddings)
}
```

#### Equilibrium Composition
```rust
pub fn compose_equilibrium(
    &self,
    rate_weight: f32,
    context_weight: f32,
    sentiment_weight: f32,
    rate_constraint: f32,
    context_constraint: f32,
    sentiment_constraint: f32,
) -> Result<f32> {
    // Multiplicative equilibrium (GPU or CPU)
    let equilibrium = rate_weight * rate_constraint
        + context_weight * context_constraint
        + sentiment_weight * sentiment_constraint;

    Ok(equilibrium)
}
```

---

## Integration Points

### 1. equilibrium-tokens Integration

```rust
// In equilibrium-tokens/src/gpu/mod.rs
use gpu_accelerator::{GPUEngine, SentimentResult, VADResult};

pub struct EquilibriumOrchestrator {
    gpu_engine: GPUEngine,
    // ... other fields
}

impl EquilibriumOrchestrator {
    pub async fn process_audio(&mut self, audio: &[f32]) -> Result<EquilibriumState> {
        // GPU-accelerated sentiment inference
        let sentiment = self.gpu_engine.run_sentiment_graph(audio).await?;

        // GPU-accelerated VAD
        let vad = self.gpu_engine.run_vad_graph(audio).await?;

        // Compose equilibrium
        let equilibrium = self.gpu_engine.compose_equilibrium(
            0.3,  // rate weight
            0.4,  // context weight
            0.3,  // sentiment weight
            self.rate_constraint,
            self.context_constraint,
            sentiment.valence,
        )?;

        Ok(EquilibriumState {
            equilibrium,
            sentiment,
            vad,
        })
    }
}
```

### 2. embeddings-engine Integration

```rust
use gpu_accelerator::GPUEngine;

pub struct EmbeddingsEngine {
    gpu_engine: GPUEngine,
}

impl EmbeddingsEngine {
    pub async fn compute(&self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
        self.gpu_engine.compute_embeddings(tokens).await
    }

    pub async fn similarity(&self, query: &[f32], contexts: &[[f32]]) -> Result<Vec<f32>> {
        self.gpu_engine.compute_similarity(query, contexts).await
    }
}
```

### 3. inference-optimizer Integration (TensorRT)

```rust
use gpu_accelerator::{CudaGraph, GPUEngine};

pub struct TensorRTInference {
    gpu_engine: GPUEngine,
    trt_engine: trt::Engine,
}

impl TensorRTInference {
    pub fn optimize_with_cuda_graphs(&mut self) -> Result<()> {
        // Capture TensorRT inference as CUDA Graph
        let mut graph = CudaGraph::new("tensorrt_inference");

        // Add TensorRT kernels
        for layer in self.trt_engine.layers() {
            graph.add_kernel(layer.name(), layer.block_size(), layer.grid_size());
        }

        // Capture graph
        graph.capture()?;

        // Use for inference
        self.gpu_engine.cache_graph("inference", graph);

        Ok(())
    }
}
```

---

## Error Handling Strategy

### Error Types

```rust
pub enum GPUError {
    // GPU-specific errors
    OutOfMemory(usize),
    InvalidGraph(String),
    KernelNotFound(String),
    LaunchFailed(String),

    // DPX-specific errors
    DPXNotAvailable,
    DPXInitializationFailed,
    DPXExecutionFailed(String),

    // Memory errors
    MemoryAllocationFailed,
    MemoryTransferFailed,
    MemoryAlignmentFailed(usize, usize),

    // Driver/runtime errors
    DriverError(i32),
    RuntimeError(String),

    // Timeout errors
    Timeout(Duration),
}
```

### Error Handling Pattern

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("GPU error: {0}")]
    GPU(#[from] GPUError),

    #[error("I/O error: {0}")]
    IO(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
```

### Error Recovery

```rust
impl GPUEngine {
    pub fn recover(&mut self) -> Result<bool> {
        // Reset GPU state
        self.reset_stream()?;

        // Clear memory pool
        self.memory_pool.clear();

        // Reinitialize
        self.initialize()?;

        Ok(true)
    }

    pub fn is_healthy(&self) -> bool {
        // Check GPU state
        if let Err(_) = self.check_gpu_state() {
            return false;
        }

        // Check memory
        if self.memory_pool.fragmentation_ratio() > 0.5 {
            return false;
        }

        true
    }
}
```

---

## Performance Optimization

### 1. CUDA Graph Optimization

**Constant-Time Launch** (CUDA 13.1):
```rust
// Before: Traditional kernel launch (5-50μs per kernel)
for kernel in &kernels {
    kernel.launch(&args)?;  // 5-50μs overhead
}

// After: CUDA Graph launch (constant-time < 5μs)
graph.launch(&args)?;  // < 5μs total
```

**Graph Reuse**:
```rust
lazy_static! {
    static ref SENTIMENT_GRAPH: CudaGraph = {
        let mut graph = CudaGraph::new("sentiment");
        graph.add_kernel("vad", 256, 512);
        graph.add_kernel("sentiment", 256, 512);
        graph.capture().unwrap();
        graph
    };
}

// Reuse graph (no capture overhead)
SENTIMENT_GRAPH.launch(&args)?;
```

### 2. DPX Optimization

**Vectorized Min/Max**:
```rust
// CPU: O(n) with branch mispredictions
let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

// DPX: O(n/32) with SIMD (40× faster)
let (min, max) = context.min_max(&data)?;
```

**Path Optimization**:
```rust
// CPU DP: O(rows * cols)
for i in 0..rows {
    for j in 0..cols {
        dp[i][j] = cost[i][j] + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
    }
}

// DPX: O(rows * cols / 32) with hardware acceleration
let path = context.compute_optimal_path(&costs)?;
```

### 3. Memory Optimization

**Pinned Memory**:
```rust
// Paged memory (slow)
let cpu_data = vec![1.0f32; 1024];
cudaMemcpy(gpu_ptr, cpu_data.as_ptr(), size, cudaMemcpyDefault);  // ~5 GB/s

// Pinned memory (fast)
let cpu_data = allocate_pinned(1024)?;
cudaMemcpy(gpu_ptr, cpu_data, size, cudaMemcpyDefault);  // ~12 GB/s
```

**Async Transfers**:
```rust
// Overlap compute and transfer
let stream = engine.create_stream()?;

// Async HtoD
stream.htod_transfer_async(cpu_data, gpu_buffer)?;

// Async kernel launch
stream.launch_async(kernel, &args)?;

// Async DtoH
stream.dtoh_transfer_async(gpu_result, cpu_result)?;

stream.synchronize()?;
```

### 4. Kernel Fusion

**Before: Multiple Kernels**
```rust
vad_kernel<<<...>>>(input, vad_output);
sentiment_kernel<<<...>>>(vad_output, sentiment_output);
dominance_kernel<<<...>>>(sentiment_output, dominance_output);
```

**After: Fused Kernel**
```rust
fused_sentiment_kernel<<<...>>>(input, vad_sentiment_dominance_output);
// Single kernel launch, better memory locality
```

---

## Testing Strategy

See `docs/TESTING_STRATEGY.md` for comprehensive testing approach.

### Key Testing Principles

1. **Mock GPU**: Test logic without hardware
2. **CPU Reference**: Validate accuracy against CPU implementation
3. **GPU Validation**: Run GPU tests nightly on hardware
4. **Performance Tracking**: Benchmark on every release

### Test Structure

```
tests/
├── unit_tests.rs           # Unit tests (no GPU)
├── integration_tests.rs    # Integration tests (mocked GPU)
├── gpu_tests.rs            # GPU tests (real GPU, feature-gated)
└── accuracy_tests.rs       # Accuracy validation (vs CPU)

benches/
├── graph_launch.rs         # CUDA Graph launch benchmarks
├── memory_transfer.rs      # HtoD/DtoH bandwidth benchmarks
├── dpx_vs_cpu.rs          # DPX speedup benchmarks
└── sentiment_inference.rs # End-to-end sentiment benchmarks
```

---

## Performance Targets

### Latency Targets

| Operation | Target (P95) | Rationale |
|-----------|-------------|-----------|
| CUDA Graph launch | < 50μs | Constant-time launch |
| VAD inference | < 5ms | Real-time processing |
| Sentiment inference | < 10ms | Sub-100ms response time |
| Full pipeline | < 20ms | < 2ms jitter requirement |
| Memory transfer (1MB) | < 100μs | PCIe bandwidth |

### Throughput Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| DPX min/max vs CPU | 10-40× | DPX acceleration |
| CUDA Graph vs traditional | 10-50× | Launch overhead elimination |
| HtoD bandwidth | > 10 GB/s | PCIe Gen4 x16 |
| DtoH bandwidth | > 10 GB/s | PCIe Gen4 x16 |
| Concurrent users | > 10 | Multi-user support |

### Accuracy Targets

| Operation | Tolerance | Rationale |
|-----------|-----------|-----------|
| Sentiment (GPU vs CPU) | ε < 1e-5 | f32 precision |
| VAD (GPU vs CPU) | ε < 1e-5 | f32 precision |
| Embeddings (GPU vs CPU) | ε < 1e-5 | f32 precision |
| DPX vs standard DP | Exact match | Integer operations |

---

## Future Enhancements

### Short-Term (CUDA 13.1)

- [ ] Tile-based programming model integration
- [ ] Green Context support (reduced context switch overhead)
- [ ] Multi-GPU CUDA Graphs (device-side execution)

### Medium-Term (Blackwell)

- [ ] FP4 quantization support
- [ ] Second-generation Transformer Engine
- [ ] 2x throughput improvement (B200 GPUs)

### Long-Term

- [ ] NVLink 6th Gen integration (Rubin platform)
- [ ] Distributed GPU orchestration
- [ ] Real-time GPU scheduling (GCAPS research)

---

## Conclusion

**gpu-accelerator** provides the hardware acceleration foundation for equilibrium-tokens:

- **CUDA Graphs**: Constant-time launch for < 2ms jitter
- **DPX Instructions**: 40× speedup for dynamic programming
- **Memory Management**: Efficient GPU memory pooling
- **Type Safety**: Rust prevents memory errors
- **Testability**: Mock GPU enables CI without hardware

**The code is ephemeral; the acceleration is eternal.**
