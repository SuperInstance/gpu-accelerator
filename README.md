# gpu-accelerator

CUDA Graph and DPX instruction wrappers for GPU acceleration in equilibrium-tokens.

## Overview

**gpu-accelerator** provides high-performance GPU acceleration for real-time conversational AI, enabling:

- **Constant-time CUDA Graph launch**: 50-90% reduction in kernel launch overhead
- **DPX instructions**: 40× acceleration for dynamic programming on H100/H200
- **Efficient memory management**: HtoD/DtoH transfers, pooling, zero-copy
- **Sub-millisecond latency**: Target < 2ms jitter for rate equilibrium

## Features

### CUDA Graphs

```rust
use gpu_accelerator::CudaGraph;

let mut graph = CudaGraph::new("sentiment_pipeline");
graph.add_kernel("vad_kernel", 256, 512);
graph.add_kernel("sentiment_kernel", 256, 512);
graph.add_dependency("sentiment_kernel", "vad_kernel");

graph.capture()?;
graph.launch(&input_data)?;

// Replay captured graph (constant-time)
for _ in 0..1000 {
    graph.replay()?;
}
```

### DPX Instructions

```rust
use gpu_accelerator::DPXContext;

let context = DPXContext::new()?;
context.initialize()?;

// Fast min/max (7× faster on H100)
let data = vec![1, 5, 2, 8, 3, 9, 4];
let (min, max) = context.min_max(&data)?;

// Optimal path optimization (40× speedup)
let cost_matrix = vec![
    vec![1.0, 3.0, 1.0],
    vec![2.0, 1.0, 4.0],
];
let path = context.compute_optimal_path(&cost_matrix)?;
```

### GPU Engine

```rust
use gpu_accelerator::GPUEngine;

let engine = GPUEngine::new()?;

// Sentiment inference
let sentiment = engine.run_sentiment_graph(&audio).await?;
println!("Valence: {}", sentiment.valence);
println!("Arousal: {}", sentiment.arousal);

// Embedding computation
let embeddings = engine.compute_embeddings(&tokens).await?;

// Equilibrium composition
let equilibrium = engine.compose_equilibrium(
    0.3,  // rate weight
    0.4,  // context weight
    0.3,  // sentiment weight
    rate_constraint,
    context_constraint,
    sentiment_constraint,
)?;
```

## Performance

### CUDA Graph Launch Latency

- **Uncaptured**: 5-50μs per kernel
- **Captured**: < 5μs total (constant-time)
- **Speedup**: 10-50× for multi-kernel graphs

### DPX Acceleration

- **Min/Max operations**: 7× faster on H100 vs Ampere
- **Path optimization**: Up to 40× speedup for DP workloads
- **VAD aggregation**: 10-20× speedup for rolling window

### Memory Bandwidth

- **HtoD**: > 10 GB/s (pinned memory)
- **DtoH**: > 10 GB/s (pinned memory)
- **Zero-copy**: Grace Hopper NVLink-C2C @ 900 GB/s

## Integration

### equilibrium-tokens

```rust
use gpu_accelerator::GPUEngine;

pub struct EquilibriumOrchestrator {
    gpu_engine: GPUEngine,
}

impl EquilibriumOrchestrator {
    pub async fn process_audio(&mut self, audio: &[f32]) -> Result<EquilibriumState> {
        let sentiment = self.gpu_engine.run_sentiment_graph(audio).await?;
        let vad = self.gpu_engine.run_vad_graph(audio).await?;

        let equilibrium = self.gpu_engine.compose_equilibrium(
            0.3, 0.4, 0.3,
            self.rate_constraint,
            self.context_constraint,
            sentiment.valence,
        )?;

        Ok(EquilibriumState { equilibrium, sentiment, vad })
    }
}
```

### embeddings-engine

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

## Testing

### Unit Tests (No GPU Required)

```bash
cargo test --test unit_tests
```

### Integration Tests (Mocked GPU)

```bash
cargo test --test integration_tests
```

### Accuracy Tests (CPU Reference)

```bash
cargo test --test accuracy_tests
```

### GPU Tests (Real GPU Required)

```bash
cargo test --features gpu_tests --test gpu_tests -- --ignored
```

### Benchmarks

```bash
# CUDA Graph launch
cargo bench --bench graph_launch

# Memory transfer
cargo bench --bench memory_transfer

# DPX vs CPU
cargo bench --bench dpx_vs_cpu

# Sentiment inference
cargo bench --bench sentiment_inference
```

## Documentation

- **[Testing Strategy](docs/TESTING_STRATEGY.md)**: Comprehensive testing approach
- **[Architecture](docs/ARCHITECTURE.md)**: System design and integration points

## Requirements

### Hardware

- **Minimum**: Any CUDA-capable GPU (CUDA 11.0+)
- **Recommended**: H100/H200 for DPX instructions
- **Optimal**: Blackwell B200 for 2x throughput

### Software

- **Rust**: 1.70+
- **CUDA**: 11.0+ (12.3 recommended, 13.1 for constant-time launch)
- **Driver**: NVIDIA driver 525.60.13+

## License

MIT

## Authors

- Casey DiGennaro <casey@deckboss.ai>

## Repository

https://github.com/SuperInstance/gpu-accelerator

---

**The grammar is eternal; the acceleration is constant-time.**
