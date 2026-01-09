# Testing Strategy for gpu-accelerator

## Overview

This document describes the comprehensive testing strategy for **gpu-accelerator**, a Rust library providing CUDA Graph and DPX instruction wrappers for GPU acceleration in the equilibrium-tokens ecosystem.

## Testing Philosophy

**Core Principle**: Test logic thoroughly without GPU hardware, validate accuracy with CPU comparisons, and reserve GPU tests for integration validation and performance benchmarking.

### Key Challenges

1. **GPU Availability**: Most CI runners don't have GPUs
2. **Numerical Accuracy**: GPU floating-point differs from CPU
3. **Performance Variance**: GPU performance varies by load/architecture
4. **DPX Availability**: Only works on H100/H200 GPUs

### Solutions

- **Mock GPU Interface**: Test logic without hardware
- **Epsilon Tolerance**: Allow 1e-5 difference for f32
- **Statistical Analysis**: Report P50/P95/P99 percentiles
- **Feature Flags**: Gate DPX tests behind `gpu_tests` feature

---

## Test Categories

### 1. Unit Tests (`tests/unit_tests.rs`)

**Purpose**: Test individual components without GPU

**When to Run**: Every PR, every commit

**Execution Time**: < 1 second

**GPU Required**: No

**Key Tests**:

- Graph creation and validation
- Kernel addition and dependency management
- DPX instruction wrapper API
- Memory pool allocation/deallocation
- Memory fragmentation handling
- Error conditions (OOM, invalid kernels, etc.)
- Graph metadata and parameters
- Buffer alignment and copy operations

**Example**:
```rust
#[test]
fn test_graph_creation() {
    let graph = CudaGraph::new("test_graph");
    assert_eq!(graph.name(), "test_graph");
    assert!(!graph.is_captured());
}
```

---

### 2. Integration Tests (`tests/integration_tests.rs`)

**Purpose**: Test end-to-end workflows with mocked GPU

**When to Run**: Every PR

**Execution Time**: 5-10 seconds

**GPU Required**: No (uses mocking)

**Key Tests**:

- Sentiment inference pipeline (VAD → Sentiment → Dominance)
- Multi-kernel graph execution
- Embedding computation
- Context similarity search
- Memory transfer workflows (HtoD, DtoH)
- Concurrent graph execution
- Error recovery
- Batch processing
- Streaming processing

**Example**:
```rust
#[tokio::test]
async fn test_end_to_end_sentiment_inference() {
    let engine = GPUEngine::new_mock().await.unwrap();
    let audio = AudioBuffer::from_test_data();

    let gpu_buffer = engine.upload_to_gpu(&audio).await.unwrap();
    let sentiment = engine.run_sentiment_graph(&gpu_buffer).await.unwrap();

    assert!(sentiment.valence >= -1.0 && sentiment.valence <= 1.0);
}
```

---

### 3. GPU-Specific Tests (`tests/gpu_tests.rs`)

**Purpose**: Validate on real GPU hardware

**When to Run**: Nightly on GPU runner, manual testing

**Execution Time**: Variable (30-60 seconds)

**GPU Required**: Yes (feature: `gpu_tests`)

**Key Tests**:

- CUDA Graph execution on real GPU
- DPX instructions on H100/H200
- GPU memory bandwidth (HtoD, DtoH)
- Multiple CUDA streams
- Sentiment inference latency
- CUDA Graph launch latency (target: < 50μs)
- GPU utilization
- Memory pool on GPU
- Thermal monitoring
- Multi-GPU (if available)

**Example**:
```rust
#[test]
#[ignore] // Requires GPU
fn test_cuda_graph_launch_latency() {
    let mut graph = CudaGraph::new("latency_test");
    graph.add_kernel("simple_kernel", 256, 1024);
    graph.capture().unwrap();

    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        graph.execute().unwrap();
    }

    let avg_latency = start.elapsed() / iterations;
    assert!(avg_latency < Duration::from_micros(50));
}
```

---

### 4. Accuracy Tests (`tests/accuracy_tests.rs`)

**Purpose**: Validate numerical correctness vs CPU

**When to Run**: Every PR

**Execution Time**: 10-15 seconds

**GPU Required**: No (uses CPU reference)

**Key Tests**:

- Sentiment inference (GPU vs CPU): ε = 1e-5
- VAD scores (GPU vs CPU): ε = 1e-5
- DPX vs standard DP algorithm: exact match
- Embedding computation: ε = 1e-5
- Cosine similarity: ε = 1e-5
- Matrix multiplication: ε = 1e-5
- VAD aggregation: ε = 1e-5
- Equilibrium composition: ε = 1e-5
- Softmax computation: ε = 1e-5

**Tolerance Rationale**:
- **f32 precision**: GPU may use different rounding modes
- **ε = 1e-5**: Allows for floating-point variance while catching bugs
- **Integer operations**: Require exact match (DPX min/max on integers)

**Example**:
```rust
#[test]
fn test_sentiment_vs_cpu() {
    let gpu_engine = GPUEngine::new_mock().unwrap();
    let cpu_engine = gpu_engine.cpu_reference().unwrap();

    let gpu_result = gpu_engine.run_sentiment(&audio).unwrap();
    let cpu_result = cpu_engine.run_sentiment(&audio).unwrap();

    assert!(gpu_result.valence.approx_eq(cpu_result.valence, (1e-5, 2)));
}
```

---

### 5. Performance Benchmarks (`benches/`)

**Purpose**: Measure performance characteristics

**When to Run**: On release, track over time

**Execution Time**: 1-5 minutes

**GPU Required**: Optional (mocked benchmarks available)

#### 5.1 CUDA Graph Launch (`benches/graph_launch.rs`)

**Metrics**:
- Single kernel launch latency
- Captured vs uncaptured graph
- Launch latency distribution (P50, P95, P99)
- Replay vs rebuild overhead
- Concurrent graph execution

**Targets**:
- **Graph launch**: < 50μs average
- **Captured graph**: 10-50x faster than uncaptured

#### 5.2 Memory Transfer (`benches/memory_transfer.rs`)

**Metrics**:
- HtoD bandwidth
- DtoH bandwidth
- Round-trip latency
- Pinned vs paged memory
- Concurrent transfers
- Allocation overhead

**Targets**:
- **HtoD bandwidth**: > 10 GB/s
- **DtoH bandwidth**: > 10 GB/s
- **Small transfer overhead**: < 100μs

#### 5.3 DPX vs CPU (`benches/dpx_vs_cpu.rs`)

**Metrics**:
- Min/max speedup
- Compare operations
- Path optimization (DP)
- VAD aggregation
- Constraint composition
- Rolling window operations

**Targets**:
- **DPX min/max**: 10-40x faster than CPU
- **Path optimization**: 5-20x faster

#### 5.4 Sentiment Inference (`benches/sentiment_inference.rs`)

**Metrics**:
- End-to-end pipeline latency
- Individual stage latencies (VAD, Sentiment, Dominance)
- Real-time processing (P50/P95/P99)
- Concurrent users
- Context similarity search
- Embedding computation
- Streaming vs batch

**Targets**:
- **Full pipeline**: < 10ms for 1s audio
- **P95 latency**: < 10ms for 100ms chunks (real-time)
- **Graph launch**: < 50μs (constant-time)

---

## CI/CD Strategy

### GitHub Actions Workflows

#### 1. Unit + Integration + Accuracy Tests (Every PR)

```yaml
name: Tests (No GPU)

on: [pull_request, push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run unit tests
        run: cargo test --test unit_tests

      - name: Run integration tests
        run: cargo test --test integration_tests

      - name: Run accuracy tests
        run: cargo test --test accuracy_tests
```

#### 2. GPU Tests (Nightly)

```yaml
name: GPU Tests

on:
  schedule:
    - cron: '0 0 * * *' # Daily at midnight
  workflow_dispatch: # Manual trigger

jobs:
  gpu-test:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1

      - name: Run GPU tests
        run: cargo test --features gpu_tests --test gpu_tests -- --ignored

      - name: Run benchmarks
        run: cargo bench --features gpu_tests
```

#### 3. Performance Regression (Weekly)

```yaml
name: Performance Benchmarks

on:
  schedule:
    - cron: '0 0 * * 0' # Weekly
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Run benchmarks
        run: |
          cargo bench --bench graph_launch | tee graph_launch.txt
          cargo bench --bench memory_transfer | tee memory_transfer.txt
          cargo bench --bench dpx_vs_cpu | tee dpx_vs_cpu.txt
          cargo bench --bench sentiment_inference | tee sentiment_inference.txt

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: "*.txt"
```

---

## Mocking Strategy

### Mock GPU Interface

All GPU operations have mock implementations for CI:

```rust
#[cfg(feature = "mock_gpu")]
use mock_gpu::{GPUDevice, GPUBuffer};

#[cfg(not(feature = "mock_gpu"))]
use real_gpu::{GPUDevice, GPUBuffer};
```

### Mock Behavior

- **Graph execution**: Simulate timing, validate parameters
- **Memory transfers**: Validate size/alignment, skip actual transfer
- **DPX instructions**: Verify API, return predictable results
- **Kernel launches**: Validate grid/block size, skip execution

### Real GPU Behavior

- **Graph execution**: Actual CUDA Graph launch
- **Memory transfers**: Real HtoD/DtoH
- **DPX instructions**: Use H100/H200 DPX hardware
- **Kernel launches**: Execute on GPU

---

## Accuracy Validation Approach

### Numerical Tolerance

**Default Tolerance**: `ε = 1e-5` for f32 operations

**Rationale**:
- GPU and CPU use different floating-point rounding modes
- GPU operations may be reordered for optimization
- 1e-5 is sufficient to catch bugs while allowing variance

**Use `float-cmp` crate**:
```rust
use float_cmp::ApproxEq;

assert!(gpu_val.approx_eq(&cpu_val, (1e-5, 2)));
```

### Exact Match Requirements

**Integer operations**: Require exact match
```rust
assert_eq!(dpx_min, cpu_min); // Exact match for integers
```

**Algorithm correctness**: DPX vs standard DP
```rust
assert_eq!(dpx_path, cpu_path); // Paths must be identical
```

### Accuracy Test Categories

| Test Type | Tolerance | Rationale |
|-----------|-----------|-----------|
| f32 operations | ε = 1e-5 | Floating-point variance |
| Integer operations | Exact match | No rounding |
| DP algorithms | Exact match | Deterministic |
| Softmax | ε = 1e-5 | Sum = 1.0 property |
| Cosine similarity | ε = 1e-5 | Bounded [0,1] |

---

## Performance Regression Detection

### Baseline Establishment

**First Run**: Establish baseline metrics
```bash
cargo bench --bench graph_launch | tee baseline.txt
```

**Subsequent Runs**: Compare against baseline
```bash
cargo bench --bench graph_launch | tee current.txt
critcmp baseline current
```

### Regression Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Graph launch latency | +10% | Warning |
| Memory bandwidth | -5% | Warning |
| DPX speedup | -20% | Warning |
| Sentiment inference P95 | +10% | Warning |
| Sentiment inference P95 | +25% | Block PR |

### Automated Detection

```yaml
- name: Check performance regression
  run: |
    cargo bench --bench sentiment_inference > current.txt
    python scripts/check_regression.py baseline.txt current.txt
```

---

## Test Execution

### Local Development

```bash
# Run all tests (no GPU required)
cargo test

# Run specific test file
cargo test --test unit_tests

# Run with output
cargo test -- --nocapture

# Run GPU tests (requires GPU)
cargo test --features gpu_tests --test gpu_tests -- --ignored

# Run benchmarks
cargo bench
```

### CI Environment

```bash
# Standard CI (no GPU)
cargo test --test unit_tests
cargo test --test integration_tests
cargo test --test accuracy_tests

# GPU CI (with GPU runner)
cargo test --features gpu_tests --test gpu_tests -- --ignored
cargo bench --features gpu_tests
```

---

## Coverage Goals

### Target Coverage

- **Unit tests**: 90%+ line coverage
- **Integration tests**: 80%+ branch coverage
- **Accuracy tests**: 100% of numeric operations
- **GPU tests**: 70%+ code paths (hardware-dependent)

### Coverage Tools

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage/
```

### Coverage Exclusions

- **Mock implementations**: Not coverage-critical
- **GPU-specific code**: Mark with `#[cfg(feature = "gpu_tests")]`
- **Error paths**: Test critical errors, skip unreachable

---

## Success Metrics

### Test Suite Health

- ✅ All tests pass in < 30 seconds (unit + integration + accuracy)
- ✅ GPU tests pass on nightly run
- ✅ Performance benchmarks stable (±5%)
- ✅ No accuracy regressions (ε = 1e-5)

### CI/CD Health

- ✅ PR tests complete in < 5 minutes
- ✅ Nightly GPU tests pass > 95% of time
- ✅ Performance regression detection functional
- ✅ Coverage reports available for each PR

### Production Readiness

- ✅ All accuracy tests pass
- ✅ Real-time targets met (P95 < 10ms)
- ✅ Memory leaks detected (valgrind, sanitizers)
- ✅ Stress tests pass (100k+ iterations)

---

## Debugging Failed Tests

### Common Issues

#### 1. GPU Tests Timeout

**Symptom**: Tests hang or timeout on GPU runner

**Diagnosis**:
```bash
# Check GPU availability
nvidia-smi

# Check CUDA version
nvcc --version

# Run with verbose logging
RUST_LOG=debug cargo test --features gpu_tests --test gpu_tests -- --nocapture
```

**Solution**:
- Verify GPU driver compatibility
- Check GPU memory availability
- Reduce test data size

#### 2. Accuracy Test Failures

**Symptom**: GPU result differs from CPU beyond ε

**Diagnosis**:
```bash
# Run with detailed output
cargo test --test accuracy_tests -- --nocapture

# Check specific test
cargo test test_sentiment_vs_cpu -- --exact --nocapture
```

**Solution**:
- Verify tolerance (ε = 1e-5 appropriate?)
- Check for NaN/Inf values
- Validate CPU reference implementation

#### 3. Performance Regression

**Symptom**: Benchmarks show >10% slowdown

**Diagnosis**:
```bash
# Compare with baseline
critcmp baseline.txt current.txt

# Run multiple times to check variance
for i in {1..10}; do cargo bench --bench graph_launch; done
```

**Solution**:
- Check for algorithm changes
- Verify GPU not thermal throttling
- Check for background processes

---

## Future Improvements

### Short-Term (1-3 months)

- [ ] Add property-based testing (proptest)
- [ ] Implement fuzzing for GPU kernels
- [ ] Add more GPU architectures (A100, Blackwell)
- [ ] Improve mock GPU realism

### Medium-Term (3-6 months)

- [ ] Automated performance regression CI
- [ ] GPU test sharding (run in parallel)
- [ ] Differential testing (multiple GPUs)
- [ ] Flame graph profiling integration

### Long-Term (6-12 months)

- [ ] Formal verification of DPX correctness
- [ ] Machine learning for anomaly detection
- [ ] Cross-platform GPU testing (AMD, Intel)
- [ ] Distributed testing across multiple GPUs

---

## Conclusion

This testing strategy ensures gpu-accelerator is:

1. **Correct**: Accuracy tests validate numerical correctness
2. **Fast**: Benchmarks verify performance targets
3. **Reliable**: GPU tests catch hardware-specific issues
4. **Maintainable**: Mocked tests enable rapid development

**The grammar is eternal; the tests ensure it stays that way.**
