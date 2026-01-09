// CUDA Graph Launch Latency Benchmark
// Measures the overhead of launching CUDA Graphs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gpu_accelerator::{graph::CudaGraph, engine::GPUEngine};
use std::time::Duration;

fn bench_graph_launch_single_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_graph_launch");

    // Test different kernel configurations
    for kernel_count in [1, 2, 4, 8, 16].iter() {
        let mut graph = setup_graph_with_kernels("single_kernel", *kernel_count);

        group.bench_with_input(
            BenchmarkId::new("kernels", kernel_count),
            kernel_count,
            |b, _| {
                b.iter(|| {
                    graph.launch_mock(black_box(&[])).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_graph_launch_captured_vs_uncaptured(c: &mut Criterion) {
    let mut group = c.benchmark_group("captured_vs_uncaptured");

    // Uncaptured graph (traditional kernel launch)
    let mut uncaptured_graph = setup_graph_with_kernels("uncaptured", 4);

    group.bench_function("uncaptured", |b| {
        b.iter(|| {
            uncaptured_graph.launch_kernels_individually(black_box(&[])).unwrap()
        })
    });

    // Captured graph (CUDA Graph)
    let mut captured_graph = setup_graph_with_kernels("captured", 4);
    captured_graph.capture_mock().unwrap();

    group.bench_function("captured", |b| {
        b.iter(|| {
            captured_graph.launch_mock(black_box(&[])).unwrap()
        })
    });

    group.finish();
}

fn bench_graph_launch_with_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_with_parameters");

    let mut graph = setup_graph_with_kernels("param_test", 4);
    graph.add_parameter("learning_rate", 0.001f32);
    graph.add_parameter("batch_size", 32usize);

    group.bench_function("with_parameters", |b| {
        b.iter(|| {
            graph.launch_with_params_mock(
                black_box(&[0.001f32, 32.0f32])
            ).unwrap()
        })
    });

    group.finish();
}

fn bench_graph_launch_latency_distribution(c: &mut Criterion) {
    let mut graph = setup_graph_with_kernels("latency_dist", 4);
    graph.capture_mock().unwrap();

    // Warmup
    for _ in 0..100 {
        graph.launch_mock(&[]).unwrap();
    }

    let mut group = c.benchmark_group("latency_distribution");

    // Measure P50, P95, P99 latencies
    let mut latencies = Vec::with_capacity(10000);

    for _ in 0..10000 {
        let start = std::time::Instant::now();
        graph.launch_mock(&[]).unwrap();
        latencies.push(start.elapsed());
    }

    latencies.sort();

    let p50 = latencies[5000];
    let p95 = latencies[9500];
    let p99 = latencies[9900];

    group.bench_function("p50", |b| b.iter(|| p50));
    group.bench_function("p95", |b| b.iter(|| p95));
    group.bench_function("p99", |b| b.iter(|| p99));

    group.finish();

    println!("\nCUDA Graph Launch Latency Distribution:");
    println!("  P50: {:?}", p50);
    println!("  P95: {:?}", p95);
    println!("  P99: {:?}", p99);
}

fn bench_graph_replay_vs_rebuild(c: &mut Criterion) {
    let mut group = c.benchmark_group("replay_vs_rebuild");

    // Replay captured graph
    let mut captured = setup_graph_with_kernels("replay", 4);
    captured.capture_mock().unwrap();

    group.bench_function("replay", |b| {
        b.iter(|| {
            captured.replay_mock().unwrap()
        })
    });

    // Rebuild and launch (for comparison)
    group.bench_function("rebuild", |b| {
        b.iter(|| {
            let mut graph = setup_graph_with_kernels("rebuild", 4);
            graph.launch_mock(&[]).unwrap()
        })
    });

    group.finish();
}

fn bench_concurrent_graph_launch(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_launch");

    let graph1 = setup_graph_with_kernels("concurrent_1", 2);
    let graph2 = setup_graph_with_kernels("concurrent_2", 2);
    let graph3 = setup_graph_with_kernels("concurrent_3", 2);

    group.bench_function("sequential", |b| {
        b.iter(|| {
            graph1.launch_mock(&[]).unwrap();
            graph2.launch_mock(&[]).unwrap();
            graph3.launch_mock(&[]).unwrap()
        })
    });

    group.bench_function("concurrent", |b| {
        b.iter(|| {
            let h1 = std::thread::spawn(|| {
                let g = setup_graph_with_kernels("thread_1", 2);
                g.launch_mock(&[]).unwrap()
            });

            let h2 = std::thread::spawn(|| {
                let g = setup_graph_with_kernels("thread_2", 2);
                g.launch_mock(&[]).unwrap()
            });

            let h3 = std::thread::spawn(|| {
                let g = setup_graph_with_kernels("thread_3", 2);
                g.launch_mock(&[]).unwrap()
            });

            h1.join().unwrap();
            h2.join().unwrap();
            h3.join().unwrap()
        })
    });

    group.finish();
}

fn bench_graph_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");

    let graph = setup_graph_with_kernels("memory", 4);

    group.bench_function("graph_creation", |b| {
        b.iter(|| {
            CudaGraph::new("test_graph")
        })
    });

    group.bench_function("graph_capture", |b| {
        b.iter(|| {
            let mut g = setup_graph_with_kernels("capture_test", 4);
            g.capture_mock().unwrap()
        })
    });

    group.finish();
}

// Helper function to setup test graphs
fn setup_graph_with_kernels(name: &str, kernel_count: usize) -> CudaGraph {
    let mut graph = CudaGraph::new(name);

    for i in 0..kernel_count {
        graph.add_kernel(
            &format!("kernel_{}", i),
            256, // block size
            1024 // grid size
        );
    }

    graph
}

criterion_group!(
    benches,
    bench_graph_launch_single_kernel,
    bench_graph_launch_captured_vs_uncaptured,
    bench_graph_launch_with_parameters,
    bench_graph_launch_latency_distribution,
    bench_graph_replay_vs_rebuild,
    bench_concurrent_graph_launch,
    bench_graph_memory_overhead
);

criterion_main!(benches);
