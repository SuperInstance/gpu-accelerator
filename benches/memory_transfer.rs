// GPU Memory Transfer Bandwidth Benchmark
// Measures HtoD and DtoH transfer performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use gpu_accelerator::{engine::GPUEngine, memory::MemoryPool};
use std::time::Duration;

fn bench_htod_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("htod_transfer");

    // Test different data sizes
    for size_in_mb in [1, 4, 16, 64, 256].iter() {
        let size_bytes = size_in_mb * 1024 * 1024;
        let data: Vec<f32> = vec![1.0f32; size_bytes / 4];

        group.throughput(Throughput::Bytes(size_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("size_mb", size_in_mb),
            size_in_mb,
            |b, _| {
                let engine = GPUEngine::new_mock().unwrap();

                b.iter(|| {
                    engine.htod_transfer_mock(black_box(&data)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_dtoh_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtoh_transfer");

    // Test different data sizes
    for size_in_mb in [1, 4, 16, 64, 256].iter() {
        let size_bytes = size_in_mb * 1024 * 1024;
        let pool = MemoryPool::new(size_bytes);
        let gpu_buffer = pool.allocate(size_bytes).unwrap();

        group.throughput(Throughput::Bytes(size_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("size_mb", size_in_mb),
            size_in_mb,
            |b, _| {
                let engine = GPUEngine::new_mock().unwrap();

                b.iter(|| {
                    engine.dtoh_transfer_mock(black_box(&gpu_buffer)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_roundtrip_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_transfer");

    for size_in_mb in [1, 4, 16, 64, 256].iter() {
        let size_bytes = size_in_mb * 1024 * 1024;
        let data: Vec<f32> = vec![1.0f32; size_bytes / 4];

        group.throughput(Throughput::Bytes((2 * size_bytes) as u64)); // HtoD + DtoH

        group.bench_with_input(
            BenchmarkId::new("size_mb", size_in_mb),
            size_in_mb,
            |b, _| {
                let engine = GPUEngine::new_mock().unwrap();

                b.iter(|| {
                    let gpu_buffer = engine.htod_transfer_mock(black_box(&data)).unwrap();
                    engine.dtoh_transfer_mock(black_box(&gpu_buffer)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_pinned_vs_paged_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_type");

    let size = 16 * 1024 * 1024; // 16MB
    let data: Vec<f32> = vec![1.0f32; size / 4];

    // Pinned memory (page-locked, faster transfers)
    group.bench_function("pinned_htod", |b| {
        let engine = GPUEngine::new_mock_with_pinned().unwrap();

        b.iter(|| {
            engine.htod_transfer_pinned(black_box(&data)).unwrap()
        })
    });

    // Paged memory (standard, slower)
    group.bench_function("paged_htod", |b| {
        let engine = GPUEngine::new_mock().unwrap();

        b.iter(|| {
            engine.htod_transfer_mock(black_box(&data)).unwrap()
        })
    });

    group.finish();
}

fn bench_concurrent_transfers(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_transfers");

    let size = 16 * 1024 * 1024; // 16MB
    let data1: Vec<f32> = vec![1.0f32; size / 4];
    let data2: Vec<f32> = vec![2.0f32; size / 4];
    let data3: Vec<f32> = vec![3.0f32; size / 4];

    group.bench_function("sequential", |b| {
        let engine = GPUEngine::new_mock().unwrap();

        b.iter(|| {
            engine.htod_transfer_mock(black_box(&data1)).unwrap();
            engine.htod_transfer_mock(black_box(&data2)).unwrap();
            engine.htod_transfer_mock(black_box(&data3)).unwrap()
        })
    });

    group.bench_function("concurrent", |b| {
        b.iter(|| {
            let engine1 = GPUEngine::new_mock().unwrap();
            let engine2 = GPUEngine::new_mock().unwrap();
            let engine3 = GPUEngine::new_mock().unwrap();

            let h1 = std::thread::spawn(move || {
                engine1.htod_transfer_mock(&data1).unwrap()
            });

            let h2 = std::thread::spawn(move || {
                engine2.htod_transfer_mock(&data2).unwrap()
            });

            let h3 = std::thread::spawn(move || {
                engine3.htod_transfer_mock(&data3).unwrap()
            });

            h1.join().unwrap();
            h2.join().unwrap();
            h3.join().unwrap()
        })
    });

    group.finish();
}

fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    for size_in_kb in [4, 16, 64, 256, 1024].iter() {
        let size_bytes = size_in_kb * 1024;

        group.bench_with_input(
            BenchmarkId::new("size_kb", size_in_kb),
            size_in_kb,
            |b, _| {
                let pool = MemoryPool::new(1024 * 1024 * 1024); // 1GB pool

                b.iter(|| {
                    pool.allocate(black_box(size_bytes)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_pool_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_fragmentation");

    group.bench_function("no_fragmentation", |b| {
        let pool = MemoryPool::new(1024 * 1024); // 1MB

        b.iter(|| {
            let buf1 = pool.allocate(256 * 1024).unwrap();
            let buf2 = pool.allocate(256 * 1024).unwrap();
            let buf3 = pool.allocate(256 * 1024).unwrap();
            let buf4 = pool.allocate(256 * 1024).unwrap();
            drop(buf1);
            drop(buf2);
            drop(buf3);
            drop(buf4);
        })
    });

    group.bench_function("with_fragmentation", |b| {
        let pool = MemoryPool::new(1024 * 1024); // 1MB

        b.iter(|| {
            let buf1 = pool.allocate(256 * 1024).unwrap();
            let buf2 = pool.allocate(256 * 1024).unwrap();
            let buf3 = pool.allocate(256 * 1024).unwrap();
            drop(buf2); // Free middle buffer -> fragmentation
            let buf4 = pool.allocate(256 * 1024).unwrap();
            drop(buf1);
            drop(buf3);
            drop(buf4);
        })
    });

    group.finish();
}

fn bench_bandwidth_saturation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bandwidth_saturation");

    let engine = GPUEngine::new_mock().unwrap();

    // Test PCIe bandwidth saturation
    let sizes_mb = [128, 256, 512, 1024, 2048];

    for size_mb in sizes_mb.iter() {
        let size_bytes = size_mb * 1024 * 1024;
        let data: Vec<f32> = vec![1.0f32; size_bytes / 4];

        group.bench_with_input(
            BenchmarkId::new("size_mb", size_mb),
            size_mb,
            |b, _| {
                b.iter(|| {
                    engine.htod_transfer_mock(black_box(&data)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_transfer_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("transfer_overhead");

    let engine = GPUEngine::new_mock().unwrap();

    // Very small transfers to measure overhead
    for size_bytes in [4, 16, 64, 256, 1024].iter() {
        let data: Vec<f32> = vec![1.0f32; size_bytes / 4];

        group.bench_with_input(
            BenchmarkId::new("size_bytes", size_bytes),
            size_bytes,
            |b, _| {
                b.iter(|| {
                    engine.htod_transfer_mock(black_box(&data)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_zero_copy_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy");

    let size = 16 * 1024 * 1024; // 16MB
    let data: Vec<f32> = vec![1.0f32; size / 4];

    // Traditional HtoD
    group.bench_function("traditional_copy", |b| {
        let engine = GPUEngine::new_mock().unwrap();

        b.iter(|| {
            engine.htod_transfer_mock(black_box(&data)).unwrap()
        })
    });

    // Zero-copy (direct access)
    group.bench_function("zero_copy", |b| {
        let engine = GPUEngine::new_mock_with_zero_copy().unwrap();

        b.iter(|| {
            engine.zero_copy_access(black_box(&data)).unwrap()
        })
    });

    group.finish();
}

fn bench_async_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_transfer");

    let size = 16 * 1024 * 1024; // 16MB
    let data: Vec<f32> = vec![1.0f32; size / 4];
    let engine = GPUEngine::new_mock().unwrap();

    // Synchronous transfer
    group.bench_function("sync", |b| {
        b.iter(|| {
            engine.htod_transfer_sync(black_box(&data)).unwrap()
        })
    });

    // Asynchronous transfer (with stream)
    group.bench_function("async", |b| {
        b.iter(|| {
            let stream = engine.create_stream().unwrap();
            engine.htod_transfer_async(black_box(&data), &stream).unwrap();
            stream.synchronize().unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_htod_transfer,
    bench_dtoh_transfer,
    bench_roundtrip_transfer,
    bench_pinned_vs_paged_memory,
    bench_concurrent_transfers,
    bench_memory_allocation,
    bench_memory_pool_fragmentation,
    bench_bandwidth_saturation,
    bench_transfer_overhead,
    bench_zero_copy_transfer,
    bench_async_transfer
);

criterion_main!(benches);
