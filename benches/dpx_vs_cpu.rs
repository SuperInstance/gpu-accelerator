// DPX vs CPU Performance Comparison Benchmark
// Measures speedup of DPX instructions over CPU implementations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gpu_accelerator::dpx::DPXContext;
use std::time::Duration;

// Benchmark: DPX min/max vs CPU
fn bench_dpx_min_max_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_max");

    let context = DPXContext::new();
    context.initialize_mock();

    // Test different data sizes
    for size in [100, 1000, 10000, 100000].iter() {
        let data: Vec<f32> = (0..*size).map(|i| i as f32).collect();

        // CPU baseline
        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            size,
            |b, _| {
                b.iter(|| {
                    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    black_box((min, max))
                })
            },
        );

        // DPX accelerated
        group.bench_with_input(
            BenchmarkId::new("dpx", size),
            size,
            |b, _| {
                b.iter(|| {
                    context.min_max_mock(black_box(&data)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: DPX compare operations
fn bench_dpx_compare_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare");

    let context = DPXContext::new();
    context.initialize_mock();

    let size = 100000;
    let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data2: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    // CPU comparison
    group.bench_function("cpu", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(size);
            for i in 0..size {
                result.push(data1[i].partial_cmp(&data2[i]).unwrap());
            }
            black_box(result)
        })
    });

    // DPX comparison
    group.bench_function("dpx", |b| {
        b.iter(|| {
            context.compare_arrays_mock(
                black_box(&data1),
                black_box(&data2)
            ).unwrap()
        })
    });

    group.finish();
}

// Benchmark: Path optimization (dynamic programming)
fn bench_path_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("path_optimization");

    let context = DPXContext::new();
    context.initialize_mock();

    // Test different matrix sizes
    for (rows, cols) in [(10, 10), (50, 50), (100, 100)].iter() {
        let costs: Vec<Vec<f32>> = (0..*rows)
            .map(|_| (0..*cols).map(|j| j as f32).collect())
            .collect();

        // CPU DP
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    compute_optimal_path_cpu(black_box(&costs))
                })
            },
        );

        // DPX DP
        group.bench_with_input(
            BenchmarkId::new("dpx", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    context.compute_optimal_path_mock(black_box(&costs)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: VAD score aggregation
fn bench_vad_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_aggregation");

    let context = DPXContext::new();
    context.initialize_mock();

    // Test different window sizes
    for window_size in [10, 50, 100, 500, 1000].iter() {
        let vad_scores: Vec<f32> = (0..*window_size)
            .map(|_| rand::random::<f32>())
            .collect();

        // CPU aggregation
        group.bench_with_input(
            BenchmarkId::new("cpu", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let min = vad_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max = vad_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let sum: f32 = vad_scores.iter().sum();
                    let mean = sum / vad_scores.len() as f32;
                    black_box((min, max, mean))
                })
            },
        );

        // DPX aggregation
        group.bench_with_input(
            BenchmarkId::new("dpx", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    context.aggregate_vad_scores_mock(black_box(&vad_scores)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: Constraint composition
fn bench_constraint_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_composition");

    let context = DPXContext::new();
    context.initialize_mock();

    let rate_constraints: Vec<f32> = vec![0.8; 1000];
    let context_constraints: Vec<f32> = vec![0.7; 1000];
    let sentiment_constraints: Vec<f32> = vec![0.9; 1000];

    let rate_weight = 0.3f32;
    let context_weight = 0.4f32;
    let sentiment_weight = 0.3f32;

    // CPU composition
    group.bench_function("cpu", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(1000);
            for i in 0..1000 {
                let eq = rate_weight * rate_constraints[i]
                    + context_weight * context_constraints[i]
                    + sentiment_weight * sentiment_constraints[i];
                results.push(eq);
            }
            black_box(results)
        })
    });

    // DPX composition
    group.bench_function("dpx", |b| {
        b.iter(|| {
            context.compose_constraints_mock(
                black_box(rate_weight),
                black_box(context_weight),
                black_box(sentiment_weight),
                black_box(&rate_constraints),
                black_box(&context_constraints),
                black_box(&sentiment_constraints),
            ).unwrap()
        })
    });

    group.finish();
}

// Benchmark: Rolling window min/max
fn bench_rolling_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_window");

    let context = DPXContext::new();
    context.initialize_mock();

    let data_size = 100000;
    let data: Vec<f32> = (0..data_size).map(|i| i as f32).collect();

    // Test different window sizes
    for window_size in [10, 50, 100, 500].iter() {
        // CPU rolling window
        group.bench_with_input(
            BenchmarkId::new("cpu", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let mut mins = Vec::new();
                    let mut maxs = Vec::new();
                    for i in 0..=(data_size - window_size) {
                        let window = &data[i..i + window_size];
                        let min = window.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max = window.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        mins.push(min);
                        maxs.push(max);
                    }
                    black_box((mins, maxs))
                })
            },
        );

        // DPX rolling window
        group.bench_with_input(
            BenchmarkId::new("dpx", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    context.rolling_window_min_max_mock(
                        black_box(&data),
                        black_box(*window_size)
                    ).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: Smith-Waterman-like DP
fn bench_smith_waterman(c: &mut Criterion) {
    let mut group = c.benchmark_group("smith_waterman");

    let context = DPXContext::new();
    context.initialize_mock();

    // Test different sequence lengths
    for seq_len in [100, 500, 1000].iter() {
        let seq1: Vec<f32> = (0..*seq_len).map(|_| rand::random()).collect();
        let seq2: Vec<f32> = (0..*seq_len).map(|_| rand::random()).collect();

        // CPU implementation
        group.bench_with_input(
            BenchmarkId::new("cpu", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    smith_waterman_cpu(black_box(&seq1), black_box(&seq2))
                })
            },
        );

        // DPX implementation
        group.bench_with_input(
            BenchmarkId::new("dpx", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    context.smith_waterman_mock(
                        black_box(&seq1),
                        black_box(&seq2)
                    ).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: Parallel DPX operations
fn bench_parallel_dpx(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_dpx");

    let context = DPXContext::new();
    context.initialize_mock();

    let data: Vec<f32> = (0..100000).map(|i| i as f32).collect();

    // Sequential DPX
    group.bench_function("sequential", |b| {
        b.iter(|| {
            for _ in 0..10 {
                context.min_max_mock(&data).unwrap();
            }
        })
    });

    // Parallel DPX (multi-threaded)
    group.bench_function("parallel", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..10)
                .map(|_| {
                    let data = data.clone();
                    std::thread::spawn(move || {
                        let ctx = DPXContext::new();
                        ctx.initialize_mock();
                        ctx.min_max_mock(&data).unwrap()
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    group.finish();
}

// Helper: CPU reference DP implementation
fn compute_optimal_path_cpu(costs: &Vec<Vec<f32>>) -> Vec<usize> {
    let rows = costs.len();
    let cols = costs[0].len();

    let mut dp = vec![vec![0.0f32; cols]; rows];

    for r in 0..rows {
        dp[r][0] = costs[r][0];
    }

    for c in 1..cols {
        for r in 0..rows {
            let prev_min = if r == 0 {
                dp[r][c - 1].min(dp[r + 1][c - 1])
            } else if r == rows - 1 {
                dp[r][c - 1].min(dp[r - 1][c - 1])
            } else {
                dp[r][c - 1].min(dp[r - 1][c - 1]).min(dp[r + 1][c - 1])
            };

            dp[r][c] = costs[r][c] + prev_min;
        }
    }

    let mut path = vec![0usize; cols];
    let mut current_row = (0..rows)
        .min_by(|&&a, &&b| dp[a].last().partial_cmp(&dp[b].last()).unwrap())
        .unwrap();

    path[cols - 1] = current_row;

    for c in (1..cols).rev() {
        current_row = if current_row == 0 {
            if dp[current_row + 1][c - 1] < dp[current_row][c] {
                current_row + 1
            } else {
                current_row
            }
        } else if current_row == rows - 1 {
            if dp[current_row - 1][c - 1] < dp[current_row][c] {
                current_row - 1
            } else {
                current_row
            }
        } else {
            let candidates = [
                (current_row, dp[current_row][c - 1]),
                (current_row - 1, dp[current_row - 1][c - 1]),
                (current_row + 1, dp[current_row + 1][c - 1]),
            ];
            candidates
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0
        };

        path[c - 1] = current_row;
    }

    path
}

// Helper: CPU Smith-Waterman implementation
fn smith_waterman_cpu(seq1: &Vec<f32>, seq2: &Vec<f32>) -> f32 {
    let m = seq1.len();
    let n = seq2.len();

    let mut dp = vec![vec![0.0f32; n + 1]; m + 1];
    let mut max_score = 0.0f32;

    for i in 1..=m {
        for j in 1..=n {
            let match_score = if seq1[i - 1] == seq2[j - 1] { 2.0 } else { -1.0 };

            dp[i][j] = (dp[i - 1][j - 1] + match_score)
                .max(dp[i - 1][j] - 1.0)
                .max(dp[i][j - 1] - 1.0)
                .max(0.0);

            max_score = max_score.max(dp[i][j]);
        }
    }

    max_score
}

criterion_group!(
    benches,
    bench_dpx_min_max_vs_cpu,
    bench_dpx_compare_vs_cpu,
    bench_path_optimization,
    bench_vad_aggregation,
    bench_constraint_composition,
    bench_rolling_window,
    bench_smith_waterman,
    bench_parallel_dpx
);

criterion_main!(benches);
