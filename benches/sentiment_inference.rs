// Sentiment Inference Integration Benchmark
// Measures end-to-end performance of equilibrium-tokens sentiment pipeline

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gpu_accelerator::{engine::GPUEngine, types::AudioBuffer};
use std::time::Duration;

// Benchmark: Complete sentiment inference pipeline
fn bench_sentiment_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentiment_inference");

    let engine = GPUEngine::new_mock().unwrap();

    // Test different audio lengths
    for duration_ms in [100, 500, 1000, 5000].iter() {
        let audio = AudioBuffer::generate_test_data(*duration_ms);

        group.bench_with_input(
            BenchmarkId::new("pipeline", duration_ms),
            duration_ms,
            |b, _| {
                b.iter(|| {
                    engine.run_sentiment_graph_mock(black_box(&audio)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: Individual stages of sentiment pipeline
fn bench_sentiment_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentiment_stages");

    let engine = GPUEngine::new_mock().unwrap();
    let audio = AudioBuffer::generate_test_data(1000); // 1 second

    // VAD stage
    group.bench_function("vad", |b| {
        b.iter(|| {
            engine.run_vad_graph_mock(black_box(&audio)).unwrap()
        })
    });

    // Sentiment analysis stage
    group.bench_function("sentiment", |b| {
        b.iter(|| {
            engine.run_sentiment_graph_mock(black_box(&audio)).unwrap()
        })
    });

    // Dominance detection stage
    group.bench_function("dominance", |b| {
        b.iter(|| {
            engine.run_dominance_graph_mock(black_box(&audio)).unwrap()
        })
    });

    // Full pipeline (VAD -> Sentiment -> Dominance)
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let vad = engine.run_vad_graph_mock(black_box(&audio)).unwrap();
            let sentiment = engine.run_sentiment_graph_mock(black_box(&audio)).unwrap();
            let dominance = engine.run_dominance_graph_mock(black_box(&audio)).unwrap();
            black_box((vad, sentiment, dominance))
        })
    });

    group.finish();
}

// Benchmark: Real-time processing (target < 10ms latency)
fn bench_realtime_sentiment(c: &mut Criterion) {
    let mut group = c.benchmark_group("realtime");

    let engine = GPUEngine::new_mock().unwrap();

    // Simulate real-time audio chunks (100ms = 10 chunks per second)
    let chunk_duration_ms = 100;
    let audio = AudioBuffer::generate_test_data(chunk_duration_ms);

    // Measure P50, P95, P99 latencies
    let mut latencies = Vec::with_capacity(1000);

    // Warmup
    for _ in 0..100 {
        engine.run_sentiment_graph_mock(&audio).unwrap();
    }

    // Measure latencies
    for _ in 0..1000 {
        let start = std::time::Instant::now();
        engine.run_sentiment_graph_mock(&audio).unwrap();
        latencies.push(start.elapsed());
    }

    latencies.sort();

    let p50 = latencies[500];
    let p95 = latencies[950];
    let p99 = latencies[990];

    println!("\nReal-Time Sentiment Inference Latency (100ms chunk):");
    println!("  P50: {:?}", p50);
    println!("  P95: {:?}", p95);
    println!("  P99: {:?}", p99);

    // Check if we meet real-time target
    if p95 < Duration::from_millis(10) {
        println!("  ✓ Meets real-time target (< 10ms P95)");
    } else {
        println!("  ✗ Does NOT meet real-time target (target: < 10ms P95)");
    }

    group.bench_function("p50_latency", |b| b.iter(|| p50));
    group.bench_function("p95_latency", |b| b.iter(|| p95));
    group.bench_function("p99_latency", |b| b.iter(|| p99));

    group.finish();
}

// Benchmark: Batch processing for multiple concurrent users
fn bench_concurrent_users(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_users");

    let audio = AudioBuffer::generate_test_data(1000); // 1 second

    // Test different numbers of concurrent users
    for num_users in [1, 5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("users", num_users),
            num_users,
            |b, _| {
                b.iter(|| {
                    let handles: Vec<_> = (0..*num_users)
                        .map(|_| {
                            let audio = audio.clone();
                            std::thread::spawn(move || {
                                let engine = GPUEngine::new_mock().unwrap();
                                engine.run_sentiment_graph_mock(&audio).unwrap()
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

// Benchmark: VAD score aggregation
fn bench_vad_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_aggregation");

    let engine = GPUEngine::new_mock().unwrap();

    // Test different window sizes
    for window_size in [10, 50, 100, 500].iter() {
        let vad_scores: Vec<f32> = (0..*window_size)
            .map(|_| rand::random::<f32>())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("window_size", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    engine.aggregate_vad_scores_mock(black_box(&vad_scores)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: Equilibrium constraint composition
fn bench_equilibrium_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("equilibrium_composition");

    let engine = GPUEngine::new_mock().unwrap();

    let rate_weight = 0.3f32;
    let context_weight = 0.4f32;
    let sentiment_weight = 0.3f32;

    let rate_constraint = 0.8f32;
    let context_constraint = 0.7f32;
    let sentiment_constraint = 0.9f32;

    group.bench_function("single_composition", |b| {
        b.iter(|| {
            engine.compose_equilibrium_mock(
                black_box(rate_weight),
                black_box(context_weight),
                black_box(sentiment_weight),
                black_box(rate_constraint),
                black_box(context_constraint),
                black_box(sentiment_constraint),
            ).unwrap()
        })
    });

    // Batch composition (for multiple contexts)
    let batch_size = 100;
    let rate_constraints: Vec<f32> = vec![0.8; batch_size];
    let context_constraints: Vec<f32> = vec![0.7; batch_size];
    let sentiment_constraints: Vec<f32> = vec![0.9; batch_size];

    group.bench_function("batch_composition", |b| {
        b.iter(|| {
            engine.compose_equilibrium_batch_mock(
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

// Benchmark: Context similarity search
fn bench_context_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_similarity");

    let engine = GPUEngine::new_mock().unwrap();

    // Create query embedding
    let query: Vec<f32> = (0..768).map(|_| rand::random()).collect();

    // Test different numbers of context embeddings
    for num_contexts in [100, 1000, 10000].iter() {
        let contexts: Vec<Vec<f32>> = (0..*num_contexts)
            .map(|_| (0..768).map(|_| rand::random()).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("contexts", num_contexts),
            num_contexts,
            |b, _| {
                b.iter(|| {
                    engine.compute_similarity_mock(black_box(&query), black_box(&contexts)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: Embedding computation
fn bench_embedding_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_computation");

    let engine = GPUEngine::new_mock().unwrap();

    // Test different sequence lengths
    for seq_len in [10, 50, 100, 500].iter() {
        let tokens: Vec<i32> = (0..*seq_len).map(|_| rand::random()).collect();

        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    engine.compute_embeddings_mock(black_box(&tokens)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// Benchmark: Memory pressure (simulating long conversations)
fn bench_long_conversation(c: &mut Criterion) {
    let mut group = c.benchmark_group("long_conversation");

    let engine = GPUEngine::new_mock().unwrap();

    // Simulate conversation with increasing context length
    for num_turns in [10, 50, 100, 500].iter() {
        let conversation: Vec<AudioBuffer> = (0..*num_turns)
            .map(|_| AudioBuffer::generate_test_data(1000))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("turns", num_turns),
            num_turns,
            |b, _| {
                b.iter(|| {
                    for audio in &conversation {
                        engine.run_sentiment_graph_mock(black_box(audio)).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

// Benchmark: CUDA Graph overhead vs traditional launch
fn bench_cuda_graph_vs_traditional(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_vs_traditional");

    let audio = AudioBuffer::generate_test_data(1000);

    // Traditional kernel launches (uncaptured)
    group.bench_function("traditional_launch", |b| {
        let engine = GPUEngine::new_mock().unwrap();

        b.iter(|| {
            engine.run_sentiment_traditional_mock(black_box(&audio)).unwrap()
        })
    });

    // CUDA Graph (captured)
    group.bench_function("cuda_graph", |b| {
        let engine = GPUEngine::new_mock().unwrap();

        b.iter(|| {
            engine.run_sentiment_graph_mock(black_box(&audio)).unwrap()
        })
    });

    group.finish();
}

// Benchmark: Streaming mode (processing audio chunks as they arrive)
fn bench_streaming_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_mode");

    let engine = GPUEngine::new_mock().unwrap();

    // Simulate streaming audio (100ms chunks)
    let num_chunks = 100;
    let chunks: Vec<AudioBuffer> = (0..num_chunks)
        .map(|_| AudioBuffer::generate_test_data(100))
        .collect();

    group.bench_function("streaming", |b| {
        b.iter(|| {
            let mut stream = engine.create_stream_mock().unwrap();

            for chunk in &chunks {
                stream.push_mock(black_box(chunk)).unwrap();
            }

            stream.finalize_mock().unwrap()
        })
    });

    // Batch mode (process all at once)
    let full_audio: Vec<f32> = chunks.iter()
        .flat_map(|chunk| chunk.data().clone())
        .collect();
    let combined = AudioBuffer::from_vec(full_audio);

    group.bench_function("batch", |b| {
        b.iter(|| {
            engine.run_sentiment_graph_mock(black_box(&combined)).unwrap()
        })
    });

    group.finish();
}

// Benchmark: Interruption detection latency
fn bench_interruption_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("interruption_detection");

    let engine = GPUEngine::new_mock().unwrap();

    // Normal speech
    let normal_audio = AudioBuffer::generate_test_data(1000);

    // Interrupted speech (with sudden energy change)
    let interrupted_audio = AudioBuffer::generate_interrupted_test_data(1000);

    group.bench_function("normal_speech", |b| {
        b.iter(|| {
            engine.detect_interruption_mock(black_box(&normal_audio)).unwrap()
        })
    });

    group.bench_function("interrupted_speech", |b| {
        b.iter(|| {
            engine.detect_interruption_mock(black_box(&interrupted_audio)).unwrap()
        })
    });

    group.finish();
}

// Benchmark: Rate equilibrium calculation
fn bench_rate_equilibrium(c: &mut Criterion) {
    let mut group = c.benchmark_group("rate_equilibrium");

    let engine = GPUEngine::new_mock().unwrap();

    // Test different token rates
    for tokens_per_second in [10, 50, 100, 200].iter() {
        let tokens: Vec<i32> = (0..*tokens_per_second).map(|_| rand::random()).collect();

        group.bench_with_input(
            BenchmarkId::new("tokens_per_sec", tokens_per_second),
            tokens_per_second,
            |b, _| {
                b.iter(|| {
                    engine.calculate_rate_equilibrium_mock(black_box(&tokens)).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sentiment_inference,
    bench_sentiment_stages,
    bench_realtime_sentiment,
    bench_concurrent_users,
    bench_vad_aggregation,
    bench_equilibrium_composition,
    bench_context_similarity,
    bench_embedding_computation,
    bench_long_conversation,
    bench_cuda_graph_vs_traditional,
    bench_streaming_mode,
    bench_interruption_detection,
    bench_rate_equilibrium
);

criterion_main!(benches);
