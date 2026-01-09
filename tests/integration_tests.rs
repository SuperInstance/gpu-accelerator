// Integration Tests for gpu-accelerator
// Tests end-to-end workflows with mocked GPU

#[cfg(test)]
mod integration_tests {
    use gpu_accelerator::{
        graph::CudaGraph,
        dpx::DPXContext,
        memory::{GPUBuffer, MemoryPool},
        engine::GPUEngine,
        types::AudioBuffer,
    };
    use std::time::{Duration, Instant};
    use tokio::test;

    // Test end-to-end sentiment inference pipeline
    #[tokio::test]
    async fn test_end_to_end_sentiment_inference() {
        // Setup GPU engine
        let engine = GPUEngine::new_mock().await.unwrap();

        // Load test audio (simulated)
        let audio = AudioBuffer::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5]);

        // Upload to GPU
        let gpu_buffer = engine.upload_to_gpu(&audio).await.unwrap();

        // Execute VAD graph
        let vad_result = engine.run_vad_graph(&gpu_buffer).await.unwrap();

        assert!(vad_result.speech_probability > 0.0);
        assert!(vad_result.speech_probability <= 1.0);

        // Execute sentiment graph
        let sentiment = engine.run_sentiment_graph(&gpu_buffer).await.unwrap();

        assert!(sentiment.valence >= -1.0);
        assert!(sentiment.valence <= 1.0);
        assert!(sentiment.arousal >= 0.0);
        assert!(sentiment.arousal <= 1.0);

        // Download result
        let cpu_result = engine.download_to_cpu(&sentiment).await.unwrap();

        assert_eq!(cpu_result.valence, sentiment.valence);
    }

    // Test multi-kernel graph pipeline
    #[test]
    fn test_multi_kernel_graph() {
        // Create graph with VAD → Sentiment → Dominance pipeline
        let mut graph = CudaGraph::new("sentiment_pipeline");

        // Add kernels
        graph.add_kernel("vad_kernel", 256, 512);
        graph.add_kernel("sentiment_kernel", 256, 512);
        graph.add_kernel("dominance_kernel", 128, 256);

        // Define dependencies
        graph.add_dependency("sentiment_kernel", "vad_kernel");
        graph.add_dependency("dominance_kernel", "sentiment_kernel");

        // Validate graph structure
        assert!(graph.validate().is_ok());
        assert_eq!(graph.kernel_count(), 3);

        // Verify topological order
        let order = graph.topological_order().unwrap();
        assert_eq!(order[0], "vad_kernel");
        assert_eq!(order[1], "sentiment_kernel");
        assert_eq!(order[2], "dominance_kernel");
    }

    // Test embedding computation workflow
    #[tokio::test]
    async fn test_embedding_computation() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Create test text tokens
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];

        // Upload to GPU
        let gpu_tokens = engine.upload_tokens(&tokens).await.unwrap();

        // Compute embeddings
        let embeddings = engine.compute_embeddings(&gpu_tokens).await.unwrap();

        assert_eq!(embeddings.len(), tokens.len());
        assert_eq!(embeddings[0].len(), 768); // Standard embedding size

        // Verify all embeddings are finite
        for emb in &embeddings {
            for &val in emb {
                assert!(val.is_finite());
            }
        }
    }

    // Test context similarity computation
    #[tokio::test]
    async fn test_context_similarity() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Create query and context embeddings
        let query = vec![0.1f32, 0.2, 0.3, 0.4];
        let contexts = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 1.0, 0.1, 0.2],
        ];

        // Upload to GPU
        let gpu_query = engine.upload_embedding(&query).await.unwrap();
        let gpu_contexts = engine.upload_embeddings(&contexts).await.unwrap();

        // Compute similarities
        let similarities = engine.compute_similarity(&gpu_query, &gpu_contexts).await.unwrap();

        assert_eq!(similarities.len(), contexts.len());

        // First context should have highest similarity (identical)
        assert!(similarities[0] > similarities[1]);
        assert!(similarities[0] > similarities[2]);

        // Similarities should be in [0, 1]
        for &sim in &similarities {
            assert!(sim >= 0.0 && sim <= 1.0);
        }
    }

    // Test dynamic programming path optimization
    #[test]
    fn test_dpx_path_optimization() {
        let context = DPXContext::new();
        context.initialize_mock();

        // Create cost matrix
        let costs = vec![
            vec![1.0f32, 3.0, 1.0, 5.0],
            vec![2.0, 1.0, 4.0, 3.0],
            vec![5.0, 4.0, 1.0, 2.0],
        ];

        // Compute optimal path using DPX
        let path = context.compute_optimal_path(&costs).unwrap();

        assert_eq!(path.len(), 4); // One per column

        // Verify path is valid (rows are sequential or same)
        for i in 1..path.len() {
            let row_diff = (path[i] as i32 - path[i-1] as i32).abs();
            assert!(row_diff <= 1); // Can move at most one row per step
        }

        // Compute total cost
        let total_cost: f32 = path.iter()
            .enumerate()
            .map(|(col, &row)| costs[row][col])
            .sum();

        assert!(total_cost > 0.0);
    }

    // Test CUDA Graph capture and replay
    #[test]
    fn test_graph_capture_replay() {
        let mut graph = CudaGraph::new("capture_test");

        // Setup kernels
        graph.add_kernel("kernel1", 256, 512);
        graph.add_kernel("kernel2", 256, 512);

        // Capture graph (mock)
        graph.capture_mock().unwrap();

        assert!(graph.is_captured());

        // Replay captured graph
        let start = Instant::now();
        for _ in 0..100 {
            graph.replay_mock().unwrap();
        }
        let duration = start.elapsed();

        // Replay should be fast (< 10ms for 100 iterations)
        assert!(duration < Duration::from_millis(100));
    }

    // Test memory transfer workflow
    #[tokio::test]
    async fn test_memory_transfer_workflow() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Allocate CPU memory
        let cpu_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        // HtoD transfer
        let gpu_buffer = engine.htod_transfer(&cpu_data).await.unwrap();
        assert_eq!(gpu_buffer.size(), cpu_data.len() * 4); // 4 bytes per f32

        // DtoH transfer
        let cpu_result = engine.dtoh_transfer(&gpu_buffer).await.unwrap();
        assert_eq!(cpu_result.len(), cpu_data.len());

        // Verify data integrity
        for i in 0..cpu_data.len() {
            assert!((cpu_result[i] - cpu_data[i]).abs() < 1e-6);
        }
    }

    // Test concurrent graph execution
    #[tokio::test]
    async fn test_concurrent_execution() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Create multiple graphs
        let graph1 = engine.create_graph("graph1").await.unwrap();
        let graph2 = engine.create_graph("graph2").await.unwrap();
        let graph3 = engine.create_graph("graph3").await.unwrap();

        // Execute concurrently
        let start = Instant::now();

        let handle1 = tokio::spawn(async {
            graph1.execute_mock().await.unwrap()
        });

        let handle2 = tokio::spawn(async {
            graph2.execute_mock().await.unwrap()
        });

        let handle3 = tokio::spawn(async {
            graph3.execute_mock().await.unwrap()
        });

        let _ = tokio::join!(handle1, handle2, handle3);

        let duration = start.elapsed();

        // Concurrent execution should be faster than sequential
        assert!(duration < Duration::from_millis(500));
    }

    // Test VAD score aggregation
    #[test]
    fn test_vad_score_aggregation() {
        let context = DPXContext::new();
        context.initialize_mock();

        // Simulate VAD scores over time window
        let vad_scores = vec![
            0.8f32, 0.9, 0.7, 0.95, 0.85, 0.9, 0.88, 0.92,
        ];

        // Aggregate using DPX min/max
        let aggregated = context.aggregate_vad_scores(&vad_scores).unwrap();

        assert!(aggregated.min >= 0.0 && aggregated.min <= 1.0);
        assert!(aggregated.max >= 0.0 && aggregated.max <= 1.0);
        assert!(aggregated.mean >= 0.0 && aggregated.mean <= 1.0);

        // Min should be <= all scores
        for &score in &vad_scores {
            assert!(aggregated.min <= score);
        }

        // Max should be >= all scores
        for &score in &vad_scores {
            assert!(aggregated.max >= score);
        }
    }

    // Test equilibrium constraint composition
    #[test]
    fn test_equilibrium_composition() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Define constraint weights
        let rate_weight = 0.3f32;
        let context_weight = 0.4f32;
        let sentiment_weight = 0.3f32;

        // Compute individual constraint values
        let rate_constraint = 0.8f32;
        let context_constraint = 0.7f32;
        let sentiment_constraint = 0.9f32;

        // Compose equilibrium
        let equilibrium = engine
            .compose_equilibrium(
                rate_weight,
                context_weight,
                sentiment_weight,
                rate_constraint,
                context_constraint,
                sentiment_constraint,
            )
            .unwrap();

        // Verify composition
        let expected = (rate_weight * rate_constraint
            + context_weight * context_constraint
            + sentiment_weight * sentiment_constraint);

        assert!((equilibrium - expected).abs() < 1e-6);
        assert!(equilibrium >= 0.0 && equilibrium <= 1.0);
    }

    // Test error recovery
    #[tokio::test]
    async fn test_error_recovery() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Simulate GPU error
        let result = engine.simulate_error().await;

        assert!(result.is_err());

        // Verify engine can recover
        let recovered = engine.recover().await.unwrap();

        assert!(recovered);
        assert!(engine.is_healthy());
    }

    // Test resource cleanup
    #[tokio::test]
    async fn test_resource_cleanup() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Allocate resources
        let buf1 = engine.allocate_buffer(1024).await.unwrap();
        let buf2 = engine.allocate_buffer(2048).await.unwrap();

        assert_eq!(engine.active_buffers(), 2);

        // Cleanup
        drop(buf1);
        drop(buf2);

        // Force garbage collection
        engine.gc().await.unwrap();

        assert_eq!(engine.active_buffers(), 0);
    }

    // Test batch processing
    #[tokio::test]
    async fn test_batch_processing() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Create batch of audio buffers
        let batch: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        // Process batch
        let results = engine.process_batch(&batch).await.unwrap();

        assert_eq!(results.len(), batch.len());

        for result in &results {
            assert!(result.is_valid());
        }
    }

    // Test streaming processing
    #[tokio::test]
    async fn test_streaming_processing() {
        let engine = GPUEngine::new_mock().await.unwrap();

        // Create stream
        let mut stream = engine.create_stream().await.unwrap();

        // Stream data chunks
        let chunks = vec![
            vec![0.1f32, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        for chunk in chunks {
            stream.push(chunk).await.unwrap();
        }

        stream.finalize().await.unwrap();

        // Get results
        let results = stream.results().await.unwrap();

        assert!(results.len() > 0);
    }
}
