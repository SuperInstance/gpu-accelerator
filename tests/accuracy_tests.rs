// Accuracy Tests for gpu-accelerator
// Validates numerical correctness of GPU computations vs CPU

#[cfg(test)]
mod accuracy_tests {
    use gpu_accelerator::{
        graph::CudaGraph,
        dpx::DPXContext,
        engine::GPUEngine,
        types::AudioBuffer,
    };
    use float_cmp::ApproxEq;

    const EPSILON_F32: f32 = 1e-5;
    const EPSILON_F64: f64 = 1e-9;

    // Test sentiment inference accuracy (GPU vs CPU)
    #[test]
    fn test_sentiment_vs_cpu() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        // Create test audio
        let audio = AudioBuffer::from_test_data();

        // GPU inference
        let gpu_result = gpu_engine.run_sentiment(&audio).unwrap();

        // CPU inference
        let cpu_result = cpu_engine.run_sentiment(&audio).unwrap();

        // Compare results
        assert!(
            gpu_result.valence.approx_eq(cpu_result.valence, (EPSILON_F32, 2)),
            "Valence mismatch: GPU={}, CPU={}",
            gpu_result.valence,
            cpu_result.valence
        );

        assert!(
            gpu_result.arousal.approx_eq(cpu_result.arousal, (EPSILON_F32, 2)),
            "Arousal mismatch: GPU={}, CPU={}",
            gpu_result.arousal,
            cpu_result.arousal
        );
    }

    // Test VAD score accuracy
    #[test]
    fn test_vad_accuracy() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        // Create test audio
        let audio = AudioBuffer::from_test_data();

        // GPU VAD
        let gpu_vad = gpu_engine.run_vad(&audio).unwrap();

        // CPU VAD
        let cpu_vad = cpu_engine.run_vad(&audio).unwrap();

        // Compare speech probability
        assert!(
            gpu_vad.speech_probability.approx_eq(cpu_vad.speech_probability, (EPSILON_F32, 2)),
            "VAD probability mismatch: GPU={}, CPU={}",
            gpu_vad.speech_probability,
            cpu_vad.speech_probability
        );
    }

    // Test DPX vs standard DP algorithm
    #[test]
    fn test_dpx_vs_standard_dp() {
        let context = DPXContext::new();
        context.initialize_mock();

        // Create cost matrix for path optimization
        let costs = vec![
            vec![1.0f32, 3.0, 1.0, 5.0],
            vec![2.0, 1.0, 4.0, 3.0],
            vec![5.0, 4.0, 1.0, 2.0],
        ];

        // DPX-accelerated
        let dpx_path = context.compute_optimal_path(&costs).unwrap();

        // Standard DP (CPU reference)
        let cpu_path = compute_optimal_path_cpu(&costs);

        // Paths should be identical
        assert_eq!(dpx_path.len(), cpu_path.len());

        for i in 0..dpx_path.len() {
            assert_eq!(
                dpx_path[i], cpu_path[i],
                "Path mismatch at index {}: DPX={}, CPU={}",
                i, dpx_path[i], cpu_path[i]
            );
        }

        // Costs should match exactly
        let dpx_cost: f32 = dpx_path.iter()
            .enumerate()
            .map(|(col, &row)| costs[row][col])
            .sum();

        let cpu_cost: f32 = cpu_path.iter()
            .enumerate()
            .map(|(col, &row)| costs[row][col])
            .sum();

        assert_eq!(dpx_cost, cpu_cost);
    }

    // Test embedding computation accuracy
    #[test]
    fn test_embedding_accuracy() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        // Create test tokens
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // GPU embeddings
        let gpu_embeddings = gpu_engine.compute_embeddings(&tokens).unwrap();

        // CPU embeddings
        let cpu_embeddings = cpu_engine.compute_embeddings(&tokens).unwrap();

        assert_eq!(gpu_embeddings.len(), cpu_embeddings.len());

        // Compare each embedding vector
        for (gpu_emb, cpu_emb) in gpu_embeddings.iter().zip(cpu_embeddings.iter()) {
            assert_eq!(gpu_emb.len(), cpu_emb.len());

            for (gpu_val, cpu_val) in gpu_emb.iter().zip(cpu_emb.iter()) {
                assert!(
                    gpu_val.approx_eq(cpu_val, (EPSILON_F32, 2)),
                    "Embedding value mismatch: GPU={}, CPU={}",
                    gpu_val,
                    cpu_val
                );
            }
        }
    }

    // Test cosine similarity accuracy
    #[test]
    fn test_cosine_similarity_accuracy() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        // Create test embeddings
        let query = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let contexts = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5], // Identical to query
            vec![0.5, 0.4, 0.3, 0.2, 0.1], // Reversed
            vec![1.0, 0.0, 0.0, 0.0, 0.0], // Orthogonal-ish
        ];

        // GPU similarities
        let gpu_sims = gpu_engine.compute_similarities(&query, &contexts).unwrap();

        // CPU similarities
        let cpu_sims = cpu_engine.compute_similarities(&query, &contexts).unwrap();

        assert_eq!(gpu_sims.len(), cpu_sims.len());

        for (i, (gpu_sim, cpu_sim)) in gpu_sims.iter().zip(cpu_sims.iter()).enumerate() {
            assert!(
                gpu_sim.approx_eq(cpu_sim, (EPSILON_F32, 2)),
                "Similarity mismatch at index {}: GPU={}, CPU={}",
                i, gpu_sim, cpu_sim
            );
        }

        // First context should have similarity = 1.0 (identical)
        assert!(gpu_sims[0].approx_eq(&1.0, (EPSILON_F32, 2)));
    }

    // Test matrix multiplication accuracy
    #[test]
    fn test_matrix_multiplication_accuracy() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        // Create test matrices
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 row-major
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2 row-major

        // GPU multiplication
        let gpu_c = gpu_engine.matmul(&a, &b, 2, 2).unwrap();

        // CPU multiplication
        let cpu_c = cpu_engine.matmul(&a, &b, 2, 2).unwrap();

        assert_eq!(gpu_c.len(), cpu_c.len());

        for (i, (gpu_val, cpu_val)) in gpu_c.iter().zip(cpu_c.iter()).enumerate() {
            assert!(
                gpu_val.approx_eq(cpu_val, (EPSILON_F32, 2)),
                "Matrix element mismatch at index {}: GPU={}, CPU={}",
                i, gpu_val, cpu_val
            );
        }

        // Verify result manually: [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50]
        assert!(gpu_c[0].approx_eq(&19.0, (EPSILON_F32, 2)));
        assert!(gpu_c[1].approx_eq(&22.0, (EPSILON_F32, 2)));
        assert!(gpu_c[2].approx_eq(&43.0, (EPSILON_F32, 2)));
        assert!(gpu_c[3].approx_eq(&50.0, (EPSILON_F32, 2)));
    }

    // Test VAD score aggregation accuracy
    #[test]
    fn test_vad_aggregation_accuracy() {
        let context = DPXContext::new();
        context.initialize_mock();

        let vad_scores = vec![0.8f32, 0.9, 0.7, 0.95, 0.85, 0.9, 0.88, 0.92];

        // DPX aggregation
        let dpx_result = context.aggregate_vad_scores(&vad_scores).unwrap();

        // CPU aggregation
        let cpu_min = vad_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let cpu_max = vad_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let cpu_mean: f32 = vad_scores.iter().sum::<f32>() / vad_scores.len() as f32;

        assert!(
            dpx_result.min.approx_eq(&cpu_min, (EPSILON_F32, 2)),
            "Min mismatch: DPX={}, CPU={}",
            dpx_result.min,
            cpu_min
        );

        assert!(
            dpx_result.max.approx_eq(&cpu_max, (EPSILON_F32, 2)),
            "Max mismatch: DPX={}, CPU={}",
            dpx_result.max,
            cpu_max
        );

        assert!(
            dpx_result.mean.approx_eq(&cpu_mean, (EPSILON_F32, 2)),
            "Mean mismatch: DPX={}, CPU={}",
            dpx_result.mean,
            cpu_mean
        );
    }

    // Test equilibrium composition accuracy
    #[test]
    fn test_equilibrium_composition_accuracy() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        let rate_weight = 0.3f32;
        let context_weight = 0.4f32;
        let sentiment_weight = 0.3f32;

        let rate_constraint = 0.8f32;
        let context_constraint = 0.7f32;
        let sentiment_constraint = 0.9f32;

        // GPU composition
        let gpu_eq = gpu_engine
            .compose_equilibrium(
                rate_weight,
                context_weight,
                sentiment_weight,
                rate_constraint,
                context_constraint,
                sentiment_constraint,
            )
            .unwrap();

        // CPU composition
        let cpu_eq = cpu_engine
            .compose_equilibrium(
                rate_weight,
                context_weight,
                sentiment_weight,
                rate_constraint,
                context_constraint,
                sentiment_constraint,
            )
            .unwrap();

        assert!(
            gpu_eq.approx_eq(cpu_eq, (EPSILON_F32, 2)),
            "Equilibrium mismatch: GPU={}, CPU={}",
            gpu_eq,
            cpu_eq
        );

        // Verify against manual computation
        let expected = rate_weight * rate_constraint
            + context_weight * context_constraint
            + sentiment_weight * sentiment_constraint;

        assert!(
            gpu_eq.approx_eq(expected, (EPSILON_F32, 2)),
            "Equilibrium doesn't match expected value: GPU={}, Expected={}",
            gpu_eq,
            expected
        );
    }

    // Test floating-point precision preservation
    #[test]
    fn test_floating_point_precision() {
        let gpu_engine = GPUEngine::new_mock().unwrap();

        // Test various precision-sensitive operations
        let test_values: Vec<f32> = vec![
            0.0,
            1.0,
            -1.0,
            0.5,
            0.123456789,
            1e-10,
            1e10,
            f32::MIN_POSITIVE,
            f32::MAX,
        ];

        for &val in &test_values {
            // Upload to GPU
            let gpu_buffer = gpu_engine.upload_scalar(val).unwrap();

            // Download back
            let result = gpu_engine.download_scalar(&gpu_buffer).unwrap();

            // Should be exactly equal for these operations
            if val.is_finite() && val > 1e-6 && val < 1e6 {
                assert_eq!(val, result, "Precision loss for value: {}", val);
            }
        }
    }

    // Test numerical stability
    #[test]
    fn test_numerical_stability() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        // Test operations that can cause overflow/underflow
        let large_vec: Vec<f32> = (0..1000).map(|i| 1e30 / (i + 1) as f32).collect();

        // GPU reduction
        let gpu_sum = gpu_engine.sum(&large_vec).unwrap();

        // CPU reduction (with Kahan summation for accuracy)
        let cpu_sum = cpu_engine.sum(&large_vec).unwrap();

        // Should be close (within 1% relative error)
        let relative_error = (gpu_sum - cpu_sum).abs() / cpu_sum.abs();
        assert!(relative_error < 0.01, "Relative error too large: {}", relative_error);
    }

    // Test DPX exact match for integer operations
    #[test]
    fn test_dpx_exact_match_integer() {
        let context = DPXContext::new();
        context.initialize_mock();

        let data: Vec<i32> = vec![100, 50, 200, 75, 150];

        // DPX min/max
        let (dpx_min, dpx_max) = context.min_max_int(&data).unwrap();

        // CPU reference
        let cpu_min = data.iter().min().unwrap();
        let cpu_max = data.iter().max().unwrap();

        // Should be EXACT match for integers
        assert_eq!(dpx_min, *cpu_min);
        assert_eq!(dpx_max, *cpu_max);
    }

    // Test softmax accuracy
    #[test]
    fn test_softmax_accuracy() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        // GPU softmax
        let gpu_softmax = gpu_engine.softmax(&logits).unwrap();

        // CPU softmax
        let cpu_softmax = cpu_engine.softmax(&logits).unwrap();

        assert_eq!(gpu_softmax.len(), cpu_softmax.len());

        // Compare each value
        for (i, (gpu_val, cpu_val)) in gpu_softmax.iter().zip(cpu_softmax.iter()).enumerate() {
            assert!(
                gpu_val.approx_eq(cpu_val, (EPSILON_F32, 2)),
                "Softmax mismatch at index {}: GPU={}, CPU={}",
                i, gpu_val, cpu_val
            );
        }

        // Verify softmax properties
        let sum: f32 = gpu_softmax.iter().sum();
        assert!(sum.approx_eq(&1.0, (EPSILON_F32, 2)), "Softmax doesn't sum to 1: sum={}", sum);
    }

    // Helper function: CPU reference DP implementation
    fn compute_optimal_path_cpu(costs: &Vec<Vec<f32>>) -> Vec<usize> {
        let rows = costs.len();
        let cols = costs[0].len();

        // DP table
        let mut dp = vec![vec![0.0f32; cols]; rows];

        // Initialize first column
        for r in 0..rows {
            dp[r][0] = costs[r][0];
        }

        // Fill DP table
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

        // Backtrack to find path
        let mut path = vec![0usize; cols];
        let mut current_row = (0..rows)
            .min_by(|&&a, &&b| dp[a].last().partial_cmp(&dp[b].last()).unwrap())
            .unwrap();

        path[cols - 1] = current_row;

        for c in (1..cols).rev() {
            let min_val = dp[current_row][c];

            current_row = if current_row == 0 {
                if dp[current_row + 1][c - 1] + 1e-6 < min_val {
                    current_row + 1
                } else {
                    current_row
                }
            } else if current_row == rows - 1 {
                if dp[current_row - 1][c - 1] + 1e-6 < min_val {
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

    // Test batch processing accuracy
    #[test]
    fn test_batch_processing_accuracy() {
        let gpu_engine = GPUEngine::new_mock().unwrap();
        let cpu_engine = gpu_engine.cpu_reference().unwrap();

        // Create batch
        let batch: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        // GPU batch processing
        let gpu_results = gpu_engine.process_batch(&batch).unwrap();

        // CPU batch processing
        let cpu_results = cpu_engine.process_batch(&batch).unwrap();

        assert_eq!(gpu_results.len(), cpu_results.len());

        for (i, (gpu_res, cpu_res)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            assert!(
                gpu_res.value.approx_eq(&cpu_res.value, (EPSILON_F32, 2)),
                "Batch result mismatch at index {}: GPU={}, CPU={}",
                i, gpu_res.value, cpu_res.value
            );
        }
    }

    // Test reproducibility
    #[test]
    fn test_reproducibility() {
        let gpu_engine = GPUEngine::new_mock().unwrap();

        let input = vec![0.5f32, 0.6, 0.7, 0.8];

        // Run same computation twice
        let result1 = gpu_engine.compute(&input).unwrap();
        let result2 = gpu_engine.compute(&input).unwrap();

        // Results should be identical
        assert_eq!(result1.len(), result2.len());

        for (i, (v1, v2)) in result1.iter().zip(result2.iter()).enumerate() {
            assert_eq!(
                v1, v2,
                "Results not reproducible at index {}: run1={}, run2={}",
                i, v1, v2
            );
        }
    }
}
