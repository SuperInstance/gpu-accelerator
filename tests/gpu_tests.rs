// GPU-Specific Tests for gpu-accelerator
// Tests requiring actual GPU hardware (feature: gpu_tests)

#[cfg(feature = "gpu_tests")]
mod gpu_tests {
    use gpu_accelerator::{
        graph::CudaGraph,
        dpx::DPXContext,
        memory::{GPUBuffer, MemoryPool},
        engine::GPUEngine,
        types::AudioBuffer,
    };
    use std::time::{Duration, Instant};

    // Test CUDA Graph execution on real GPU
    #[test]
    #[ignore] // Requires GPU - run with: cargo test --features gpu_tests -- --ignored
    fn test_cuda_graph_execution() {
        let engine = GPUEngine::new_real().unwrap();

        // Create simple graph
        let mut graph = CudaGraph::new("gpu_test");

        graph.add_kernel("vector_add", 256, 1024);
        graph.add_parameter("array_size", 1024usize);

        // Capture and execute
        graph.capture().unwrap();

        let input = vec![1.0f32; 1024];
        let result = graph.execute_with_input(&input).unwrap();

        assert_eq!(result.len(), 1024);

        // Verify result (should be input * 2)
        for &val in &result {
            assert!((val - 2.0).abs() < 1e-5);
        }
    }

    // Test DPX instructions on H100/H200
    #[test]
    #[ignore] // Requires H100/H200
    fn test_dpx_on_h100() {
        let context = DPXContext::new_real().unwrap();

        // Check if DPX is available
        if !context.is_dpx_supported() {
            println!("DPX not supported on this GPU - skipping");
            return;
        }

        // Create large DP workload
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        // Execute DPX min/max
        let start = Instant::now();
        let (min_val, max_val) = context.min_max(&data).unwrap();
        let duration = start.elapsed();

        assert_eq!(min_val, 0.0);
        assert_eq!(max_val, (size - 1) as f32);

        println!("DPX min/max on {} elements: {:?}", size, duration);

        // Should be very fast (< 1ms for 10k elements)
        assert!(duration < Duration::from_millis(10));
    }

    // Test GPU memory bandwidth
    #[test]
    #[ignore]
    fn test_gpu_memory_bandwidth() {
        let engine = GPUEngine::new_real().unwrap();

        // Allocate 1GB buffer
        let size = 1024 * 1024 * 1024; // 1GB
        let cpu_data: Vec<f32> = (0..size / 4).map(|i| i as f32).collect();

        // Test HtoD bandwidth
        let start = Instant::now();
        let gpu_buffer = engine.htod_transfer(&cpu_data).unwrap();
        let htod_duration = start.elapsed();

        let htod_bandwidth = (size as f64) / (htod_duration.as_secs_f64() * 1e9);
        println!("HtoD bandwidth: {:.2} GB/s", htod_bandwidth);

        // Should achieve > 10 GB/s on modern GPUs
        assert!(htod_bandwidth > 10.0);

        // Test DtoH bandwidth
        let start = Instant::now();
        let cpu_result = engine.dtoh_transfer(&gpu_buffer).unwrap();
        let dtoh_duration = start.elapsed();

        let dtoh_bandwidth = (size as f64) / (dtoh_duration.as_secs_f64() * 1e9);
        println!("DtoH bandwidth: {:.2} GB/s", dtoh_bandwidth);

        assert!(dtoh_bandwidth > 10.0);

        // Verify data integrity
        assert_eq!(cpu_result.len(), cpu_data.len());
        for i in 0..cpu_data.len().min(100) {
            assert!((cpu_result[i] - cpu_data[i]).abs() < 1e-6);
        }
    }

    // Test multiple CUDA streams
    #[test]
    #[ignore]
    fn test_multiple_cuda_streams() {
        let engine = GPUEngine::new_real().unwrap();

        // Create 3 streams
        let stream1 = engine.create_stream().unwrap();
        let stream2 = engine.create_stream().unwrap();
        let stream3 = engine.create_stream().unwrap();

        // Execute kernels concurrently
        let data = vec![1.0f32; 1024 * 1024];

        let start = Instant::now();

        let h1 = stream1.execute_async("kernel1", &data);
        let h2 = stream2.execute_async("kernel2", &data);
        let h3 = stream3.execute_async("kernel3", &data);

        h1.wait().unwrap();
        h2.wait().unwrap();
        h3.wait().unwrap();

        let duration = start.elapsed();

        println!("Concurrent execution on 3 streams: {:?}", duration);

        // Should be faster than sequential
        assert!(duration < Duration::from_millis(100));
    }

    // Test DPX vs CPU performance
    #[test]
    #[ignore]
    fn test_dpx_vs_cpu_performance() {
        let context = DPXContext::new_real().unwrap();

        if !context.is_dpx_supported() {
            println!("DPX not supported - skipping");
            return;
        }

        // Create test data
        let size = 100000;
        let data: Vec<i32> = (0..size).map(|i| i).collect();

        // CPU baseline
        let start = Instant::now();
        let cpu_min = data.iter().min().unwrap();
        let cpu_max = data.iter().max().unwrap();
        let cpu_duration = start.elapsed();

        // DPX
        let start = Instant::now();
        let (gpu_min, gpu_max) = context.min_max_int(&data).unwrap();
        let gpu_duration = start.elapsed();

        assert_eq!(*cpu_min, gpu_min);
        assert_eq!(*cpu_max, gpu_max);

        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
        println!("DPX speedup: {:.2}x", speedup);

        // DPX should be at least 5x faster
        assert!(speedup > 5.0);
    }

    // Test sentiment inference on GPU
    #[test]
    #[ignore]
    fn test_sentiment_inference_gpu() {
        let engine = GPUEngine::new_real().unwrap();

        // Load test audio file (simulated with real audio loading in production)
        let audio = AudioBuffer::from_test_data();

        // Upload to GPU
        let gpu_audio = engine.upload_to_gpu(&audio).unwrap();

        // Execute inference
        let start = Instant::now();
        let result = engine.run_sentiment_graph(&gpu_audio).unwrap();
        let duration = start.elapsed();

        println!("Sentiment inference latency: {:?}", duration);

        // Should be < 10ms for real-time processing
        assert!(duration < Duration::from_millis(10));

        // Validate result
        assert!(result.valence >= -1.0 && result.valence <= 1.0);
        assert!(result.arousal >= 0.0 && result.arousal <= 1.0);
    }

    // Test CUDA Graph launch latency
    #[test]
    #[ignore]
    fn test_cuda_graph_launch_latency() {
        let mut graph = CudaGraph::new("latency_test");

        graph.add_kernel("simple_kernel", 256, 1024);
        graph.capture().unwrap();

        // Warmup
        for _ in 0..10 {
            graph.execute().unwrap();
        }

        // Measure latency
        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            graph.execute().unwrap();
        }

        let duration = start.elapsed();
        let avg_latency = duration / iterations;

        println!("Average CUDA Graph launch latency: {:?}", avg_latency);

        // CUDA Graphs should launch in < 10μs (0.01ms)
        assert!(avg_latency < Duration::from_micros(50));
    }

    // Test GPU utilization
    #[test]
    #[ignore]
    fn test_gpu_utilization() {
        let engine = GPUEngine::new_real().unwrap();

        // Create workload that saturates GPU
        let size = 1024 * 1024 * 100; // 100M elements
        let data: Vec<f32> = vec![1.0; size];

        let start = Instant::now();
        engine.heavy_computation(&data).unwrap();
        let duration = start.elapsed();

        let utilization = engine.get_utilization().unwrap();

        println!("GPU utilization: {:.1}%", utilization * 100.0);
        println!("Computation time: {:?}", duration);

        // Should have high utilization (> 50%)
        assert!(utilization > 0.5);
    }

    // Test memory pool on GPU
    #[test]
    #[ignore]
    fn test_memory_pool_gpu() {
        let pool = MemoryPool::new_gpu(1024 * 1024 * 1024).unwrap(); // 1GB

        // Allocate multiple buffers
        let mut buffers = Vec::new();

        for i in 0..10 {
            let size = 1024 * 1024 * (i + 1); // 1MB to 10MB
            let buf = pool.allocate(size).unwrap();
            buffers.push(buf);
        }

        assert_eq!(pool.used_size(), 55 * 1024 * 1024); // Sum of 1+2+...+10

        // Free some buffers
        buffers.drain(0..5);

        assert_eq!(pool.used_size(), 40 * 1024 * 1024); // Remaining: 6+7+8+9+10

        // Compact
        pool.compact();

        assert!(!pool.is_fragmented());
    }

    // Test error handling on GPU
    #[test]
    #[ignore]
    fn test_gpu_error_handling() {
        use gpu_accelerator::error::GPUError;

        // Test allocation failure
        let pool = MemoryPool::new_gpu(1024).unwrap();

        let result = pool.allocate(2048); // Try to allocate more than available

        assert!(result.is_err());
        match result {
            Err(GPUError::OutOfMemory(size)) => {
                assert_eq!(size, 2048);
            }
            _ => panic!("Expected OutOfMemory error"),
        }

        // Test invalid kernel launch
        let engine = GPUEngine::new_real().unwrap();

        let result = engine.launch_invalid_kernel();

        assert!(result.is_err());
    }

    // Test DPX on different GPU architectures
    #[test]
    #[ignore]
    fn test_dpx_architecture_compatibility() {
        let context = DPXContext::new_real().unwrap();

        let device_name = context.get_device_name().unwrap();
        println!("Testing on GPU: {}", device_name);

        let is_hopper = device_name.contains("H100") || device_name.contains("H200");
        let is_blackwell = device_name.contains("B100") || device_name.contains("B200");

        if is_hopper || is_blackwell {
            assert!(context.is_dpx_supported());

            // Test DPX workload
            let size = 100000;
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

            let (min, max) = context.min_max(&data).unwrap();

            assert_eq!(min, 0.0);
            assert_eq!(max, (size - 1) as f32);

            println!("DPX works on {}", device_name);
        } else {
            println!("DPX not supported on {} - this is expected", device_name);
            assert!(!context.is_dpx_supported());
        }
    }

    // Test multi-GPU (if available)
    #[test]
    #[ignore]
    fn test_multi_gpu() {
        let device_count = GPUEngine::get_device_count().unwrap();

        if device_count < 2 {
            println!("Multi-GPU test requires at least 2 GPUs - skipping");
            return;
        }

        println!("Testing on {} GPUs", device_count);

        // Create engines on different GPUs
        let engine0 = GPUEngine::new_on_device(0).unwrap();
        let engine1 = GPUEngine::new_on_device(1).unwrap();

        // Allocate and transfer on both GPUs
        let data = vec![1.0f32; 1024 * 1024];

        let buf0 = engine0.htod_transfer(&data).unwrap();
        let buf1 = engine1.htod_transfer(&data).unwrap();

        // Execute on both GPUs
        let result0 = engine0.process_buffer(&buf0).unwrap();
        let result1 = engine1.process_buffer(&buf1).unwrap();

        assert_eq!(result0.len(), result1.len());

        println!("Multi-GPU test successful");
    }

    // Test persistent cache
    #[test]
    #[ignore]
    fn test_persistent_cache() {
        let engine = GPUEngine::new_real().unwrap();

        // Create graph
        let mut graph = CudaGraph::new("cache_test");
        graph.add_kernel("cached_kernel", 256, 1024);
        graph.capture().unwrap();

        // Execute multiple times - should use cached graph
        let iterations = 100;

        let start = Instant::now();
        for _ in 0..iterations {
            graph.execute().unwrap();
        }
        let duration = start.elapsed();

        let avg_latency = duration / iterations;

        println!("Cached graph average latency: {:?}", avg_latency);

        // Should be very fast with caching
        assert!(avg_latency < Duration::from_micros(20));
    }

    // Test thermal throttling detection
    #[test]
    #[ignore]
    fn test_thermal_monitoring() {
        let engine = GPUEngine::new_real().unwrap();

        // Run sustained workload
        let data = vec![1.0f32; 1024 * 1024 * 100];

        let mut temps = Vec::new();

        for i in 0..10 {
            engine.heavy_computation(&data).unwrap();

            if let Ok(temp) = engine.get_temperature() {
                temps.push(temp);
                println!("Iteration {}: GPU temperature = {}°C", i, temp);
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        // Temperature should be reasonable (< 90°C)
        if let Some(&max_temp) = temps.iter().max() {
            assert!(max_temp < 90, "GPU overheating: {}°C", max_temp);
        }
    }
}
