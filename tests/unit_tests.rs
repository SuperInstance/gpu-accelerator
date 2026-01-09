// Unit Tests for gpu-accelerator
// Tests individual components without requiring GPU hardware

#[cfg(test)]
mod unit_tests {
    use gpu_accelerator::{
        graph::CudaGraph,
        dpx::{DPXContext, DPXInstruction},
        memory::{GPUBuffer, MemoryPool},
        error::{GPUError, Result},
    };
    use std::time::Duration;

    // Test CUDA Graph creation logic (mocked)
    #[test]
    fn test_graph_creation() {
        // Test graph structure creation without actual GPU
        let graph = CudaGraph::new("test_graph");

        assert_eq!(graph.name(), "test_graph");
        assert!(!graph.is_captured());
        assert_eq!(graph.kernel_count(), 0);
    }

    // Test kernel addition to graph
    #[test]
    fn test_kernel_addition() {
        let mut graph = CudaGraph::new("test_graph");

        // Add mock kernel
        graph.add_kernel("vad_kernel", 256, 512);

        assert_eq!(graph.kernel_count(), 1);
        assert!(graph.has_kernel("vad_kernel"));
    }

    // Test graph validation
    #[test]
    fn test_graph_validation() {
        let mut graph = CudaGraph::new("test_graph");

        // Empty graph should not be valid
        assert!(!graph.validate().is_ok());

        // Add kernel
        graph.add_kernel("test_kernel", 256, 512);

        // Should now be valid
        assert!(graph.validate().is_ok());
    }

    // Test graph parameter management
    #[test]
    fn test_graph_parameters() {
        let mut graph = CudaGraph::new("param_test");

        graph.add_parameter("learning_rate", 0.001f32);
        graph.add_parameter("batch_size", 32usize);

        assert_eq!(graph.parameter_count(), 2);

        let lr = graph.get_parameter::<f32>("learning_rate").unwrap();
        assert!((lr - 0.001).abs() < 1e-6);

        let bs = graph.get_parameter::<usize>("batch_size").unwrap();
        assert_eq!(bs, 32);
    }

    // Test DPX instruction wrapper API
    #[test]
    fn test_dpx_instruction_wrapper() {
        let context = DPXContext::new();

        // Test min instruction
        let min_inst = DPXInstruction::Min(10, 20);
        assert_eq!(min_inst.execute(), 10);

        // Test max instruction
        let max_inst = DPXInstruction::Max(10, 20);
        assert_eq!(max_inst.execute(), 20);

        // Test compare instruction
        let cmp_inst = DPXInstruction::Compare(15, 20);
        assert!(cmp_inst.execute() < 0);
    }

    // Test DPX context state
    #[test]
    fn test_dpx_context_state() {
        let context = DPXContext::new();

        assert!(!context.is_initialized());
        assert_eq!(context.device_count(), 0);

        // Mock initialization
        context.initialize_mock();

        assert!(context.is_initialized());
        assert!(context.device_count() > 0);
    }

    // Test DPX instruction sequence
    #[test]
    fn test_dpx_instruction_sequence() {
        let mut context = DPXContext::new();
        context.initialize_mock();

        // Create sequence: min(max(a,b), min(c,d))
        let seq = context.create_sequence();

        seq.add_instruction(DPXInstruction::Max(10, 20)); // -> 20
        seq.add_instruction(DPXInstruction::Min(5, 15));  // -> 5
        seq.add_instruction(DPXInstruction::Min(0, 1));   // -> min(20, 5) = 5

        let result = seq.execute().unwrap();
        assert_eq!(result, 5);
    }

    // Test GPU memory allocation logic
    #[test]
    fn test_memory_allocation() {
        let pool = MemoryPool::new(1024 * 1024); // 1MB pool

        assert_eq!(pool.total_size(), 1024 * 1024);
        assert_eq!(pool.used_size(), 0);
        assert!(!pool.is_fragmented());

        // Allocate buffer
        let buffer = pool.allocate(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
        assert_eq!(pool.used_size(), 1024);
    }

    // Test memory deallocation
    #[test]
    fn test_memory_deallocation() {
        let pool = MemoryPool::new(1024 * 1024);

        let buffer = pool.allocate(1024).unwrap();
        assert_eq!(pool.used_size(), 1024);

        drop(buffer);
        assert_eq!(pool.used_size(), 0);
    }

    // Test memory pool fragmentation
    #[test]
    fn test_memory_fragmentation() {
        let pool = MemoryPool::new(1024);

        let buf1 = pool.allocate(256).unwrap();
        let buf2 = pool.allocate(256).unwrap();
        let buf3 = pool.allocate(256).unwrap();

        assert_eq!(pool.used_size(), 768);

        // Free middle buffer
        drop(buf2);

        assert!(pool.is_fragmented());

        // Compact should reduce fragmentation
        pool.compact();
        assert!(!pool.is_fragmented());
    }

    // Test buffer alignment
    #[test]
    fn test_buffer_alignment() {
        let pool = MemoryPool::new(1024);

        let buf1 = pool.allocate_aligned(256, 256).unwrap();
        assert_eq!(buf1.size(), 256);
        assert_eq!(buf1.alignment(), 256);

        let buf2 = pool.allocate_aligned(512, 512).unwrap();
        assert_eq!(buf2.size(), 512);
        assert_eq!(buf2.alignment(), 512);
    }

    // Test error handling - out of memory
    #[test]
    fn test_error_out_of_memory() {
        let pool = MemoryPool::new(1024);

        // Allocate more than pool size
        let result = pool.allocate(2048);

        assert!(result.is_err());
        match result {
            Err(GPUError::OutOfMemory(size)) => assert_eq!(size, 2048),
            _ => panic!("Expected OutOfMemory error"),
        }
    }

    // Test error handling - invalid graph
    #[test]
    fn test_error_invalid_graph() {
        let graph = CudaGraph::new("empty_graph");

        // Try to execute empty graph
        let result = graph.execute();

        assert!(result.is_err());
        match result {
            Err(GPUError::InvalidGraph(msg)) => {
                assert!(msg.contains("empty") || msg.contains("no kernels"));
            }
            _ => panic!("Expected InvalidGraph error"),
        }
    }

    // Test error handling - DPX not available
    #[test]
    fn test_error_dpx_unavailable() {
        let context = DPXContext::new();

        // Don't initialize, try to execute DPX
        let result = context.execute_dpx(DPXInstruction::Min(1, 2));

        assert!(result.is_err());
        match result {
            Err(GPUError::DPXNotAvailable) => (),
            _ => panic!("Expected DPXNotAvailable error"),
        }
    }

    // Test error handling - kernel not found
    #[test]
    fn test_error_kernel_not_found() {
        let mut graph = CudaGraph::new("test_graph");
        graph.add_kernel("existing_kernel", 256, 512);

        // Try to launch non-existent kernel
        let result = graph.launch_kernel("missing_kernel", &[]);

        assert!(result.is_err());
        match result {
            Err(GPUError::KernelNotFound(name)) => assert_eq!(name, "missing_kernel"),
            _ => panic!("Expected KernelNotFound error"),
        }
    }

    // Test timeout configuration
    #[test]
    fn test_timeout_configuration() {
        let graph = CudaGraph::new("timeout_test");

        graph.set_timeout(Duration::from_secs(5));

        assert_eq!(graph.timeout(), Some(Duration::from_secs(5)));

        // Timeout too short
        graph.set_timeout(Duration::from_micros(1));
        assert!(graph.timeout().is_some());
    }

    // Test graph metadata
    #[test]
    fn test_graph_metadata() {
        let mut graph = CudaGraph::new("metadata_test");

        graph.set_metadata("version", "1.0");
        graph.set_metadata("author", "test_suite");
        graph.add_kernel("kernel1", 256, 512);

        assert_eq!(graph.get_metadata("version"), Some("1.0".to_string()));
        assert_eq!(graph.get_metadata("author"), Some("test_suite".to_string()));
        assert_eq!(graph.get_metadata("missing"), None);
    }

    // Test buffer copy metadata
    #[test]
    fn test_buffer_copy_metadata() {
        let pool = MemoryPool::new(1024);
        let src = pool.allocate(512).unwrap();
        let dst = pool.allocate(512).unwrap();

        // Mock copy operation
        let copy_info = pool.prepare_copy(&src, &dst).unwrap();

        assert_eq!(copy_info.src_size(), 512);
        assert_eq!(copy_info.dst_size(), 512);
        assert_eq!(copy_info.bytes_to_copy(), 512);
    }

    // Test multiple graph instances
    #[test]
    fn test_multiple_graph_instances() {
        let graph1 = CudaGraph::new("graph1");
        let graph2 = CudaGraph::new("graph2");

        assert_ne!(graph1.id(), graph2.id());

        graph1.add_kernel("kernel1", 256, 512);
        graph2.add_kernel("kernel2", 512, 1024);

        assert_eq!(graph1.kernel_count(), 1);
        assert_eq!(graph2.kernel_count(), 1);
    }

    // Test DPX instruction batching
    #[test]
    fn test_dpx_instruction_batching() {
        let context = DPXContext::new();
        context.initialize_mock();

        let batch = context.create_batch();

        // Add 100 instructions
        for i in 0..100 {
            batch.add(DPXInstruction::Min(i, i + 1));
        }

        assert_eq!(batch.len(), 100);

        let results = batch.execute_all().unwrap();
        assert_eq!(results.len(), 100);

        // Verify first result
        assert_eq!(results[0], 0);
    }

    // Test memory pool statistics
    #[test]
    fn test_memory_pool_statistics() {
        let pool = MemoryPool::new(1024 * 1024);

        let stats = pool.statistics();

        assert_eq!(stats.total_allocations(), 0);
        assert_eq!(stats.active_allocations(), 0);
        assert_eq!(stats.total_bytes_allocated(), 0);

        let _buf1 = pool.allocate(1024).unwrap();
        let _buf2 = pool.allocate(2048).unwrap();

        let stats = pool.statistics();

        assert_eq!(stats.total_allocations(), 2);
        assert_eq!(stats.active_allocations(), 2);
        assert_eq!(stats.total_bytes_allocated(), 3072);
    }
}
