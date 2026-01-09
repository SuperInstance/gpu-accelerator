//! GPU memory management

use crate::error::{GPUError, Result};
use std::collections::HashMap;

/// GPU buffer
#[derive(Debug, Clone)]
pub struct GPUBuffer {
    ptr: usize,
    size: usize,
    alignment: usize,
    is_pinned: bool,
}

impl GPUBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }
}

/// Memory pool for GPU allocations
#[derive(Debug)]
pub struct MemoryPool {
    total_size: usize,
    used_size: usize,
    buffers: Vec<GPUBuffer>,
    free_list: Vec<GPUBuffer>,
    next_buffer_id: usize,
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    total_allocations: usize,
    active_allocations: usize,
    total_bytes_allocated: usize,
}

impl MemoryPoolStats {
    pub fn total_allocations(&self) -> usize {
        self.total_allocations
    }

    pub fn active_allocations(&self) -> usize {
        self.active_allocations
    }

    pub fn total_bytes_allocated(&self) -> usize {
        self.total_bytes_allocated
    }
}

/// Copy information for memory transfers
#[derive(Debug, Clone)]
pub struct CopyInfo {
    src_size: usize,
    dst_size: usize,
    bytes_to_copy: usize,
}

impl CopyInfo {
    pub fn src_size(&self) -> usize {
        self.src_size
    }

    pub fn dst_size(&self) -> usize {
        self.dst_size
    }

    pub fn bytes_to_copy(&self) -> usize {
        self.bytes_to_copy
    }
}

impl MemoryPool {
    pub fn new(total_size: usize) -> Self {
        Self {
            total_size,
            used_size: 0,
            buffers: Vec::new(),
            free_list: Vec::new(),
            next_buffer_id: 0,
        }
    }

    pub fn total_size(&self) -> usize {
        self.total_size
    }

    pub fn used_size(&self) -> usize {
        self.used_size
    }

    pub fn allocate(&mut self, size: usize) -> Result<GPUBuffer> {
        if self.used_size + size > self.total_size {
            return Err(GPUError::OutOfMemory(size));
        }

        let buffer = GPUBuffer {
            ptr: self.next_buffer_id,
            size,
            alignment: 0,
            is_pinned: false,
        };

        self.next_buffer_id += 1;
        self.used_size += size;
        self.buffers.push(buffer.clone());

        Ok(buffer)
    }

    pub fn allocate_aligned(&mut self, size: usize, alignment: usize) -> Result<GPUBuffer> {
        let aligned_size = (size + alignment - 1) / alignment * alignment;

        if self.used_size + aligned_size > self.total_size {
            return Err(GPUError::OutOfMemory(aligned_size));
        }

        let buffer = GPUBuffer {
            ptr: self.next_buffer_id,
            size: aligned_size,
            alignment,
            is_pinned: false,
        };

        self.next_buffer_id += 1;
        self.used_size += aligned_size;
        self.buffers.push(buffer.clone());

        Ok(buffer)
    }

    pub fn prepare_copy(&self, src: &GPUBuffer, dst: &GPUBuffer) -> Result<CopyInfo> {
        let bytes_to_copy = src.size.min(dst.size);

        Ok(CopyInfo {
            src_size: src.size,
            dst_size: dst.size,
            bytes_to_copy,
        })
    }

    pub fn is_fragmented(&self) -> bool {
        self.free_list.len() > 0
    }

    pub fn compact(&mut self) {
        self.free_list.clear();
    }

    pub fn statistics(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            total_allocations: self.buffers.len(),
            active_allocations: self.buffers.len() - self.free_list.len(),
            total_bytes_allocated: self.used_size,
        }
    }

    pub fn fragmentation_ratio(&self) -> f32 {
        if self.used_size == 0 {
            0.0
        } else {
            self.free_list.len() as f32 / self.buffers.len() as f32
        }
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
        self.free_list.clear();
        self.used_size = 0;
    }

    pub fn new_gpu(total_size: usize) -> Result<Self> {
        Ok(Self::new(total_size))
    }
}

impl Drop for GPUBuffer {
    fn drop(&mut self) {
        // In real implementation, would free GPU memory here
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_allocation() {
        let pool = MemoryPool::new(1024 * 1024);

        assert_eq!(pool.total_size(), 1024 * 1024);
        assert_eq!(pool.used_size(), 0);

        let buffer = pool.allocate(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
        assert_eq!(pool.used_size(), 1024);
    }

    #[test]
    fn test_memory_deallocation() {
        let pool = MemoryPool::new(1024);

        let buffer = pool.allocate(1024).unwrap();
        assert_eq!(pool.used_size(), 1024);

        drop(buffer);
        // Used size doesn't decrease because we don't track deallocations in this simple mock
    }

    #[test]
    fn test_out_of_memory() {
        let pool = MemoryPool::new(1024);

        let result = pool.allocate(2048);

        assert!(result.is_err());
    }

    #[test]
    fn test_aligned_allocation() {
        let pool = MemoryPool::new(1024);

        let buffer = pool.allocate_aligned(256, 256).unwrap();
        assert_eq!(buffer.size(), 256);
        assert_eq!(buffer.alignment(), 256);
    }

    #[test]
    fn test_statistics() {
        let pool = MemoryPool::new(1024);

        let _buf1 = pool.allocate(512).unwrap();
        let _buf2 = pool.allocate(256).unwrap();

        let stats = pool.statistics();

        assert_eq!(stats.total_allocations(), 2);
        assert_eq!(stats.active_allocations(), 2);
        assert_eq!(stats.total_bytes_allocated(), 768);
    }

    #[test]
    fn test_compact() {
        let pool = MemoryPool::new(1024);

        let buf1 = pool.allocate(256).unwrap();
        let buf2 = pool.allocate(256).unwrap();
        drop(buf1);

        assert!(pool.is_fragmented());

        pool.compact();

        assert!(!pool.is_fragmented());
    }

    #[test]
    fn test_copy_info() {
        let pool = MemoryPool::new(1024);

        let src = pool.allocate(512).unwrap();
        let dst = pool.allocate(1024).unwrap();

        let info = pool.prepare_copy(&src, &dst).unwrap();

        assert_eq!(info.src_size(), 512);
        assert_eq!(info.dst_size(), 1024);
        assert_eq!(info.bytes_to_copy(), 512);
    }
}
