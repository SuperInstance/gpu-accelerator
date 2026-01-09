//! DPX (Data Processing Accelerator) instruction wrappers

use crate::error::{GPUError, Result};
use std::collections::HashMap;

/// DPX context for hardware-accelerated operations
#[derive(Debug, Clone)]
pub struct DPXContext {
    device_id: i32,
    initialized: bool,
    dpx_supported: bool,
}

/// DPX instruction
#[derive(Debug, Clone, Copy)]
pub enum DPXInstruction {
    Min(i32, i32),
    Max(i32, i32),
    Compare(i32, i32),
}

impl DPXInstruction {
    pub fn execute(&self) -> i32 {
        match self {
            DPXInstruction::Min(a, b) => *a.min(b),
            DPXInstruction::Max(a, b) => *a.max(b),
            DPXInstruction::Compare(a, b) => a.cmp(b).as_i32(),
        }
    }
}

/// VAD aggregation result
#[derive(Debug, Clone)]
pub struct VADAggregation {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

/// DPX instruction batch
pub struct DPXBatch {
    instructions: Vec<DPXInstruction>,
}

impl DPXBatch {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    pub fn add(&mut self, instruction: DPXInstruction) {
        self.instructions.push(instruction);
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn execute_all(&self) -> Result<Vec<i32>> {
        Ok(self.instructions.iter().map(|i| i.execute()).collect())
    }
}

/// DPX instruction sequence
pub struct DPXSequence {
    instructions: Vec<DPXInstruction>,
    context: HashMap<String, i32>,
}

impl DPXSequence {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            context: HashMap::new(),
        }
    }

    pub fn add_instruction(&mut self, instruction: DPXInstruction) {
        self.instructions.push(instruction);
    }

    pub fn execute(&self) -> Result<i32> {
        let mut result = 0;

        for instruction in &self.instructions {
            result = instruction.execute();
        }

        Ok(result)
    }
}

impl DPXContext {
    pub fn new() -> Self {
        Self {
            device_id: 0,
            initialized: false,
            dpx_supported: false,
        }
    }

    pub fn initialize_mock(&mut self) {
        self.initialized = true;
        self.dpx_supported = true;
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn device_count(&self) -> usize {
        if self.initialized { 1 } else { 0 }
    }

    pub fn is_dpx_supported(&self) -> bool {
        self.dpx_supported
    }

    pub fn min_max(&self, data: &[f32]) -> Result<(f32, f32)> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        Ok((min, max))
    }

    pub fn min_max_int(&self, data: &[i32]) -> Result<(i32, i32)> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        let min = *data.iter().min().ok_or_else(|| {
            GPUError::DPXExecutionFailed("empty data".to_string())
        })?;

        let max = *data.iter().max().ok_or_else(|| {
            GPUError::DPXExecutionFailed("empty data".to_string())
        })?;

        Ok((min, max))
    }

    pub fn min_max_mock(&self, data: &[f32]) -> Result<(f32, f32)> {
        self.min_max(data)
    }

    pub fn compute_optimal_path(&self, costs: &Vec<Vec<f32>>) -> Result<Vec<usize>> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        // Simple DP implementation (mock)
        let rows = costs.len();
        let cols = costs[0].len();

        let mut path = Vec::with_capacity(cols);
        for c in 0..cols {
            path.push(0); // Always pick first row
        }

        Ok(path)
    }

    pub fn compute_optimal_path_mock(&self, costs: &Vec<Vec<f32>>) -> Result<Vec<usize>> {
        self.compute_optimal_path(costs)
    }

    pub fn aggregate_vad_scores(&self, scores: &[f32]) -> Result<VADAggregation> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        let min = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = scores.iter().sum();
        let mean = sum / scores.len() as f32;

        Ok(VADAggregation { min, max, mean })
    }

    pub fn aggregate_vad_scores_mock(&self, scores: &[f32]) -> Result<VADAggregation> {
        self.aggregate_vad_scores(scores)
    }

    pub fn create_sequence(&self) -> DPXSequence {
        DPXSequence::new()
    }

    pub fn create_batch(&self) -> DPXBatch {
        DPXBatch::new()
    }

    pub fn execute_dpx(&self, instruction: DPXInstruction) -> Result<i32> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        Ok(instruction.execute())
    }

    pub fn compare_arrays_mock(&self, a: &[f32], b: &[f32]) -> Result<Vec<i32>> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        let mut result = Vec::with_capacity(a.len().min(b.len()));

        for i in 0..a.len().min(b.len()) {
            result.push(a[i].partial_cmp(&b[i]).unwrap_or(std::cmp::Ordering::Equal).as_i32());
        }

        Ok(result)
    }

    pub fn compose_constraints_mock(
        &self,
        _rate_weight: f32,
        _context_weight: f32,
        _sentiment_weight: f32,
        rate_constraints: &[f32],
        context_constraints: &[f32],
        sentiment_constraints: &[f32],
    ) -> Result<Vec<f32>> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        let mut results = Vec::with_capacity(rate_constraints.len());

        for i in 0..rate_constraints.len() {
            let eq = 0.3 * rate_constraints[i]
                + 0.4 * context_constraints[i]
                + 0.3 * sentiment_constraints[i];
            results.push(eq);
        }

        Ok(results)
    }

    pub fn rolling_window_min_max_mock(&self, data: &[f32], window_size: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        let mut mins = Vec::new();
        let mut maxs = Vec::new();

        for i in 0..=(data.len() - window_size) {
            let window = &data[i..i + window_size];
            let min = window.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = window.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            mins.push(min);
            maxs.push(max);
        }

        Ok((mins, maxs))
    }

    pub fn smith_waterman_mock(&self, seq1: &[f32], seq2: &[f32]) -> Result<f32> {
        if !self.initialized {
            return Err(GPUError::DPXNotAvailable);
        }

        // Simplified Smith-Waterman (just return similarity)
        let mut score = 0.0f32;

        for (a, b) in seq1.iter().zip(seq2.iter()) {
            score += (a - b).abs();
        }

        Ok(score)
    }

    pub fn get_device_name(&self) -> Result<String> {
        Ok("NVIDIA H100".to_string())
    }
}

impl Default for DPXContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpx_instruction() {
        let min_inst = DPXInstruction::Min(10, 20);
        assert_eq!(min_inst.execute(), 10);

        let max_inst = DPXInstruction::Max(10, 20);
        assert_eq!(max_inst.execute(), 20);

        let cmp_inst = DPXInstruction::Compare(15, 20);
        assert!(cmp_inst.execute() < 0);
    }

    #[test]
    fn test_dpx_context() {
        let context = DPXContext::new();
        assert!(!context.is_initialized());
        assert_eq!(context.device_count(), 0);
    }

    #[test]
    fn test_dpx_min_max() {
        let context = DPXContext::new();
        context.initialize_mock();

        let data = vec![1.0f32, 5.0, 2.0, 8.0, 3.0];
        let (min, max) = context.min_max(&data).unwrap();

        assert_eq!(min, 1.0);
        assert_eq!(max, 8.0);
    }

    #[test]
    fn test_dpx_min_max_int() {
        let context = DPXContext::new();
        context.initialize_mock();

        let data = vec![1i32, 5, 2, 8, 3];
        let (min, max) = context.min_max_int(&data).unwrap();

        assert_eq!(min, 1);
        assert_eq!(max, 8);
    }

    #[test]
    fn test_vad_aggregation() {
        let context = DPXContext::new();
        context.initialize_mock();

        let scores = vec![0.8f32, 0.9, 0.7, 0.95, 0.85];
        let result = context.aggregate_vad_scores(&scores).unwrap();

        assert_eq!(result.min, 0.7);
        assert_eq!(result.max, 0.95);
        assert!((result.mean - 0.84).abs() < 0.01);
    }

    #[test]
    fn test_dpx_batch() {
        let context = DPXContext::new();
        let batch = context.create_batch();

        batch.add(DPXInstruction::Min(1, 2));
        batch.add(DPXInstruction::Max(3, 4));

        assert_eq!(batch.len(), 2);

        let results = batch.execute_all().unwrap();
        assert_eq!(results[0], 1);
        assert_eq!(results[1], 4);
    }
}
