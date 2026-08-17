//! Memory optimization utilities for Athena
//! 
//! This module provides various memory optimization techniques to reduce
//! memory footprint and improve cache efficiency in neural networks.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use std::mem;
use crate::network::NeuralNetwork;
use crate::layers::Layer;

/// Memory pool for reusing arrays to reduce allocations
pub struct ArrayPool {
    /// Pools for different array sizes
    pool_1d: Vec<(usize, Array1<f32>)>,
    pool_2d: Vec<((usize, usize), Array2<f32>)>,
    /// Maximum number of arrays to keep in each pool
    max_pool_size: usize,
}

impl ArrayPool {
    /// Create a new array pool
    pub fn new(max_pool_size: usize) -> Self {
        ArrayPool {
            pool_1d: Vec::with_capacity(max_pool_size),
            pool_2d: Vec::with_capacity(max_pool_size),
            max_pool_size,
        }
    }
    
    /// Arrays currently held, as (1D count, 2D count), and the per-pool cap
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.pool_1d.len(), self.pool_2d.len(), self.max_pool_size)
    }

    /// Bytes held by the pooled arrays
    pub fn pooled_bytes(&self) -> usize {
        let ones: usize = self.pool_1d.iter().map(|(size, _)| *size).sum();
        let twos: usize = self.pool_2d.iter().map(|((r, c), _)| r * c).sum();
        (ones + twos) * std::mem::size_of::<f32>()
    }

    /// Get or create a 1D array
    pub fn get_array_1d(&mut self, size: usize) -> Array1<f32> {
        // Try to find a matching array in the pool
        if let Some(pos) = self.pool_1d.iter().position(|(s, _)| *s == size) {
            let (_, array) = self.pool_1d.swap_remove(pos);
            array
        } else {
            Array1::zeros(size)
        }
    }
    
    /// Return a 1D array to the pool
    pub fn return_array_1d(&mut self, mut array: Array1<f32>) {
        if self.pool_1d.len() < self.max_pool_size {
            let size = array.len();
            array.fill(0.0);  // Clear the array
            self.pool_1d.push((size, array));
        }
    }
    
    /// Get or create a 2D array
    pub fn get_array_2d(&mut self, shape: (usize, usize)) -> Array2<f32> {
        if let Some(pos) = self.pool_2d.iter().position(|(s, _)| *s == shape) {
            let (_, array) = self.pool_2d.swap_remove(pos);
            array
        } else {
            Array2::zeros(shape)
        }
    }
    
    /// Return a 2D array to the pool
    pub fn return_array_2d(&mut self, mut array: Array2<f32>) {
        if self.pool_2d.len() < self.max_pool_size {
            let shape = array.dim();
            array.fill(0.0);
            self.pool_2d.push((shape, array));
        }
    }
}

/// Gradient accumulator for memory-efficient mini-batch processing
pub struct GradientAccumulator {
    /// Accumulated gradients for each layer
    weight_grads: Vec<Array2<f32>>,
    bias_grads: Vec<Array1<f32>>,
    /// Number of samples accumulated
    num_samples: usize,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator for a network
    pub fn new(network: &NeuralNetwork) -> Self {
        let mut weight_grads = Vec::new();
        let mut bias_grads = Vec::new();
        
        for layer in &network.layers {
            weight_grads.push(Array2::zeros(layer.weights.dim()));
            bias_grads.push(Array1::zeros(layer.biases.len()));
        }
        
        GradientAccumulator {
            weight_grads,
            bias_grads,
            num_samples: 0,
        }
    }
    
    /// Accumulate gradients from a batch
    pub fn accumulate(&mut self, weight_gradients: &[Array2<f32>], bias_gradients: &[Array1<f32>]) {
        for (i, (w_grad, b_grad)) in weight_gradients.iter().zip(bias_gradients.iter()).enumerate() {
            self.weight_grads[i] += w_grad;
            self.bias_grads[i] += b_grad;
        }
        self.num_samples += 1;
    }
    
    /// Take the mean of everything accumulated so far, and clear.
    ///
    /// The divisor is the number of `accumulate` calls, not the number of rows those
    /// calls covered, so accumulating one batch of 32 and one of 8 weights them equally.
    /// Call it once per batch of equal size, or scale before accumulating.
    ///
    /// This both reads and resets: a second call with nothing in between returns zeros.
    pub fn get_gradients(&mut self) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        if self.num_samples == 0 {
            return (self.weight_grads.clone(), self.bias_grads.clone());
        }
        
        let scale = 1.0 / self.num_samples as f32;
        
        // Scale gradients by number of samples
        for w_grad in &mut self.weight_grads {
            *w_grad *= scale;
        }
        for b_grad in &mut self.bias_grads {
            *b_grad *= scale;
        }
        
        // Reset for next accumulation
        self.num_samples = 0;
        let weight_grads = self.weight_grads.clone();
        let bias_grads = self.bias_grads.clone();
        
        // Clear accumulators
        for w_grad in &mut self.weight_grads {
            w_grad.fill(0.0);
        }
        for b_grad in &mut self.bias_grads {
            b_grad.fill(0.0);
        }
        
        (weight_grads, bias_grads)
    }
}

/// Walks a large batch in fixed-size chunks, so peak memory follows the chunk rather
/// than the whole batch.
///
/// It used to hold an `ArrayPool` it never touched. The pool is a separate type; use it
/// directly if the closure needs scratch space.
pub struct ChunkedBatchProcessor {
    chunk_size: usize,
}

impl ChunkedBatchProcessor {
    /// Create a new chunked batch processor.
    ///
    /// A `chunk_size` of zero would loop forever, so it is raised to one.
    pub fn new(chunk_size: usize) -> Self {
        ChunkedBatchProcessor {
            chunk_size: chunk_size.max(1),
        }
    }

    /// Rows handed to the closure at a time.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
    
    /// Process a large batch in memory-efficient chunks
    pub fn process_batch<F>(
        &mut self,
        batch: ArrayView2<f32>,
        mut process_chunk: F,
    ) -> Vec<Array1<f32>>
    where
        F: FnMut(ArrayView2<f32>) -> Vec<Array1<f32>>,
    {
        let batch_size = batch.nrows();
        let mut results = Vec::with_capacity(batch_size);
        
        // Process in chunks to reduce peak memory usage
        for chunk_start in (0..batch_size).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(batch_size);
            let chunk = batch.slice(s![chunk_start..chunk_end, ..]);
            
            let chunk_results = process_chunk(chunk);
            results.extend(chunk_results);
        }
        
        results
    }
}

/// Sparse layer representation for layers with many zero weights
pub struct SparseLayer {
    /// Non-zero weight values
    values: Vec<f32>,
    /// Row indices of non-zero weights
    row_indices: Vec<usize>,
    /// Column pointers for CSC format
    col_pointers: Vec<usize>,
    /// Shape of the weight matrix
    #[allow(dead_code)]
    shape: (usize, usize),
    /// Bias values
    biases: Array1<f32>,
    /// Sparsity threshold
    #[allow(dead_code)]
    threshold: f32,
}

impl SparseLayer {
    /// Convert a dense layer to sparse representation if beneficial
    pub fn from_dense(layer: &Layer, threshold: f32) -> Option<Self> {
        let weights = &layer.weights;
        let (rows, cols) = weights.dim();
        
        // Count non-zero elements
        let nnz = weights.iter().filter(|&&x| x.abs() > threshold).count();
        
        // Only convert if sparsity is high enough (>70% zeros)
        let sparsity = 1.0 - (nnz as f32 / (rows * cols) as f32);
        if sparsity < 0.7 {
            return None;
        }
        
        // Convert to CSC format
        let mut values = Vec::with_capacity(nnz);
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_pointers = vec![0];
        
        for col in 0..cols {
            for row in 0..rows {
                let val = weights[[row, col]];
                if val.abs() > threshold {
                    values.push(val);
                    row_indices.push(row);
                }
            }
            col_pointers.push(values.len());
        }
        
        Some(SparseLayer {
            values,
            row_indices,
            col_pointers,
            shape: (rows, cols),
            biases: layer.biases.clone(),
            threshold,
        })
    }
    
    /// Perform sparse matrix-vector multiplication
    pub fn forward(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let mut output = self.biases.clone();
        
        // Sparse matrix-vector multiplication
        for (col, &input_val) in input.iter().enumerate() {
            if col < self.col_pointers.len() - 1 {
                let start = self.col_pointers[col];
                let end = self.col_pointers[col + 1];
                
                for i in start..end {
                    let row = self.row_indices[i];
                    let weight = self.values[i];
                    output[row] += weight * input_val;
                }
            }
        }
        
        output
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        mem::size_of_val(&self.values) +
        mem::size_of_val(&self.row_indices) +
        mem::size_of_val(&self.col_pointers) +
        mem::size_of_val(&self.biases)
    }
}

/// Weight sharing for convolutional-like patterns in dense layers
pub struct WeightSharingLayer {
    /// Shared weight patterns
    patterns: Vec<Array1<f32>>,
    /// Pattern assignments for each weight
    pattern_indices: Array2<usize>,
    /// Scale factors for each weight
    scales: Array2<f32>,
    /// Biases
    biases: Array1<f32>,
}

impl WeightSharingLayer {
    /// Create weight sharing layer from dense layer using k-means clustering
    pub fn from_dense(layer: &Layer, num_patterns: usize) -> Self {
        let weights = &layer.weights;
        let (rows, cols) = weights.dim();
        
        // Simple k-means clustering on weight vectors
        let mut patterns = Vec::new();
        let mut pattern_indices = Array2::zeros((rows, cols));
        let mut scales = Array2::ones((rows, cols));
        
        // Initialize patterns with random weight vectors
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..num_patterns {
            let idx = rng.gen_range(0..rows);
            patterns.push(weights.row(idx).to_owned());
        }
        
        // Assign each weight vector to nearest pattern
        for row in 0..rows {
            let weight_vec = weights.row(row);
            
            // Find closest pattern
            let (best_idx, best_scale) = patterns.iter()
                .enumerate()
                .map(|(idx, pattern)| {
                    // Compute optimal scale factor
                    let dot_product: f32 = weight_vec.iter()
                        .zip(pattern.iter())
                        .map(|(w, p)| w * p)
                        .sum();
                    let pattern_norm: f32 = pattern.iter()
                        .map(|p| p * p)
                        .sum();
                    
                    let scale = if pattern_norm > 0.0 {
                        dot_product / pattern_norm
                    } else {
                        0.0
                    };
                    
                    // Compute distance with optimal scale
                    let diff: f32 = weight_vec.iter()
                        .zip(pattern.iter())
                        .map(|(w, p)| {
                            let diff = w - scale * p;
                            diff * diff
                        })
                        .sum();
                    
                    (idx, scale, diff)
                })
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                .map(|(idx, scale, _)| (idx, scale))
                .unwrap();
            
            for col in 0..cols {
                pattern_indices[[row, col]] = best_idx;
                scales[[row, col]] = best_scale;
            }
        }
        
        WeightSharingLayer {
            patterns,
            pattern_indices,
            scales,
            biases: layer.biases.clone(),
        }
    }
    
    /// Forward pass with weight sharing
    pub fn forward(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let mut output = self.biases.clone();
        let (rows, cols) = self.pattern_indices.dim();
        
        for row in 0..rows {
            for col in 0..cols {
                let pattern_idx = self.pattern_indices[[row, col]];
                let scale = self.scales[[row, col]];
                let weight = scale * self.patterns[pattern_idx][col];
                output[row] += weight * input[col];
            }
        }
        
        output
    }
}

/// In-place operations to reduce memory allocations
pub trait InPlaceOps {
    /// Apply activation function in-place
    fn apply_activation_inplace(&mut self, activation: &crate::activations::Activation);
    
    /// Add another array in-place
    fn add_inplace(&mut self, other: ArrayView1<f32>);
    
    /// Scale array in-place
    fn scale_inplace(&mut self, scale: f32);
}

impl InPlaceOps for Array1<f32> {
    fn apply_activation_inplace(&mut self, activation: &crate::activations::Activation) {
        activation.apply(self);
    }
    
    fn add_inplace(&mut self, other: ArrayView1<f32>) {
        *self += &other;
    }
    
    fn scale_inplace(&mut self, scale: f32) {
        *self *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    
    #[test]
    fn test_array_pool() {
        let mut pool = ArrayPool::new(5);
        
        // Get an array
        let array1 = pool.get_array_1d(10);
        assert_eq!(array1.len(), 10);
        
        // Return it to the pool
        pool.return_array_1d(array1);
        
        // Get it again - should reuse the same memory
        let array2 = pool.get_array_1d(10);
        assert_eq!(array2.len(), 10);
    }
    
    #[test]
    fn test_gradient_accumulator() {
        let network = NeuralNetwork::new(
            &[4, 3, 2],
            &[Activation::Relu, Activation::Linear],
            crate::optimizer::OptimizerWrapper::SGD(crate::optimizer::SGD::new())
        );
        
        let mut accumulator = GradientAccumulator::new(&network);
        
        // Get correct dimensions from the network
        let layer0_shape = network.layers[0].weights.dim();
        let layer1_shape = network.layers[1].weights.dim();
        
        // Simulate gradient accumulation with correct dimensions
        let weight_grads = vec![
            Array2::ones(layer0_shape),
            Array2::ones(layer1_shape),
        ];
        let bias_grads = vec![
            Array1::ones(network.layers[0].biases.len()),
            Array1::ones(network.layers[1].biases.len()),
        ];
        
        accumulator.accumulate(&weight_grads, &bias_grads);
        accumulator.accumulate(&weight_grads, &bias_grads);
        
        let (avg_weight_grads, avg_bias_grads) = accumulator.get_gradients();
        
        // Should be averaged (divided by 2)
        assert_eq!(avg_weight_grads[0][[0, 0]], 1.0);
        assert_eq!(avg_bias_grads[0][0], 1.0);
    }
    
    #[test]
    fn test_sparse_layer() {
        // Create a layer with many zeros (5 outputs, 10 inputs)
        let mut layer = Layer::new(10, 5, Activation::Relu);
        layer.weights.fill(0.0);
        // Set a few non-zero weights
        if layer.weights.dim().0 > 0 && layer.weights.dim().1 > 0 {
            layer.weights[[0, 0]] = 1.0;
        }
        if layer.weights.dim().0 > 1 && layer.weights.dim().1 > 1 {
            layer.weights[[1, 1]] = 2.0;
        }
        if layer.weights.dim().0 > 2 && layer.weights.dim().1 > 2 {
            layer.weights[[2, 2]] = 3.0;
        }
        
        let sparse_opt = SparseLayer::from_dense(&layer, 0.1);
        
        // Only test if sparse conversion was successful
        if let Some(sparse) = sparse_opt {
            // Test forward pass
            let input = Array1::ones(10);
            let output = sparse.forward(input.view());
            assert_eq!(output.len(), 5);
        }
    }
    
    #[test]
    fn test_weight_sharing() {
        let layer = Layer::new(4, 4, Activation::Relu);
        let weight_sharing = WeightSharingLayer::from_dense(&layer, 2);
        
        let input = Array1::ones(4);
        let output = weight_sharing.forward(input.view());
        assert_eq!(output.len(), 4);
    }
}
#[cfg(test)]
mod pool_tests {
    use super::*;
    use crate::activations::Activation;
    use crate::optimizer::{OptimizerWrapper, SGD};

    #[test]
    fn a_returned_array_comes_back_out_of_the_pool() {
        let mut pool = ArrayPool::new(4);
        assert_eq!(pool.stats(), (0, 0, 4));

        let array = pool.get_array_1d(32);
        assert_eq!(array.len(), 32);
        // Nothing pooled yet: the first request has nothing to reuse
        assert_eq!(pool.stats().0, 0);

        pool.return_array_1d(array);
        assert_eq!(pool.stats().0, 1);

        // The next request of that size is served from the pool, and comes back zeroed
        let reused = pool.get_array_1d(32);
        assert!(reused.iter().all(|&v| v == 0.0), "a reused array kept its old values");
        assert_eq!(pool.stats().0, 0);

        // A different size is not served from it
        pool.return_array_1d(reused);
        let other = pool.get_array_2d((4, 8));
        assert_eq!(other.dim(), (4, 8));
        assert_eq!(pool.stats().0, 1, "the 1D pool was disturbed by a 2D request");
    }

    #[test]
    fn the_pool_stops_growing_at_its_cap() {
        let mut pool = ArrayPool::new(2);
        for _ in 0..10 {
            pool.return_array_1d(Array1::zeros(16));
        }
        let (held_1d, _, cap) = pool.stats();
        assert!(held_1d <= cap, "pool holds {} arrays against a cap of {}", held_1d, cap);
        assert!(pool.pooled_bytes() > 0);
    }

    #[test]
    fn chunking_a_batch_gives_the_same_rows_as_processing_it_whole() {
        let net = NeuralNetwork::new(
            &[4, 8, 3],
            &[Activation::Relu, Activation::Linear],
            OptimizerWrapper::SGD(SGD::new()),
        );

        let batch = Array2::from_shape_fn((17, 4), |(i, j)| ((i * 4 + j) as f32 * 0.21).sin());

        let whole = net.predict_batch(batch.view());

        // 17 rows in chunks of 5 leaves a short final chunk, which is where an
        // off-by-one shows up
        let mut processor = ChunkedBatchProcessor::new(5);
        let chunked = processor.process_batch(batch.view(), |chunk| {
            net.predict_batch(chunk)
                .outer_iter()
                .map(|row| row.to_owned())
                .collect()
        });

        assert_eq!(chunked.len(), 17);
        for (i, row) in chunked.iter().enumerate() {
            for (a, b) in row.iter().zip(whole.row(i).iter()) {
                assert!((a - b).abs() < 1e-6, "row {}: {} vs {}", i, a, b);
            }
        }
    }

    #[test]
    fn a_zero_chunk_size_does_not_loop_forever() {
        let mut processor = ChunkedBatchProcessor::new(0);
        assert_eq!(processor.chunk_size(), 1);

        let batch = Array2::<f32>::zeros((3, 2));
        let rows = processor.process_batch(batch.view(), |chunk| {
            chunk.outer_iter().map(|row| row.to_owned()).collect()
        });
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn the_gradient_accumulator_averages_and_then_clears() {
        let net = NeuralNetwork::new(
            &[3, 4, 2],
            &[Activation::Relu, Activation::Linear],
            OptimizerWrapper::SGD(SGD::new()),
        );

        let mut accumulator = GradientAccumulator::new(&net);

        let weight_grads: Vec<Array2<f32>> = net
            .layers
            .iter()
            .map(|layer| Array2::from_elem(layer.weights.dim(), 1.0))
            .collect();
        let bias_grads: Vec<Array1<f32>> = net
            .layers
            .iter()
            .map(|layer| Array1::from_elem(layer.biases.len(), 0.5))
            .collect();

        accumulator.accumulate(&weight_grads, &bias_grads);
        accumulator.accumulate(&weight_grads, &bias_grads);

        // get_gradients divides by the number of accumulate calls, so two identical
        // contributions average back to the original value rather than doubling
        let (weights, biases) = accumulator.get_gradients();
        assert!(
            weights[0].iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "expected the mean of two identical gradients, got {}",
            weights[0][[0, 0]]
        );
        assert!(biases[0].iter().all(|&v| (v - 0.5).abs() < 1e-6));

        // get_gradients hands the sums over and resets, so a second call is zero
        let (weights, biases) = accumulator.get_gradients();
        assert!(weights[0].iter().all(|&v| v == 0.0), "the accumulator did not clear");
        assert!(biases[0].iter().all(|&v| v == 0.0));
    }
}
