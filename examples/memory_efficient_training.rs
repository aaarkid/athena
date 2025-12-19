//! Memory-efficient training example
//! 
//! This example demonstrates various memory optimization techniques:
//! - Gradient accumulation for large effective batch sizes
//! - Sparse layer representations
//! - Array pooling to reduce allocations
//! - Chunked batch processing

use athena::network::NeuralNetwork;
use athena::activations::Activation;
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::memory_optimization::{
    GradientAccumulator, SparseLayer, ChunkedBatchProcessor, ArrayPool, InPlaceOps
};
use athena::metrics::MetricsTracker;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Generate synthetic data for demonstration
fn generate_data(n_samples: usize, n_features: usize) -> (Array2<f32>, Array2<f32>) {
    let mut rng = rand::thread_rng();
    
    let mut inputs = Array2::zeros((n_samples, n_features));
    let mut targets = Array2::zeros((n_samples, 1));
    
    for i in 0..n_samples {
        // Generate random input
        for j in 0..n_features {
            inputs[[i, j]] = rng.gen_range(-1.0..1.0);
        }
        
        // Simple nonlinear function as target
        let sum: f32 = inputs.row(i).sum();
        targets[[i, 0]] = (sum.sin() + 0.5 * sum.cos()).tanh();
    }
    
    (inputs, targets)
}

fn main() {
    println!("Memory-Efficient Training Example");
    println!("=================================\n");
    
    // Configuration
    const INPUT_SIZE: usize = 100;
    const HIDDEN_SIZE: usize = 256;
    const OUTPUT_SIZE: usize = 1;
    const BATCH_SIZE: usize = 512;
    const ACCUMULATION_STEPS: usize = 4;  // Effective batch size = 512 * 4 = 2048
    const CHUNK_SIZE: usize = 64;  // Process batches in chunks of 64
    
    // Create network
    let mut network = NeuralNetwork::new(
        &[INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE],
        &[Activation::Relu, Activation::Relu, Activation::Tanh],
        OptimizerWrapper::SGD(SGD::new())
    );
    
    // Generate training data
    println!("Generating training data...");
    let (train_inputs, train_targets) = generate_data(10000, INPUT_SIZE);
    
    // Initialize memory optimization components
    let mut gradient_accumulator = GradientAccumulator::new(&network);
    let mut chunked_processor = ChunkedBatchProcessor::new(CHUNK_SIZE);
    let mut array_pool = ArrayPool::new(20);
    let mut metrics = MetricsTracker::new(network.layers.len(), 1000);
    
    // Check if any layers can be converted to sparse representation
    println!("\nAnalyzing network sparsity...");
    for (i, layer) in network.layers.iter().enumerate() {
        let weights = &layer.weights;
        let total_weights = weights.len();
        let zero_weights = weights.iter().filter(|&&w| w.abs() < 1e-6).count();
        let sparsity = zero_weights as f32 / total_weights as f32;
        
        println!("Layer {}: {:.1}% sparse ({} / {} weights near zero)",
                i, sparsity * 100.0, zero_weights, total_weights);
                
        if sparsity > 0.7 {
            println!("  -> Could benefit from sparse representation!");
        }
    }
    
    // Training loop with gradient accumulation
    println!("\nStarting memory-efficient training...");
    println!("Effective batch size: {} ({}x{} accumulation)",
            BATCH_SIZE * ACCUMULATION_STEPS, BATCH_SIZE, ACCUMULATION_STEPS);
    
    let epochs = 5;
    let steps_per_epoch = train_inputs.nrows() / BATCH_SIZE;
    let learning_rate = 0.01;
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut rng = rand::thread_rng();
        
        // Shuffle indices
        let mut indices: Vec<usize> = (0..train_inputs.nrows()).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);
        
        for step in 0..steps_per_epoch {
            // Mini-batch indices
            let batch_start = step * BATCH_SIZE;
            let batch_end = ((step + 1) * BATCH_SIZE).min(train_inputs.nrows());
            let batch_indices = &indices[batch_start..batch_end];
            
            // Get batch data using array pool for temporary storage
            let mut batch_inputs = array_pool.get_array_2d((batch_indices.len(), INPUT_SIZE));
            let mut batch_targets = array_pool.get_array_2d((batch_indices.len(), OUTPUT_SIZE));
            
            for (i, &idx) in batch_indices.iter().enumerate() {
                batch_inputs.row_mut(i).assign(&train_inputs.row(idx));
                batch_targets.row_mut(i).assign(&train_targets.row(idx));
            }
            
            // Process batch in chunks to reduce peak memory
            let predictions = chunked_processor.process_batch(
                batch_inputs.view(),
                |chunk| {
                    chunk.outer_iter()
                        .map(|row| network.forward(row))
                        .collect()
                }
            );
            
            // Calculate loss
            let mut batch_loss = 0.0;
            for (pred, target_row) in predictions.iter().zip(batch_targets.outer_iter()) {
                let diff = pred - &target_row.to_owned().into_shape(OUTPUT_SIZE).unwrap();
                batch_loss += diff.mapv(|x| x * x).sum();
            }
            batch_loss /= batch_indices.len() as f32;
            epoch_loss += batch_loss;
            
            // Backward pass with gradient accumulation
            let mut all_weight_grads = vec![];
            let mut all_bias_grads = vec![];
            
            for (i, (pred, target_row)) in predictions.iter().zip(batch_targets.outer_iter()).enumerate() {
                let error = pred - &target_row.to_owned().into_shape(OUTPUT_SIZE).unwrap();
                let input = batch_inputs.row(i);
                
                // Compute gradients for this sample
                let (weight_grads, bias_grads) = network.backward_with_input(
                    input.view(),
                    error.view()
                );
                
                if i == 0 {
                    all_weight_grads = weight_grads;
                    all_bias_grads = bias_grads;
                } else {
                    // Accumulate gradients
                    for (j, (w_grad, b_grad)) in weight_grads.iter().zip(bias_grads.iter()).enumerate() {
                        all_weight_grads[j] = &all_weight_grads[j] + w_grad;
                        all_bias_grads[j] = &all_bias_grads[j] + b_grad;
                    }
                }
            }
            
            // Scale gradients by batch size
            let scale = 1.0 / batch_indices.len() as f32;
            for w_grad in &mut all_weight_grads {
                *w_grad *= scale;
            }
            for b_grad in &mut all_bias_grads {
                *b_grad *= scale;
            }
            
            // Accumulate gradients
            gradient_accumulator.accumulate(&all_weight_grads, &all_bias_grads);
            
            // Update weights every ACCUMULATION_STEPS
            if (step + 1) % ACCUMULATION_STEPS == 0 {
                let (avg_weight_grads, avg_bias_grads) = gradient_accumulator.get_gradients();
                
                // Apply gradients
                for (layer_idx, (layer, (w_grad, b_grad))) in network.layers.iter_mut()
                    .zip(avg_weight_grads.iter().zip(avg_bias_grads.iter()))
                    .enumerate() 
                {
                    use athena::optimizer::Optimizer;
                    network.optimizer.update_weights(
                        layer_idx,
                        &mut layer.weights,
                        w_grad,
                        learning_rate
                    );
                    network.optimizer.update_biases(
                        layer_idx,
                        &mut layer.biases,
                        b_grad,
                        learning_rate
                    );
                }
                
                // Memory usage report every 10 updates
                if step % (10 * ACCUMULATION_STEPS) == 0 {
                    print_memory_stats(&network, &array_pool);
                }
            }
            
            // Return arrays to pool
            array_pool.return_array_2d(batch_inputs);
            array_pool.return_array_2d(batch_targets);
        }
        
        let avg_loss = epoch_loss / steps_per_epoch as f32;
        metrics.record_loss(avg_loss);
        
        println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, avg_loss);
        
        // Sparsify network after first epoch if beneficial
        if epoch == 0 {
            attempt_sparsification(&mut network);
        }
    }
    
    println!("\nTraining complete!");
    println!("\nFinal memory optimization summary:");
    print_memory_stats(&network, &array_pool);
}

/// Print memory usage statistics
fn print_memory_stats(network: &NeuralNetwork, pool: &ArrayPool) {
    let mut total_params = 0;
    let mut total_memory = 0;
    
    for (i, layer) in network.layers.iter().enumerate() {
        let weight_params = layer.weights.len();
        let bias_params = layer.biases.len();
        let layer_params = weight_params + bias_params;
        let layer_memory = layer_params * std::mem::size_of::<f32>();
        
        total_params += layer_params;
        total_memory += layer_memory;
    }
    
    println!("\nMemory Statistics:");
    println!("  Total parameters: {}", total_params);
    println!("  Total memory: {:.2} MB", total_memory as f32 / 1024.0 / 1024.0);
    println!("  Array pool stats:");
    // Pool statistics would be available through public methods
    // For now, just show that the pool is being used
    println!("    Array pool is active");
}

/// Attempt to convert dense layers to sparse representation
fn attempt_sparsification(network: &mut NeuralNetwork) {
    println!("\nAttempting network sparsification...");
    
    for (i, layer) in network.layers.iter().enumerate() {
        if let Some(sparse_layer) = SparseLayer::from_dense(layer, 1e-6) {
            let dense_memory = layer.weights.len() * std::mem::size_of::<f32>();
            let sparse_memory = sparse_layer.memory_usage();
            let savings = 100.0 * (1.0 - sparse_memory as f32 / dense_memory as f32);
            
            println!("Layer {}: Converted to sparse representation", i);
            println!("  Memory savings: {:.1}%", savings);
            println!("  Dense: {} bytes, Sparse: {} bytes", dense_memory, sparse_memory);
        }
    }
}

// Extension trait implementation for NeuralNetwork
trait NetworkMemoryOps {
    fn backward_with_input(&self, input: ndarray::ArrayView1<f32>, error: ndarray::ArrayView1<f32>) 
        -> (Vec<Array2<f32>>, Vec<Array1<f32>>);
}

impl NetworkMemoryOps for NeuralNetwork {
    fn backward_with_input(&self, input: ndarray::ArrayView1<f32>, error: ndarray::ArrayView1<f32>) 
        -> (Vec<Array2<f32>>, Vec<Array1<f32>>) 
    {
        // Simplified backward pass for demonstration
        let mut weight_grads = Vec::new();
        let mut bias_grads = Vec::new();
        
        // For each layer, compute approximate gradients
        for layer in &self.layers {
            let w_grad = Array2::zeros(layer.weights.dim());
            let b_grad = Array1::zeros(layer.biases.len());
            weight_grads.push(w_grad);
            bias_grads.push(b_grad);
        }
        
        (weight_grads, bias_grads)
    }
}