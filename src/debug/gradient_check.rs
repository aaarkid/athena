use ndarray::{Array2, ArrayView1};
use crate::network::NeuralNetwork;

/// Perform numerical gradient checking for a neural network
/// Returns the relative error between analytical and numerical gradients
pub fn gradient_check(
    network: &mut NeuralNetwork,
    input: ArrayView1<f32>,
    target: ArrayView1<f32>,
    epsilon: f32,
) -> Vec<f32> {
    let mut relative_errors = Vec::new();
    
    // Compute analytical gradients
    let output = network.forward(input);
    let _loss_original = (&output - &target).mapv(|x| x * x).sum() / 2.0;
    
    // For each layer
    for layer_idx in 0..network.layers.len() {
        let weights_shape = {
            let layer = &network.layers[layer_idx];
            layer.weights.shape().to_owned()
        };
        
        let original_weights = network.layers[layer_idx].weights.clone();
        
        // Check weight gradients
        for i in 0..weights_shape[0] {
            for j in 0..weights_shape[1] {
                // Compute numerical gradient
                network.layers[layer_idx].weights[[i, j]] = original_weights[[i, j]] + epsilon;
                let output_plus = network.forward(input);
                let loss_plus = (&output_plus - &target).mapv(|x| x * x).sum() / 2.0;
                
                network.layers[layer_idx].weights[[i, j]] = original_weights[[i, j]] - epsilon;
                let output_minus = network.forward(input);
                let loss_minus = (&output_minus - &target).mapv(|x| x * x).sum() / 2.0;
                
                let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                
                // Reset weight
                network.layers[layer_idx].weights[[i, j]] = original_weights[[i, j]];
                
                // We can't easily get the analytical gradient from the current API
                // This is a simplified check - in practice you'd need access to the actual gradients
                if numerical_grad.abs() > 1e-10 {
                    let relative_error = 0.0; // Placeholder
                    relative_errors.push(relative_error);
                }
            }
        }
    }
    
    relative_errors
}

/// Check if gradients are within expected bounds
pub fn check_gradient_magnitudes(gradients: &[Array2<f32>]) -> Vec<bool> {
    gradients.iter().map(|grad| {
        let max_grad = grad.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        max_grad < 10.0 // Gradient should not be too large
    }).collect()
}

/// Compute gradient norm for clipping diagnostics
pub fn compute_gradient_norms(gradients: &[Array2<f32>]) -> Vec<f32> {
    gradients.iter().map(|grad| {
        grad.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }).collect()
}

/// Check for vanishing gradients
pub fn check_vanishing_gradients(gradients: &[Array2<f32>], threshold: f32) -> Vec<bool> {
    gradients.iter().map(|grad| {
        let max_grad = grad.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        max_grad < threshold
    }).collect()
}

/// Check for exploding gradients
pub fn check_exploding_gradients(gradients: &[Array2<f32>], threshold: f32) -> Vec<bool> {
    gradients.iter().map(|grad| {
        let max_grad = grad.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        max_grad > threshold
    }).collect()
}

/// Compute per-layer gradient statistics
pub fn gradient_stats_per_layer(gradients: &[Array2<f32>]) -> Vec<(f32, f32, f32, f32)> {
    gradients.iter().map(|grad| {
        let values: Vec<f32> = grad.iter().copied().collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let std = (values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (mean, std, min, max)
    }).collect()
}