use ndarray::{Array2, ArrayView1, Axis};
use crate::network::NeuralNetwork;

/// Compare the network's analytical weight gradients against numerical ones.
///
/// Uses the squared error loss `0.5 * sum((output - target)^2)`, the same loss
/// `train_minibatch` backpropagates, and returns one relative error per weight.
/// In f32 the finite difference is itself noisy, so treat values below roughly
/// 1e-2 as agreement; a broken backward pass shows up as errors near 1.
///
/// Keep `epsilon` at 1e-2 or above. The loss is an f32, so the difference of two
/// losses is quantized at about `loss * 1e-7`, and dividing that by `2 * epsilon`
/// sets a noise floor on the numerical gradient: at 1e-3 the floor is around 6e-5,
/// which swamps any weight whose true gradient is smaller than that.
///
/// This is O(number of weights) forward passes, so keep it to small networks.
pub fn gradient_check(
    network: &mut NeuralNetwork,
    input: ArrayView1<f32>,
    target: ArrayView1<f32>,
    epsilon: f32,
) -> Vec<f32> {
    let mut relative_errors = Vec::new();

    // Analytical gradients: dL/dy is (output - target) for this loss
    let output = network.forward(input);
    let output_errors = (&output - &target).insert_axis(Axis(0));
    let analytical = network.backward_batch(output_errors.view());

    for layer_idx in 0..network.layers.len() {
        let weights_shape = network.layers[layer_idx].weights.shape().to_owned();
        let original_weights = network.layers[layer_idx].weights.clone();
        let analytical_weights = &analytical[layer_idx].0;

        for i in 0..weights_shape[0] {
            for j in 0..weights_shape[1] {
                network.layers[layer_idx].weights[[i, j]] = original_weights[[i, j]] + epsilon;
                let output_plus = network.forward(input);
                let loss_plus = (&output_plus - &target).mapv(|x| x * x).sum() / 2.0;

                network.layers[layer_idx].weights[[i, j]] = original_weights[[i, j]] - epsilon;
                let output_minus = network.forward(input);
                let loss_minus = (&output_minus - &target).mapv(|x| x * x).sum() / 2.0;

                network.layers[layer_idx].weights[[i, j]] = original_weights[[i, j]];

                let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytic = analytical_weights[[i, j]];

                // Relative error, with a floor so that two near-zero gradients do not
                // turn f32 rounding noise into a large ratio
                let denominator = (analytic.abs() + numerical.abs()).max(1e-3);
                relative_errors.push((analytic - numerical).abs() / denominator);
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::optimizer::{OptimizerWrapper, SGD};
    use ndarray::array;

    #[test]
    fn backprop_matches_numerical_gradients() {
        let mut network = NeuralNetwork::new(
            &[3, 4, 2],
            &[Activation::Tanh, Activation::Linear],
            OptimizerWrapper::SGD(SGD::new()),
        );

        let input = array![0.5, -0.2, 0.8];
        let target = array![1.0, -0.5];

        let errors = gradient_check(&mut network, input.view(), target.view(), 1e-2);

        assert!(!errors.is_empty());
        let worst = errors.iter().copied().fold(0.0f32, f32::max);
        assert!(worst < 5e-2, "largest relative error was {worst}");
    }
}
