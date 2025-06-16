#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;
    use athena::network::NeuralNetwork;
    use athena::activations::Activation;
    use athena::layers::{Layer, WeightInit, LayerTrait, DenseLayer};
    use athena::optimizer::{OptimizerWrapper, SGD};
    use ndarray::{Array1, Array2};
    use std::f32::EPSILON;

    // Strategy for generating valid layer sizes
    fn layer_sizes_strategy() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(1usize..=100, 2..=5)
    }

    // Strategy for generating valid input arrays
    fn input_array_strategy(size: usize) -> impl Strategy<Value = Array1<f32>> {
        prop::collection::vec(any::<f32>().prop_filter("not NaN or Inf", |f| f.is_finite()), size)
            .prop_map(|v| Array1::from_vec(v))
    }

    // Strategy for generating valid weight matrices
    fn weight_matrix_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Array2<f32>> {
        prop::collection::vec(
            prop::collection::vec((-10.0f32..10.0).prop_filter("not NaN", |f| f.is_finite()), cols),
            rows
        ).prop_map(move |v| {
            let flat: Vec<f32> = v.into_iter().flatten().collect();
            Array2::from_shape_vec((rows, cols), flat).unwrap()
        })
    }

    proptest! {
        #[test]
        fn test_forward_propagation_output_shape(layer_sizes in layer_sizes_strategy()) {
            // Ensure we have at least 2 layer sizes
            if layer_sizes.len() < 2 {
                return Ok(());
            }
            
            let activations = vec![Activation::Relu; layer_sizes.len() - 1];
            let optimizer = OptimizerWrapper::SGD(SGD::new());
            let mut network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
            
            // Create input with appropriate size
            let input = Array1::zeros(layer_sizes[0]);
            let output = network.forward(input.view());
            
            // Output shape should match last layer size
            prop_assert_eq!(output.len(), layer_sizes[layer_sizes.len() - 1]);
        }

        #[test]
        fn test_forward_propagation_finite_outputs(
            input in input_array_strategy(10)
        ) {
            let layer_sizes = vec![10, 5, 3];
            let activations = vec![Activation::Relu, Activation::Sigmoid];
            let optimizer = OptimizerWrapper::SGD(SGD::new());
            let mut network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
            
            let output = network.forward(input.view());
            
            // All outputs should be finite
            for &val in output.iter() {
                prop_assert!(val.is_finite(), "Output contains non-finite values");
            }
        }

        #[test]
        fn test_activation_bounded_outputs(
            input in prop::collection::vec(-100.0f32..100.0, 10..100)
        ) {
            let input_array = Array1::from_vec(input);
            
            // Test Sigmoid bounds [0, 1]
            let mut sigmoid_layer = DenseLayer::new(input_array.len(), input_array.len(), Activation::Sigmoid);
            let sigmoid_output = sigmoid_layer.forward(input_array.view());
            for &val in sigmoid_output.iter() {
                prop_assert!(val >= 0.0 && val <= 1.0, "Sigmoid output out of bounds: {}", val);
            }
            
            // Test Tanh bounds [-1, 1]
            let mut tanh_layer = DenseLayer::new(input_array.len(), input_array.len(), Activation::Tanh);
            let tanh_output = tanh_layer.forward(input_array.view());
            for &val in tanh_output.iter() {
                prop_assert!(val >= -1.0 && val <= 1.0, "Tanh output out of bounds: {}", val);
            }
        }

        #[test]
        fn test_weight_initialization_ranges(
            rows in 1usize..=50,
            cols in 1usize..=50
        ) {
            // Test Xavier initialization
            let xavier_layer = Layer::new_with_init(rows, cols, Activation::Relu, WeightInit::XavierUniform);
            let xavier_bound = (6.0 / (rows + cols) as f32).sqrt();
            for &weight in xavier_layer.weights.iter() {
                prop_assert!(weight.abs() <= xavier_bound * 1.1, "Xavier weight out of expected range");
            }
            
            // Test He initialization
            let he_layer = Layer::new_with_init(rows, cols, Activation::Relu, WeightInit::HeUniform);
            let he_bound = (6.0 / rows as f32).sqrt();
            for &weight in he_layer.weights.iter() {
                prop_assert!(weight.abs() <= he_bound * 1.1, "He weight out of expected range");
            }
        }

        #[test]
        fn test_gradient_descent_reduces_loss(
            initial_weights in weight_matrix_strategy(3, 2),
            learning_rate in 0.001f32..0.1
        ) {
            let mut layer = DenseLayer::new(3, 2, Activation::Linear);
            layer.weights = initial_weights;
            
            // Simple target and input
            let input = Array1::from_vec(vec![1.0, 0.5, -0.5]);
            let target = Array1::from_vec(vec![1.0, 0.0]);
            
            // Forward pass
            let output1 = layer.forward(input.view());
            let loss1 = (&target - &output1).mapv(|x| x * x).sum();
            
            // Backward pass with small gradient
            let grad = (&output1 - &target) * 2.0;
            let (weight_grad, bias_grad) = layer.backward(grad.view());
            
            // Apply gradients manually
            layer.weights = &layer.weights - &(weight_grad * learning_rate);
            layer.biases = &layer.biases - &(bias_grad * learning_rate);
            
            // Second forward pass
            let output2 = layer.forward(input.view());
            let loss2 = (&target - &output2).mapv(|x| x * x).sum();
            
            // Loss should not increase (allowing for numerical precision)
            prop_assert!(loss2 <= loss1 + EPSILON * 100.0, 
                "Loss increased: {} -> {}", loss1, loss2);
        }

        #[test]
        fn test_layer_dimension_consistency(
            input_dim in 1usize..=100,
            hidden_dim in 1usize..=100,
            output_dim in 1usize..=100
        ) {
            let layer1 = Layer::new(input_dim, hidden_dim, Activation::Relu);
            let layer2 = Layer::new(hidden_dim, output_dim, Activation::Sigmoid);
            
            // Weight dimensions should match layer specifications
            prop_assert_eq!(layer1.weights.shape(), &[input_dim, hidden_dim]);
            prop_assert_eq!(layer1.biases.shape(), &[hidden_dim]);
            prop_assert_eq!(layer2.weights.shape(), &[hidden_dim, output_dim]);
            prop_assert_eq!(layer2.biases.shape(), &[output_dim]);
        }

        #[test]
        fn test_network_invariants(
            layer_sizes in prop::collection::vec(2usize..=50, 2..=5)
        ) {
            if layer_sizes.len() < 2 {
                return Ok(());
            }
            
            let activations = vec![Activation::Relu; layer_sizes.len() - 1];
            let optimizer = OptimizerWrapper::SGD(SGD::new());
            let mut network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
            
            // Test multiple forward passes with same input produce same output
            let input = Array1::ones(layer_sizes[0]);
            let output1 = network.forward(input.view());
            let output2 = network.forward(input.view());
            
            // Outputs should be identical
            for (a, b) in output1.iter().zip(output2.iter()) {
                prop_assert!((a - b).abs() < EPSILON, "Forward pass not deterministic");
            }
            
            // Number of layers should match
            prop_assert_eq!(network.layers.len(), layer_sizes.len() - 1);
        }

        #[test]
        fn test_zero_input_behavior(layer_sizes in layer_sizes_strategy()) {
            if layer_sizes.len() < 2 {
                return Ok(());
            }
            
            let activations = vec![Activation::Relu; layer_sizes.len() - 1];
            let optimizer = OptimizerWrapper::SGD(SGD::new());
            let mut network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
            
            // Zero all biases
            for layer in &mut network.layers {
                layer.biases.fill(0.0);
            }
            
            // Zero input should produce zero output with ReLU (ignoring biases)
            let input = Array1::zeros(layer_sizes[0]);
            let output = network.forward(input.view());
            
            // All outputs should be zero (or very close due to numerical precision)
            for &val in output.iter() {
                prop_assert!(val.abs() < EPSILON * 100.0, "Non-zero output from zero input: {}", val);
            }
        }
    }
}