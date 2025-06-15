use ndarray::{arr1, arr2};
use crate::network::NeuralNetwork;
use crate::activations::Activation;
use crate::optimizer::{OptimizerWrapper, SGD};
use crate::loss::{Loss, MSE as MSELoss, CrossEntropyLoss, HuberLoss};

#[test]
fn test_neural_network_creation() {
    let layer_sizes = &[3, 4, 2];
    let activations = &[Activation::Relu, Activation::Relu];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let network = NeuralNetwork::new(layer_sizes, activations, optimizer);

    assert_eq!(network.layers.len(), 2);
    assert_eq!(network.layers[0].weights.shape(), [3, 4]);
    assert_eq!(network.layers[0].biases.shape(), [4]);
    assert_eq!(network.layers[1].weights.shape(), [4, 2]);
    assert_eq!(network.layers[1].biases.shape(), [2]);
}

#[test]
fn test_forward_pass() {
    let layer_sizes = &[3, 4, 2];
    let activations = &[Activation::Relu, Activation::Relu];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);

    let input = arr1(&[1.0, 2.0, 3.0]);
    let output = network.forward(input.view());

    assert_eq!(output.shape(), [2]);
}

#[test]
fn test_forward_batch() {
    let layer_sizes = &[3, 4, 2];
    let activations = &[Activation::Relu, Activation::Relu];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);

    let inputs = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);
    
    let outputs = network.forward_batch(inputs.view());
    assert_eq!(outputs.shape(), [2, 2]);
}

#[test]
fn test_train_minibatch() {
    let layer_sizes = &[2, 4, 1];
    let activations = &[Activation::Relu, Activation::Relu];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);

    let inputs = arr2(&[
        [1.0, 2.0],
        [3.0, 4.0],
    ]);

    let targets = arr2(&[
        [1.0],
        [2.0],
    ]);

    network.train_minibatch(inputs.view(), targets.view(), 0.01);

    let new_output1 = network.forward(inputs.row(0));
    let new_output2 = network.forward(inputs.row(1));

    assert_eq!(new_output1.shape(), [1]);
    assert_eq!(new_output2.shape(), [1]);
}

#[test]
fn test_loss_functions() {
    let predictions = arr1(&[0.9, 0.1, 0.8]);
    let targets = arr1(&[1.0, 0.0, 1.0]);
    
    // Test MSE Loss
    let mse = MSELoss;
    let mse_loss = mse.compute(predictions.view(), targets.view());
    assert!((mse_loss - 0.01).abs() < 1e-4); // ((0.1)^2 + (0.1)^2 + (0.2)^2) / (2 * 3)
    
    // Test gradient
    let mse_grad = mse.gradient(predictions.view(), targets.view());
    assert_eq!(mse_grad.shape(), predictions.shape());
    
    // Test Huber Loss
    let huber = HuberLoss { delta: 1.0 };
    let huber_loss = huber.compute(predictions.view(), targets.view());
    assert!(huber_loss > 0.0);
    
    // Test Cross Entropy (binary)
    let ce = CrossEntropyLoss;
    let ce_loss = ce.compute(predictions.view(), targets.view());
    assert!(ce_loss > 0.0);
}

#[test]
fn test_network_with_different_activations() {
    let layer_sizes = &[2, 3, 1];
    let activations = &[Activation::LeakyRelu { alpha: 0.1 }, Activation::Sigmoid];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);
    
    let input = arr1(&[0.5, -0.5]);
    let output = network.forward(input.view());
    
    assert_eq!(output.shape(), [1]);
    assert!(output[0] >= 0.0 && output[0] <= 1.0); // Sigmoid output
}

#[test]
fn test_network_save_load() {
    use std::fs;
    
    let layer_sizes = &[2, 3, 1];
    let activations = &[Activation::Relu, Activation::Relu];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);
    
    // Forward pass to get initial output
    let input = arr1(&[1.0, 2.0]);
    let initial_output = network.forward(input.view()).to_owned();
    
    // Save network
    let path = "test_network.bin";
    network.save(path).unwrap();
    
    // Load network
    let mut loaded_network = NeuralNetwork::load(path).unwrap();
    
    // Verify same output
    let loaded_output = loaded_network.forward(input.view());
    assert_eq!(initial_output, loaded_output);
    
    // Cleanup
    fs::remove_file(path).ok();
}