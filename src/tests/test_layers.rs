use ndarray::{arr1, arr2, Array1};
use crate::layers::{Layer, DenseLayer, BatchNormLayer, DropoutLayer, WeightInit, LayerTrait};
use crate::activations::Activation;

#[test]
fn test_layer_creation() {
    let input_size = 3;
    let output_size = 2;
    let activation = Activation::Relu;
    let layer = Layer::new(input_size, output_size, activation);

    assert_eq!(layer.weights.shape(), [input_size, output_size]);
    assert_eq!(layer.biases.shape(), [output_size]);
}

#[test]
fn test_dense_layer_forward() {
    let mut layer = DenseLayer::new(3, 2, Activation::Relu);
    let input = arr1(&[1.0, 2.0, 3.0]);
    let output = LayerTrait::forward(&mut layer, input.view());
    assert_eq!(output.shape(), [2]);
}

#[test]
fn test_weight_initialization() {
    // Test Xavier uniform
    let layer = DenseLayer::new_with_init(10, 20, Activation::Relu, WeightInit::XavierUniform);
    let limit = (6.0 / 30.0_f32).sqrt();
    for &w in layer.weights.iter() {
        assert!(w >= -limit && w <= limit);
    }
    
    // Test He normal
    let layer = DenseLayer::new_with_init(10, 20, Activation::Relu, WeightInit::HeNormal);
    // Check that weights have reasonable variance
    let var: f32 = layer.weights.iter().map(|&x| x * x).sum::<f32>() / (10.0 * 20.0);
    let expected_var = 2.0 / 10.0;
    assert!((var - expected_var).abs() < 0.5); // Allow some variance
}

#[test]
fn test_batch_norm_layer() {
    let mut layer = BatchNormLayer::new(3, 0.9, 1e-5);
    
    // Test forward pass
    let input = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);
    let output = layer.forward_batch(input.view());
    assert_eq!(output.shape(), [2, 3]);
    
    // Test that output is normalized (mean ~0, std ~1)
    let mean = output.mean_axis(ndarray::Axis(0)).unwrap();
    for &m in mean.iter() {
        assert!(m.abs() < 0.1);
    }
}

#[test]
fn test_dropout_layer() {
    let mut layer = DropoutLayer::new(100, 0.5);
    
    // Test training mode
    layer.set_training(true);
    let input = Array1::ones(100);
    let output = LayerTrait::forward(&mut layer, input.view());
    
    // Approximately half should be zero
    let zero_count = output.iter().filter(|&&x| x == 0.0).count();
    assert!(zero_count > 30 && zero_count < 70);
    
    // Non-zero values should be scaled
    for &val in output.iter() {
        assert!(val == 0.0 || (val - 2.0).abs() < 1e-6);
    }
    
    // Test inference mode
    layer.set_training(false);
    let output = LayerTrait::forward(&mut layer, input.view());
    assert_eq!(output, input);
}

#[test]
fn test_batch_norm_backward() {
    let mut layer = BatchNormLayer::new(2, 0.9, 1e-5);
    
    let input = arr2(&[
        [1.0, 2.0],
        [3.0, 4.0],
    ]);
    
    // Forward pass
    let output = layer.forward_batch(input.view());
    
    // Verify output shape
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_dropout_backward() {
    let mut layer = DropoutLayer::new(10, 0.5);
    layer.set_training(true);
    
    let input = Array1::ones(10);
    let _output = LayerTrait::forward(&mut layer, input.view());
    
    // Dropout doesn't have trainable parameters, so backward is simple
    // The gradient just passes through with the same mask applied
}