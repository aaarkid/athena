use ndarray::array;
use crate::activations::Activation;

#[test]
fn test_relu_activation() {
    let relu = Activation::Relu;
    let mut input = array![-1.0, 0.0, 1.0, 2.0];
    relu.apply(&mut input);
    assert_eq!(input, array![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sigmoid_activation() {
    let sigmoid = Activation::Sigmoid;
    let mut input = array![0.0];
    sigmoid.apply(&mut input);
    assert!((input[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_tanh_activation() {
    let tanh = Activation::Tanh;
    let mut input = array![0.0];
    tanh.apply(&mut input);
    assert_eq!(input[0], 0.0);
}

#[test]
fn test_leaky_relu() {
    let leaky = Activation::LeakyRelu { alpha: 0.01 };
    let mut input = array![-1.0, 0.0, 1.0];
    leaky.apply(&mut input);
    assert_eq!(input, array![-0.01, 0.0, 1.0]);
}

#[test]
fn test_elu() {
    let elu = Activation::Elu { alpha: 1.0 };
    let mut input = array![-1.0, 0.0, 1.0];
    elu.apply(&mut input);
    assert!((input[0] - (-0.632)).abs() < 0.001);
    assert_eq!(input[1], 0.0);
    assert_eq!(input[2], 1.0);
}

#[test]
fn test_gelu() {
    let gelu = Activation::Gelu;
    let mut input = array![0.0, 1.0, -1.0];
    gelu.apply(&mut input);
    
    // GELU(0) ≈ 0
    assert!(input[0].abs() < 1e-6);
    // GELU(1) ≈ 0.841
    assert!((input[1] - 0.841).abs() < 0.01);
    // GELU(-1) ≈ -0.159
    assert!((input[2] - (-0.159)).abs() < 0.01);
}

#[test]
fn test_activation_derivatives() {
    
    // Test ReLU derivative
    let relu = Activation::Relu;
    let output = array![-1.0, 0.0, 1.0, 2.0];
    let deriv = relu.derivative(&output);
    assert_eq!(deriv, array![0.0, 0.0, 1.0, 1.0]);
    
    // Test LeakyReLU derivative
    let leaky = Activation::LeakyRelu { alpha: 0.1 };
    let output = array![-1.0, 0.0, 1.0];
    let deriv = leaky.derivative(&output);
    assert_eq!(deriv, array![0.1, 0.1, 1.0]);
}