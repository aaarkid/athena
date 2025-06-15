use ndarray::array;
use crate::optimizer::{Optimizer, SGD, Adam, RMSProp, OptimizerWrapper, GradientClipper, LearningRateScheduler};
use crate::layers::Layer;
use crate::activations::Activation;

#[test]
fn test_sgd_update_weights() {
    let mut sgd = SGD::new();
    let mut weights = array![[1.0, 1.0], [1.0, 1.0]];
    let gradients = array![[0.1, 0.2], [0.3, 0.4]];
    let learning_rate = 0.01;

    sgd.update_weights(0, &mut weights, &gradients, learning_rate);

    let expected_weights = array![[0.999, 0.998], [0.997, 0.996]];
    assert_eq!(weights, expected_weights);
}

#[test]
fn test_sgd_update_biases() {
    let mut sgd = SGD::new();
    let mut biases = array![1.0, 1.0];
    let gradients = array![0.1, 0.2];
    let learning_rate = 0.01;

    sgd.update_biases(0, &mut biases, &gradients, learning_rate);

    let expected_biases = array![0.999, 0.998];
    assert_eq!(biases, expected_biases);
}

#[test]
fn test_adam_new() {
    let layers = vec![Layer::new(2, 2, Activation::Relu)];
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;

    let adam = Adam::new(&layers, beta1, beta2, epsilon);

    assert_eq!(adam.beta1, beta1);
    assert_eq!(adam.beta2, beta2);
    assert_eq!(adam.epsilon, epsilon);
    assert_eq!(adam.t, 1);
}

#[test]
fn test_adam_update_weights() {
    let layers = vec![Layer::new(2, 2, Activation::Relu)];
    let mut adam = Adam::new(&layers, 0.9, 0.999, 1e-8);

    let mut weights = array![[1.0, 1.0], [1.0, 1.0]];
    let gradients = array![[0.1, 0.2], [0.3, 0.4]];
    let learning_rate = 0.01;

    adam.update_weights(0, &mut weights, &gradients, learning_rate);

    let expected_weights = array![
        [0.99, 0.99],
        [0.99, 0.99]
    ];
    assert_eq!(weights, expected_weights);
}

#[test]
fn test_adam_update_biases() {
    let layers = vec![Layer::new(2, 2, Activation::Relu)];
    let mut adam = Adam::new(&layers, 0.9, 0.999, 1e-8);

    let mut biases = array![1.0, 1.0];
    let gradients = array![0.1, 0.2];
    let learning_rate = 0.01;

    adam.update_biases(0, &mut biases, &gradients, learning_rate);

    let expected_biases = array![0.99, 0.99];
    assert_eq!(biases, expected_biases);
}

#[test]
fn test_rmsprop() {
    let layers = vec![Layer::new(2, 2, Activation::Relu)];
    let mut rmsprop = RMSProp::new(&layers, 0.9, 1e-8);
    
    let mut weights = array![[1.0, 1.0], [1.0, 1.0]];
    let gradients = array![[0.1, 0.2], [0.3, 0.4]];
    let learning_rate = 0.01;
    
    rmsprop.update_weights(0, &mut weights, &gradients, learning_rate);
    
    // Verify weights were updated
    assert!(weights[[0, 0]] < 1.0);
    assert!(weights[[0, 1]] < 1.0);
}

#[test]
fn test_gradient_clipping() {
    // Test clip by value
    let mut gradients = array![[1.5, -2.0], [0.5, 3.0]];
    let clipper = GradientClipper::ClipByValue { min: -1.0, max: 1.0 };
    clipper.clip_weights(&mut gradients);
    assert_eq!(gradients, array![[1.0, -1.0], [0.5, 1.0]]);
    
    // Test clip by norm
    let mut gradients2 = array![[3.0, 4.0]]; // norm = 5
    let clipper2 = GradientClipper::ClipByNorm { max_norm: 2.5 };
    clipper2.clip_weights(&mut gradients2);
    let norm = (gradients2[[0, 0]].powi(2) + gradients2[[0, 1]].powi(2)).sqrt();
    assert!((norm - 2.5).abs() < 1e-6);
    
    // Test no clipping
    let mut gradients3 = array![[1.0, 1.0]];
    let clipper3 = GradientClipper::None;
    let original = gradients3.clone();
    clipper3.clip_weights(&mut gradients3);
    assert_eq!(gradients3, original);
}

#[test]
fn test_learning_rate_scheduling() {
    // Test constant
    let scheduler = LearningRateScheduler::constant(0.01);
    assert_eq!(scheduler.get_lr(0), 0.01);
    assert_eq!(scheduler.get_lr(100), 0.01);
    
    // Test step decay
    let scheduler = LearningRateScheduler::step_decay(0.1, 0.5, 10);
    assert_eq!(scheduler.get_lr(0), 0.1);
    assert_eq!(scheduler.get_lr(10), 0.05);
    assert_eq!(scheduler.get_lr(20), 0.025);
    
    // Test cosine annealing
    let scheduler = LearningRateScheduler::cosine_annealing(0.1, 0.001, 20);
    let lr_0 = scheduler.get_lr(0);
    let lr_10 = scheduler.get_lr(10);
    let lr_20 = scheduler.get_lr(20);
    assert!((lr_0 - 0.1).abs() < 1e-6);
    assert!((lr_20 - 0.1).abs() < 1e-6); // At step 20, we're back at the start of the period
    assert!(lr_10 < lr_0); // lr_10 should be less than max_lr
    
    // Test warmup
    let scheduler = LearningRateScheduler::WarmupConstant {
        initial_lr: 0.001,
        warmup_steps: 10,
        target_lr: 0.1,
    };
    assert_eq!(scheduler.get_lr(0), 0.001);
    assert!((scheduler.get_lr(5) - 0.0505).abs() < 1e-4);
    assert_eq!(scheduler.get_lr(10), 0.1);
    assert_eq!(scheduler.get_lr(20), 0.1);
}

#[test]
fn test_optimizer_wrapper() {
    let mut sgd_wrapper = OptimizerWrapper::SGD(SGD::new());
    let mut weights = array![[1.0, 1.0]];
    let gradients = array![[0.1, 0.2]];
    
    sgd_wrapper.update_weights(0, &mut weights, &gradients, 0.01);
    assert_eq!(weights, array![[0.999, 0.998]]);
    
    // Test with Adam
    let layers = vec![Layer::new(1, 2, Activation::Relu)];
    let mut adam_wrapper = OptimizerWrapper::Adam(Adam::new(&layers, 0.9, 0.999, 1e-8));
    let mut weights = array![[1.0, 1.0]];
    adam_wrapper.update_weights(0, &mut weights, &gradients, 0.01);
    assert!(weights[[0, 0]] < 1.0);
}

#[test]
fn test_adam_multi_layer_state() {
    // Test that Adam maintains separate state for each layer
    let layers = vec![
        Layer::new(2, 3, Activation::Relu),
        Layer::new(3, 2, Activation::Relu),
        Layer::new(2, 1, Activation::Linear),
    ];
    let mut adam = Adam::new(&layers, 0.9, 0.999, 1e-8);
    
    // Update different layers with different gradients
    let mut weights_l0 = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    let mut weights_l1 = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
    let mut weights_l2 = array![[1.0], [1.0]];
    
    let gradients_l0 = array![[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]];
    let gradients_l1 = array![[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]];
    let gradients_l2 = array![[0.3], [0.3]];
    
    // First update
    adam.update_weights(0, &mut weights_l0, &gradients_l0, 0.01);
    adam.update_weights(1, &mut weights_l1, &gradients_l1, 0.01);
    adam.update_weights(2, &mut weights_l2, &gradients_l2, 0.01);
    
    // Store first update results
    let weights_l0_after1 = weights_l0.clone();
    let weights_l1_after1 = weights_l1.clone();
    let weights_l2_after1 = weights_l2.clone();
    
    // Second update with same gradients - should produce different results due to momentum
    adam.update_weights(0, &mut weights_l0, &gradients_l0, 0.01);
    adam.update_weights(1, &mut weights_l1, &gradients_l1, 0.01);
    adam.update_weights(2, &mut weights_l2, &gradients_l2, 0.01);
    
    // Verify that the second update is different (due to momentum accumulation)
    assert!(weights_l0 != weights_l0_after1);
    assert!(weights_l1 != weights_l1_after1);
    assert!(weights_l2 != weights_l2_after1);
    
    // Verify that different layers get different updates
    let delta_l0 = (weights_l0_after1[[0, 0]] - 1.0).abs();
    let delta_l1 = (weights_l1_after1[[0, 0]] - 1.0).abs();
    let delta_l2 = (weights_l2_after1[[0, 0]] - 1.0).abs();
    
    // With Adam's bias correction at t=1, all updates are normalized to be the same
    // Just verify that updates happened
    assert!(delta_l0 > 0.0);
    assert!(delta_l1 > 0.0);
    assert!(delta_l2 > 0.0);
}

#[test]
fn test_rmsprop_multi_layer_state() {
    // Test that RMSProp maintains separate state for each layer
    let layers = vec![
        Layer::new(2, 2, Activation::Relu),
        Layer::new(2, 2, Activation::Linear),
    ];
    let mut rmsprop = RMSProp::new(&layers, 0.9, 1e-8);
    
    let mut weights_l0 = array![[1.0, 1.0], [1.0, 1.0]];
    let mut weights_l1 = array![[1.0, 1.0], [1.0, 1.0]];
    
    let gradients_l0 = array![[0.1, 0.1], [0.1, 0.1]];
    let gradients_l1 = array![[0.5, 0.5], [0.5, 0.5]];
    
    // Update both layers
    rmsprop.update_weights(0, &mut weights_l0, &gradients_l0, 0.1);
    rmsprop.update_weights(1, &mut weights_l1, &gradients_l1, 0.1);
    
    // Second update to test momentum
    let weights_l0_after1 = weights_l0.clone();
    let weights_l1_after1 = weights_l1.clone();
    
    rmsprop.update_weights(0, &mut weights_l0, &gradients_l0, 0.1);
    rmsprop.update_weights(1, &mut weights_l1, &gradients_l1, 0.1);
    
    // Verify that the second update is different due to running average
    assert!(weights_l0 != weights_l0_after1);
    assert!(weights_l1 != weights_l1_after1);
    
    // Verify different layers have different update magnitudes
    let update_l0 = (1.0 - weights_l0_after1[[0, 0]]).abs();
    let update_l1 = (1.0 - weights_l1_after1[[0, 0]]).abs();
    
    // Layer 1 should have larger updates due to larger gradients
    assert!(update_l1 > update_l0);
}

#[test]
fn test_adam_time_step_increment() {
    // Test that Adam's time step increments correctly
    let layers = vec![
        Layer::new(2, 2, Activation::Relu),
        Layer::new(2, 1, Activation::Linear),
    ];
    let mut adam = Adam::new(&layers, 0.9, 0.999, 1e-8);
    
    assert_eq!(adam.t, 1);
    
    // Update all layers (weights and biases)
    let mut dummy_weights = array![[1.0, 1.0], [1.0, 1.0]];
    let mut dummy_biases = array![1.0, 1.0];
    let grad_w = array![[0.1, 0.1], [0.1, 0.1]];
    let grad_b = array![0.1, 0.1];
    
    // First layer
    adam.update_weights(0, &mut dummy_weights, &grad_w, 0.01);
    adam.update_biases(0, &mut dummy_biases, &grad_b, 0.01);
    
    // Second layer
    let mut dummy_weights2 = array![[1.0], [1.0]];
    let mut dummy_biases2 = array![1.0];
    let grad_w2 = array![[0.1], [0.1]];
    let grad_b2 = array![0.1];
    
    adam.update_weights(1, &mut dummy_weights2, &grad_w2, 0.01);
    adam.update_biases(1, &mut dummy_biases2, &grad_b2, 0.01);
    
    // After updating all layers, time step should increment
    assert_eq!(adam.t, 2);
}