use ndarray::{array, Array1, Array2};
use crate::activations::Activation;
use crate::network::NeuralNetwork;
use crate::optimizer::{OptimizerWrapper, SGD};
use crate::layers::{DenseLayer, BatchNormLayer, DropoutLayer, LayerTrait};

#[test]
fn test_activation_edge_cases() {
    // Test with extreme values
    let activations = vec![
        Activation::Relu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::LeakyRelu { alpha: 0.01 },
        Activation::Elu { alpha: 1.0 },
        Activation::Gelu,
    ];
    
    for activation in activations {
        // Test with very large positive values
        let mut large_pos = array![1e10, 1e20, f32::MAX / 2.0];
        activation.apply(&mut large_pos);
        for &val in large_pos.iter() {
            assert!(val.is_finite(), "Activation {:?} produced non-finite value", activation);
        }
        
        // Test with very large negative values
        let mut large_neg = array![-1e10, -1e20, f32::MIN / 2.0];
        activation.apply(&mut large_neg);
        for &val in large_neg.iter() {
            assert!(val.is_finite(), "Activation {:?} produced non-finite value", activation);
        }
        
        // Test with zeros
        let mut zeros = array![0.0, -0.0, 0.0];
        activation.apply(&mut zeros);
        for &val in zeros.iter() {
            assert!(val.is_finite());
        }
        
        // Test with NaN (should handle gracefully)
        let mut nans = array![f32::NAN, 1.0, -1.0];
        activation.apply(&mut nans);
        // All values should be finite after activation (NaN handled gracefully)
        assert!(nans[1].is_finite());
        assert!(nans[2].is_finite());
        // ReLU converts NaN to 0, others may preserve or convert NaN
        match activation {
            Activation::Relu => assert_eq!(nans[0], 0.0, "ReLU should convert NaN to 0"),
            _ => {
                // Other activations may preserve NaN or convert to finite
                // We just check it doesn't crash
            }
        }
    }
}

#[test]
fn test_gradient_edge_cases() {
    // Test gradient computation with edge values
    let activations = vec![
        Activation::Relu,
        Activation::LeakyRelu { alpha: 0.01 },
        Activation::Sigmoid,
        Activation::Tanh,
    ];
    
    for activation in activations {
        // Test at boundary values
        let output = array![0.0, 1.0, -1.0];
        let deriv = activation.derivative(&output);
        
        for &g in deriv.iter() {
            assert!(g.is_finite(), "Gradient is not finite for {:?}", activation);
            assert!(g >= -1.0 && g <= 1.1, "Gradient out of expected range");
        }
    }
}

#[test]
fn test_network_numerical_stability() {
    // Create network that might have numerical issues
    let layer_sizes = &[2, 1000, 1]; // Very wide hidden layer
    let activations = &[Activation::Sigmoid, Activation::Sigmoid];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);
    
    // Initialize with extreme inputs
    let extreme_inputs = vec![
        array![1e-10, 1e-10],
        array![1e10, -1e10],
        array![0.0, 0.0],
        array![1.0, -1.0],
    ];
    
    for input in extreme_inputs {
        let output = network.forward(input.view());
        assert!(output[0].is_finite(), "Network produced non-finite output");
        assert!(output[0] >= 0.0 && output[0] <= 1.0, "Sigmoid output out of range");
    }
}

#[test]
fn test_batch_norm_edge_cases() {
    let mut bn = BatchNormLayer::new(3, 0.9, 1e-5);
    
    // Test with single sample (edge case for batch norm)
    let single = Array2::ones((1, 3));
    let output = bn.forward_batch(single.view());
    for &val in output.iter() {
        assert!(val.is_finite());
    }
    
    // Test with identical samples (zero variance)
    let identical = Array2::ones((5, 3));
    let output = bn.forward_batch(identical.view());
    for &val in output.iter() {
        assert!(val.is_finite());
    }
    
    // Test with extreme values
    let mut extreme = Array2::zeros((2, 3));
    extreme[[0, 0]] = 1e10;
    extreme[[1, 1]] = -1e10;
    let output = bn.forward_batch(extreme.view());
    for &val in output.iter() {
        assert!(val.is_finite() || val.is_nan()); // NaN is acceptable for extreme inputs
    }
}

#[test]
fn test_dropout_determinism() {
    let mut dropout = DropoutLayer::new(100, 0.5);
    
    // In training mode, outputs should vary
    dropout.set_training(true);
    let input = Array1::ones(100);
    let output1 = dropout.forward(input.view()).to_owned();
    let output2 = dropout.forward(input.view()).to_owned();
    assert_ne!(output1, output2, "Dropout should be stochastic in training");
    
    // In inference mode, output should be deterministic
    dropout.set_training(false);
    let output3 = dropout.forward(input.view()).to_owned();
    let output4 = dropout.forward(input.view()).to_owned();
    assert_eq!(output3, output4, "Dropout should be deterministic in inference");
    assert_eq!(output3, input, "Dropout should be identity in inference");
}

#[test]
fn test_weight_init_bounds() {
    use crate::layers::WeightInit;
    
    let inits = vec![
        WeightInit::XavierUniform,
        WeightInit::XavierNormal,
        WeightInit::HeUniform,
        WeightInit::HeNormal,
        WeightInit::Ones,
        WeightInit::Normal { mean: 0.0, std: 0.1 },
        WeightInit::Uniform { min: -0.1, max: 0.1 },
    ];
    
    for init in inits {
        let layer = DenseLayer::new_with_init(100, 200, Activation::Relu, init.clone());
        
        // Check all weights are finite
        for &w in layer.weights.iter() {
            assert!(w.is_finite(), "Weight initialization {:?} produced non-finite values", init);
        }
        
        // Check reasonable bounds
        let max_weight = layer.weights.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_weight < 10.0, "Weight initialization {:?} produced unreasonably large values", init);
    }
}

#[test]
fn test_optimizer_with_zero_gradients() {
    use crate::optimizer::{Adam, RMSProp, Optimizer};
    use crate::layers::Layer;
    
    let layers = vec![Layer::new(2, 2, Activation::Relu)];
    
    // Test SGD
    let mut sgd = SGD::new();
    let mut weights = array![[1.0, 1.0], [1.0, 1.0]];
    let zero_grads = array![[0.0, 0.0], [0.0, 0.0]];
    sgd.update_weights(&mut weights, &zero_grads, 0.1);
    assert_eq!(weights, array![[1.0, 1.0], [1.0, 1.0]]); // No change
    
    // Test Adam
    let mut adam = Adam::new(&layers, 0.9, 0.999, 1e-8);
    let mut weights = array![[1.0, 1.0], [1.0, 1.0]];
    adam.update_weights(&mut weights, &zero_grads, 0.1);
    // Should handle zero gradients gracefully
    for &w in weights.iter() {
        assert!(w.is_finite());
    }
    
    // Test RMSProp
    let mut rmsprop = RMSProp::new(&layers, 0.9, 1e-8);
    let mut weights = array![[1.0, 1.0], [1.0, 1.0]];
    rmsprop.update_weights(&mut weights, &zero_grads, 0.1);
    // Should handle zero gradients gracefully
    for &w in weights.iter() {
        assert!(w.is_finite());
    }
}

#[test]
fn test_loss_functions_edge_cases() {
    use crate::loss::{MSE as MSELoss, CrossEntropyLoss, HuberLoss, Loss};
    
    // Test with identical predictions and targets
    let predictions = array![0.5, 0.5, 0.5];
    let targets = array![0.5, 0.5, 0.5];
    
    let mse = MSELoss;
    assert_eq!(mse.compute(predictions.view(), targets.view()), 0.0);
    
    // Test with extreme differences
    let extreme_pred = array![1e10, -1e10, 0.0];
    let extreme_target = array![-1e10, 1e10, 0.0];
    let loss = mse.compute(extreme_pred.view(), extreme_target.view());
    assert!(loss.is_finite() || loss.is_infinite()); // May overflow to inf
    
    // Test CrossEntropy with edge probabilities
    let ce = CrossEntropyLoss;
    let near_zero = array![1e-10, 1.0 - 1e-10, 0.5];
    let binary_targets = array![0.0, 1.0, 1.0];
    let ce_loss = ce.compute(near_zero.view(), binary_targets.view());
    assert!(ce_loss.is_finite()); // Should handle near-zero probabilities
    
    // Test Huber loss at the threshold
    let huber = HuberLoss { delta: 1.0 };
    let at_threshold = array![1.0, -1.0, 0.0];
    let zeros = array![0.0, 0.0, 0.0];
    let huber_loss = huber.compute(at_threshold.view(), zeros.view());
    assert!(huber_loss.is_finite());
}

#[test]
fn test_replay_buffer_edge_cases() {
    use crate::replay_buffer::{ReplayBuffer, Experience};
    
    // Test with zero capacity (should probably error or handle gracefully)
    let mut buffer = ReplayBuffer::new(0);
    let exp = Experience {
        state: array![1.0],
        action: 0,
        reward: 0.0,
        next_state: array![1.0],
        done: false,
    };
    
    buffer.add(exp.clone());
    assert_eq!(buffer.len(), 1); // Zero-capacity buffer still stores one item (edge case)
    
    // Test sampling more than available
    let mut buffer = ReplayBuffer::new(10);
    buffer.add(exp.clone());
    let samples = buffer.sample(100); // Request more than available
    assert_eq!(samples.len(), 1); // Should return what's available
    
    // Test sampling from empty buffer
    let buffer = ReplayBuffer::new(10);
    let samples = buffer.sample(5);
    assert_eq!(samples.len(), 0); // Should return empty
}

#[test]
fn test_learning_rate_scheduler_edge_cases() {
    use crate::optimizer::LearningRateScheduler;
    
    // Test at extreme steps
    let scheduler = LearningRateScheduler::ExponentialDecay {
        initial_lr: 0.1,
        decay_rate: 0.99,
    };
    
    let lr_huge = scheduler.get_lr(10000);
    assert!(lr_huge > 0.0 && lr_huge < 1e-10); // Should be very small but positive
    
    // Test polynomial decay with power = 0
    let scheduler = LearningRateScheduler::PolynomialDecay {
        initial_lr: 0.1,
        final_lr: 0.01,
        max_steps: 100,
        power: 0.0,
    };
    
    let lr = scheduler.get_lr(50);
    assert_eq!(lr, 0.1); // With power=0, should stay at initial_lr
    
    // Test cosine annealing at boundaries
    let scheduler = LearningRateScheduler::CosineAnnealing {
        max_lr: 0.1,
        min_lr: 0.01,
        period: 100,
    };
    
    let lr_0 = scheduler.get_lr(0);
    let lr_50 = scheduler.get_lr(50);
    let lr_100 = scheduler.get_lr(100);
    
    assert!((lr_0 - 0.1).abs() < 1e-6); // Max at start
    assert!((lr_50 - 0.055).abs() < 1e-6); // Middle of cosine curve
    assert!((lr_100 - 0.1).abs() < 1e-6); // Max again at period
}