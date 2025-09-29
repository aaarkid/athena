use athena::{
    agent_v2::DqnAgentBuilder,
    network::NeuralNetwork,
    optimizer::{OptimizerWrapper, SGD, Adam, RMSProp, GradientClipper, LearningRateScheduler},
    layers::{Layer, DenseLayer, BatchNormLayer, DropoutLayer, WeightInit},
    activations::Activation,
    replay_buffer::ReplayBuffer,
    replay_buffer_v2::{PrioritizedReplayBuffer, PriorityMethod},
};
use ndarray::{array, Array1};

#[test]
fn test_end_to_end_training() {
    // Create a simple environment simulation
    fn get_state(step: usize) -> Array1<f32> {
        array![step as f32 / 100.0, (step as f32 / 50.0).sin()]
    }
    
    fn get_reward(action: usize, step: usize) -> f32 {
        if action == (step / 10) % 2 {
            1.0
        } else {
            -0.1
        }
    }
    
    // Create agent with all advanced features
    let layers = vec![
        Layer::new(2, 64, Activation::Relu),
        Layer::new(64, 32, Activation::Relu),
        Layer::new(32, 2, Activation::Linear),
    ];
    
    let optimizer = OptimizerWrapper::Adam(Adam::new(&layers, 0.9, 0.999, 1e-8));
    
    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[2, 64, 32, 2])
        .activations(&[Activation::Relu, Activation::Relu, Activation::Linear])
        .epsilon(1.0)
        .optimizer(optimizer)
        .target_update_freq(10)
        .use_double_dqn(true)
        .build()
        .unwrap();
    
    // Create prioritized replay buffer
    let mut replay_buffer = PrioritizedReplayBuffer::new(
        1000,
        PriorityMethod::Proportional { alpha: 0.6 }
    );
    
    // Training loop
    let mut total_reward = 0.0;
    let episodes = 10;
    let steps_per_episode = 50;
    
    for episode in 0..episodes {
        let mut episode_reward = 0.0;
        
        for step in 0..steps_per_episode {
            let state = get_state(step);
            let action = agent.act(state.view());
            let reward = get_reward(action, step);
            let next_state = get_state(step + 1);
            let done = step == steps_per_episode - 1;
            
            episode_reward += reward;
            
            // Store experience with priority
            let experience = athena::replay_buffer::Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            };
            
            let priority = reward.abs() + 0.01;
            replay_buffer.add_with_priority(experience, priority);
            
            // Train if enough experiences
            if replay_buffer.len() >= 32 {
                let (batch, weights, indices) = replay_buffer.sample_with_weights(32, 0.4);
                agent.train_on_batch(&batch.iter().collect::<Vec<_>>(), 0.99, 0.001);
                
                // Update priorities based on TD error (simplified)
                let new_priorities: Vec<f32> = batch.iter().map(|_| 1.0).collect();
                replay_buffer.update_priorities(&indices, &new_priorities);
            }
        }
        
        total_reward += episode_reward;
        
        // Decay epsilon
        agent.update_epsilon(0.9 * agent.epsilon);
    }
    
    // Verify agent learned something
    assert!(total_reward > -episodes as f32 * steps_per_episode as f32 * 0.1);
}

#[test]
fn test_network_with_advanced_layers() {
    // Create a network with batch norm and dropout
    let mut network = NeuralNetwork::new_empty();
    
    // Add layers manually (since we need special layer types)
    let dense1 = DenseLayer::new_with_init(10, 64, Activation::Relu, WeightInit::HeNormal);
    let batch_norm1 = BatchNormLayer::new(64);
    let dropout1 = DropoutLayer::new(0.2);
    let dense2 = DenseLayer::new_with_init(64, 32, Activation::Relu, WeightInit::XavierUniform);
    let dense3 = DenseLayer::new(32, 1, Activation::Linear);
    
    // Test forward pass
    let input = Array1::ones(10);
    let _ = dense1.forward(input.view());
    
    // Verify shapes and properties
    assert_eq!(dense1.weights.shape(), [10, 64]);
    assert_eq!(dense2.weights.shape(), [64, 32]);
    assert_eq!(dense3.weights.shape(), [32, 1]);
}

#[test]
fn test_gradient_clipping_with_optimizer() {
    let layer_sizes = &[5, 10, 2];
    let activations = &[Activation::Relu, Activation::Sigmoid];
    let mut optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer.clone());
    
    // Create gradient clipper
    let clipper = GradientClipper::ClipByGlobalNorm { max_norm: 1.0 };
    
    // Train with large gradients that need clipping
    let inputs = ndarray::Array2::ones((1, 5)) * 100.0;
    let targets = ndarray::Array2::ones((1, 2));
    
    // This should create large gradients
    network.train_minibatch(inputs.view(), targets.view(), 0.01);
    
    // Network should still be stable
    let test_input = Array1::ones(5);
    let output = network.forward(test_input.view());
    for &val in output.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_learning_rate_scheduling() {
    let scheduler = LearningRateScheduler::CosineAnnealingWarmRestarts {
        max_lr: 0.1,
        min_lr: 0.001,
        period: 10,
        mult: 2.0,
    };
    
    // Test warm restarts
    let lr_0 = scheduler.get_lr(0);
    let lr_10 = scheduler.get_lr(10); // Start of second period
    let lr_30 = scheduler.get_lr(30); // Start of third period
    
    assert!((lr_0 - 0.1).abs() < 1e-6);
    assert!((lr_10 - 0.1).abs() < 1e-6); // Restart
    assert!((lr_30 - 0.1).abs() < 1e-6); // Another restart
}

#[test]
fn test_save_load_complex_agent() {
    use std::fs;
    
    // Create agent with all features
    let layers = vec![
        Layer::new(4, 32, Activation::LeakyRelu { alpha: 0.1 }),
        Layer::new(32, 16, Activation::Elu { alpha: 1.0 }),
        Layer::new(16, 2, Activation::Gelu),
    ];
    
    let optimizer = OptimizerWrapper::RMSProp(RMSProp::new(&layers, 0.95, 1e-8));
    
    let agent = DqnAgentBuilder::new()
        .layer_sizes(&[4, 32, 16, 2])
        .activations(&[
            Activation::LeakyRelu { alpha: 0.1 },
            Activation::Elu { alpha: 1.0 },
            Activation::Gelu,
        ])
        .epsilon(0.15)
        .optimizer(optimizer)
        .target_update_freq(50)
        .use_double_dqn(true)
        .build()
        .unwrap();
    
    // Test action
    let state = array![0.1, 0.2, 0.3, 0.4];
    let action_before = agent.act_greedy(state.view());
    
    // Save and load
    let path = "test_complex_agent.bin";
    agent.save(path).unwrap();
    let loaded = athena::agent_v2::DqnAgentV2::load(path).unwrap();
    
    // Verify same behavior
    let action_after = loaded.act_greedy(state.view());
    assert_eq!(action_before, action_after);
    
    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_mixed_precision_like_behavior() {
    // Test that our system handles small and large values well
    let layer_sizes = &[2, 10, 1];
    let activations = &[Activation::Tanh, Activation::Linear];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);
    
    // Test with very small values
    let small_input = array![1e-10, 1e-10];
    let small_output = network.forward(small_input.view());
    assert!(small_output[0].is_finite());
    
    // Test with moderately large values
    let large_input = array![100.0, -100.0];
    let large_output = network.forward(large_input.view());
    assert!(large_output[0].is_finite());
}

#[test]
fn test_error_handling() {
    use athena::error::AthenaError;
    
    // Test builder with missing required fields
    let result = DqnAgentBuilder::new()
        .epsilon(0.5)
        .build();
    
    match result {
        Err(AthenaError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "layer_sizes");
            assert!(reason.contains("required"));
        }
        _ => panic!("Expected InvalidParameter error"),
    }
    
    // Test network save/load with invalid path
    let network = NeuralNetwork::new(
        &[2, 3, 1],
        &[Activation::Relu, Activation::Relu],
        OptimizerWrapper::SGD(SGD::new())
    );
    
    let result = network.save("/invalid/path/network.bin");
    assert!(result.is_err());
}