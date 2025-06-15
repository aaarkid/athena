use ndarray::array;
use crate::agent::{DqnAgent, DqnAgentBuilder};
use crate::optimizer::{OptimizerWrapper, SGD};
use crate::replay_buffer::Experience;
use crate::activations::Activation;

#[test]
fn test_enhanced_dqn_agent() {
    let layer_sizes = [4, 32, 2];
    let epsilon = 0.5;
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let agent = DqnAgent::new(&layer_sizes, epsilon, optimizer, 100, true);
    
    assert_eq!(agent.epsilon, 0.5);
    assert_eq!(agent.q_network.layers.len(), 2);
    assert_eq!(agent.target_network.layers.len(), 2);
    assert_eq!(agent.target_update_freq, 100);
    assert!(agent.use_double_dqn);
}

#[test]
fn test_dqn_builder() {
    let agent = DqnAgentBuilder::new()
        .layer_sizes(&[4, 32, 2])
        .epsilon(0.3)
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .target_update_freq(200)
        .use_double_dqn(false)
        .build()
        .unwrap();
        
    assert_eq!(agent.epsilon, 0.3);
    assert_eq!(agent.target_update_freq, 200);
    assert!(!agent.use_double_dqn);
}

#[test]
fn test_builder_with_activations() {
    let activations = vec![Activation::LeakyRelu { alpha: 0.1 }, Activation::Sigmoid];
    let agent = DqnAgentBuilder::new()
        .layer_sizes(&[4, 32, 2])
        .activations(&activations)
        .epsilon(0.1)
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .build()
        .unwrap();
        
    assert_eq!(agent.epsilon, 0.1);
}

#[test]
fn test_target_network_update() {
    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[2, 4, 2])
        .epsilon(0.1)
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .target_update_freq(5)
        .build()
        .unwrap();
    
    // Modify Q-network weights
    agent.q_network.layers[0].weights[[0, 0]] = 999.0;
    
    // Target network should still have original weights
    assert_ne!(agent.target_network.layers[0].weights[[0, 0]], 999.0);
    
    // Update target network
    agent.update_target_network();
    
    // Now target network should match Q-network
    assert_eq!(agent.target_network.layers[0].weights[[0, 0]], 999.0);
}

#[test]
fn test_double_dqn_training() {
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    
    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[2, 4, 2])
        .epsilon(0.1)
        .optimizer(optimizer)
        .use_double_dqn(true)
        .build()
        .unwrap();
    
    let experience = Experience {
        state: array![0.5, -0.5],
        action: 0,
        reward: 1.0,
        next_state: array![0.6, -0.4],
        done: false,
    };
    
    // Train with Double DQN
    let _ = agent.train_on_batch(&[&experience], 0.99, 0.001).unwrap();
    
    // Should complete without errors
    assert_eq!(agent.train_steps, 1);
}

#[test]
fn test_train_steps_tracking() {
    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[2, 4, 2])
        .epsilon(0.1)
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .target_update_freq(3)
        .build()
        .unwrap();
    
    let experience = Experience {
        state: array![0.5, -0.5],
        action: 0,
        reward: 1.0,
        next_state: array![0.6, -0.4],
        done: false,
    };
    
    // Get initial target network weights
    let initial_target_weight = agent.target_network.layers[0].weights[[0, 0]];
    
    // Train for less than target_update_freq steps
    let _ = agent.train_on_batch(&[&experience], 0.99, 0.001).unwrap();
    assert_eq!(agent.train_steps, 1);
    assert_eq!(agent.target_network.layers[0].weights[[0, 0]], initial_target_weight);
    
    let _ = agent.train_on_batch(&[&experience], 0.99, 0.001).unwrap();
    assert_eq!(agent.train_steps, 2);
    assert_eq!(agent.target_network.layers[0].weights[[0, 0]], initial_target_weight);
    
    // Third step should trigger target update
    let _ = agent.train_on_batch(&[&experience], 0.99, 0.001).unwrap();
    assert_eq!(agent.train_steps, 3);
    
    // Target network should have been updated
    let q_weight = agent.q_network.layers[0].weights[[0, 0]];
    let target_weight = agent.target_network.layers[0].weights[[0, 0]];
    assert_eq!(q_weight, target_weight);
}

#[test]
fn test_agent_save_load() {
    use std::fs;
    
    let agent = DqnAgentBuilder::new()
        .layer_sizes(&[2, 4, 2])
        .epsilon(0.25)
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .target_update_freq(150)
        .use_double_dqn(true)
        .build()
        .unwrap();
    
    // Save agent
    let path = "test_agent.bin";
    agent.save(path).unwrap();
    
    // Load agent
    let loaded_agent = DqnAgent::load(path).unwrap();
    
    // Verify properties
    assert_eq!(loaded_agent.epsilon, 0.25);
    assert_eq!(loaded_agent.target_update_freq, 150);
    assert!(loaded_agent.use_double_dqn);
    assert_eq!(loaded_agent.train_steps, 0);
    
    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_agent_act_with_target() {
    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[2, 4, 2])
        .epsilon(0.0) // No exploration
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .build()
        .unwrap();
    
    let state = array![0.5, -0.5];
    
    // Both networks should give same action initially
    let action1 = agent.act(state.view());
    let action2 = agent.act(state.view());
    // Extract the actual action values from Result
    let action1_val = action1.unwrap();
    let action2_val = action2.unwrap();
    assert_eq!(action1_val, action2_val);
}

#[test]
fn test_builder_error_handling() {
    // Missing layer sizes
    let result = DqnAgentBuilder::new()
        .epsilon(0.1)
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .build();
    
    assert!(result.is_err());
    
    // Missing optimizer
    let result = DqnAgentBuilder::new()
        .layer_sizes(&[2, 4, 2])
        .epsilon(0.1)
        .build();
    
    assert!(result.is_err());
}