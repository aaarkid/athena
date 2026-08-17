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
    use crate::optimizer::Adam;
    use ndarray::Array1;
    use std::fs;

    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[2, 4, 2])
        .epsilon(0.25)
        .optimizer(OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8)))
        .target_update_freq(150)
        .use_double_dqn(true)
        .build()
        .unwrap();

    // Train first: an untrained agent's weights would round-trip even if the file were
    // saving something unrelated to what training changes
    let experiences: Vec<Experience> = (0..8)
        .map(|i| Experience {
            state: array![i as f32 * 0.1, 1.0 - i as f32 * 0.1],
            action: i % 2,
            reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            next_state: array![0.5, 0.5],
            done: i % 4 == 0,
        })
        .collect();
    for _ in 0..200 {
        let batch: Vec<&Experience> = experiences.iter().collect();
        agent.train_on_batch(&batch, 0.99, 1e-3).unwrap();
    }

    let path = "test_agent.bin";
    agent.save(path).unwrap();
    let loaded_agent = DqnAgent::load(path).unwrap();

    assert_eq!(loaded_agent.epsilon, 0.25);
    assert_eq!(loaded_agent.target_update_freq, 150);
    assert!(loaded_agent.use_double_dqn);
    assert_eq!(loaded_agent.train_steps, agent.train_steps);
    assert_eq!(agent.train_steps, 200);

    // Every parameter of both networks, bit for bit
    for (name, original, loaded) in [
        ("q", &agent.q_network, &loaded_agent.q_network),
        ("target", &agent.target_network, &loaded_agent.target_network),
    ] {
        assert_eq!(original.layers.len(), loaded.layers.len());
        for (i, (a, b)) in original.layers.iter().zip(loaded.layers.iter()).enumerate() {
            assert_eq!(a.weights, b.weights, "{} network layer {} weights", name, i);
            assert_eq!(a.biases, b.biases, "{} network layer {} biases", name, i);
        }
    }

    // And the same numbers out of it
    let probe = Array1::from_vec(vec![0.3, -0.7]);
    assert_eq!(
        agent.q_network.predict(probe.view()),
        loaded_agent.q_network.predict(probe.view())
    );

    fs::remove_file(path).ok();
}

#[test]
fn adam_state_survives_a_round_trip() {
    use crate::optimizer::Adam;
    use std::fs;

    let mut agent = DqnAgent::new(
        &[2, 8, 2],
        0.0,
        OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8)),
        1_000_000,
        false,
    );

    let experiences: Vec<Experience> = (0..8)
        .map(|i| Experience {
            state: array![i as f32 * 0.1, 1.0],
            action: i % 2,
            reward: 0.5,
            next_state: array![0.2, 0.4],
            done: true,
        })
        .collect();
    for _ in 0..50 {
        let batch: Vec<&Experience> = experiences.iter().collect();
        agent.train_on_batch(&batch, 0.99, 1e-3).unwrap();
    }

    let path = "test_agent_adam.bin";
    agent.save(path).unwrap();
    let mut loaded = DqnAgent::load(path).unwrap();
    fs::remove_file(path).ok();

    // One more identical step. Adam's moment estimates decide the step size, so if they
    // had not come across the two networks would separate here.
    let batch: Vec<&Experience> = experiences.iter().collect();
    agent.train_on_batch(&batch, 0.99, 1e-3).unwrap();
    loaded.train_on_batch(&batch, 0.99, 1e-3).unwrap();

    for (i, (a, b)) in agent
        .q_network
        .layers
        .iter()
        .zip(loaded.q_network.layers.iter())
        .enumerate()
    {
        for (x, y) in a.weights.iter().zip(b.weights.iter()) {
            assert!(
                (x - y).abs() < 1e-9,
                "layer {} diverged after one more step: {} vs {}",
                i,
                x,
                y
            );
        }
    }
}

#[test]
fn a_truncated_or_foreign_file_is_reported_rather_than_decoded() {
    use std::fs;

    let agent = DqnAgent::new(&[2, 4, 2], 0.1, OptimizerWrapper::SGD(SGD::new()), 100, false);
    let path = "test_agent_truncated.bin";
    agent.save(path).unwrap();

    let whole = fs::read(path).unwrap();
    assert!(whole.len() > 32);

    fs::write(path, &whole[..whole.len() / 2]).unwrap();
    assert!(DqnAgent::load(path).is_err(), "a truncated file loaded");

    // Right length, wrong magic
    let mut foreign = whole.clone();
    foreign[0] = b'X';
    fs::write(path, &foreign).unwrap();
    assert!(DqnAgent::load(path).is_err(), "a file with the wrong magic loaded");

    // Right magic, unknown version
    let mut future = whole.clone();
    future[4] = 99;
    fs::write(path, &future).unwrap();
    assert!(DqnAgent::load(path).is_err(), "an unknown format version loaded");

    fs::remove_file(path).ok();
}

#[test]
fn a_saved_network_does_not_carry_the_forward_pass_caches() {
    use crate::network::NeuralNetwork;
    use crate::activations::Activation;
    use ndarray::Array2;

    let mut net = NeuralNetwork::new(
        &[4, 128, 64, 2],
        &[Activation::Relu, Activation::Relu, Activation::Linear],
        OptimizerWrapper::SGD(SGD::new()),
    );

    let untouched = crate::serialization::encode(&net).unwrap().len();

    // A batch of 32 through this shape caches more floats than the network has weights
    let inputs = Array2::from_shape_fn((32, 4), |(i, j)| (i + j) as f32 * 0.01);
    let _ = net.forward_batch(inputs.view());

    let after_a_batch = crate::serialization::encode(&net).unwrap().len();
    assert_eq!(
        untouched, after_a_batch,
        "the file grew by {} bytes after a forward pass",
        after_a_batch as i64 - untouched as i64
    );
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
#[test]
fn the_reported_loss_is_the_mean_squared_td_error_before_the_update() {
    // The loss used to be measured after the update and averaged over every action
    // column, so it read num_actions times too small and described a network that no
    // longer existed. It is now the pre-update mean over the batch.
    let mut agent = DqnAgent::new(&[3, 8, 4], 0.0, OptimizerWrapper::SGD(SGD::new()), 1_000_000, false);

    let experiences = vec![
        Experience {
            state: array![1.0, 0.0, 0.0],
            action: 2,
            reward: 1.0,
            next_state: array![0.0, 1.0, 0.0],
            done: true,
        },
        Experience {
            state: array![0.0, 0.0, 1.0],
            action: 0,
            reward: -0.5,
            next_state: array![0.0, 1.0, 0.0],
            done: true,
        },
    ];
    let batch: Vec<&Experience> = experiences.iter().collect();

    // Both transitions terminate, so the target is the reward and needs no bootstrap
    let expected: f32 = experiences
        .iter()
        .map(|e| {
            let q = agent.q_network.predict(e.state.view());
            let td = e.reward - q[e.action];
            td * td
        })
        .sum::<f32>()
        / experiences.len() as f32;

    let reported = agent.train_on_batch(&batch, 0.99, 0.0).unwrap();
    assert!(
        (reported - expected).abs() < 1e-5,
        "reported {} against hand-computed {}",
        reported,
        expected
    );
}

#[test]
fn training_leaves_the_q_network_ready_for_another_batch() {
    // train_on_batch now backpropagates through the caches its own forward pass wrote
    // rather than forwarding again. Two batches in a row must still work.
    let mut agent = DqnAgent::new(&[3, 8, 4], 0.0, OptimizerWrapper::SGD(SGD::new()), 2, true);

    let experiences: Vec<Experience> = (0..6)
        .map(|i| Experience {
            state: array![i as f32 * 0.1, 1.0, -0.5],
            action: i % 4,
            reward: 0.5,
            next_state: array![0.2, 0.3, 0.4],
            done: i % 3 == 0,
        })
        .collect();

    for _ in 0..5 {
        let batch: Vec<&Experience> = experiences.iter().collect();
        let loss = agent.train_on_batch(&batch, 0.99, 0.01).unwrap();
        assert!(loss.is_finite(), "loss went non-finite: {}", loss);
    }
}
