//! Reproducibility and thread-safety of the agents.
//!
//! Two properties a game needs and `ThreadRng` prevented: an agent has to be movable
//! onto a worker thread, and a run has to be repeatable from a seed.

use ndarray::{array, Array1};

use crate::activations::Activation;
use crate::agent::DqnAgent;
use crate::algorithms::{A2CAgent, PPOAgent, SACAgent, TD3Agent};
use crate::layers::WeightInit;
use crate::optimizer::{OptimizerWrapper, SGD};
use crate::replay_buffer::{Experience, ReplayBuffer};
use crate::rng::seeded_rng;
use crate::types::ActionSpace;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn every_agent_can_cross_a_thread_boundary() {
    assert_send_sync::<DqnAgent>();
    assert_send_sync::<A2CAgent>();
    assert_send_sync::<PPOAgent>();
    assert_send_sync::<SACAgent>();
    assert_send_sync::<TD3Agent>();

    // The pieces a worker thread would carry alongside an agent
    assert_send_sync::<ReplayBuffer>();
    assert_send_sync::<crate::network::NeuralNetwork>();
}

#[test]
fn an_agent_actually_moves_onto_a_worker_thread() {
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new_seeded(&[4, 16, 2], 0.5, optimizer, 100, false, 7);

    let handle = std::thread::spawn(move || {
        let state = array![0.1, 0.2, 0.3, 0.4];
        let mut actions = Vec::new();
        for _ in 0..20 {
            actions.push(agent.act(state.view()).unwrap());
        }
        actions
    });

    let actions = handle.join().expect("the worker thread panicked");
    assert_eq!(actions.len(), 20);
}

#[test]
fn the_same_seed_gives_the_same_action_sequence() {
    let states: Vec<Array1<f32>> = (0..100)
        .map(|i| {
            let x = i as f32 * 0.01;
            array![x.sin(), x.cos(), x, -x]
        })
        .collect();

    let sequence = |seed: u64| -> Vec<usize> {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        // Epsilon 1.0 so every action comes from the generator, not the weights
        let mut agent = DqnAgent::new_seeded(&[4, 16, 3], 1.0, optimizer, 100, false, seed);
        states.iter().map(|s| agent.act(s.view()).unwrap()).collect()
    };

    assert_eq!(sequence(42), sequence(42));
    assert_ne!(sequence(42), sequence(43));
}

#[test]
fn set_seed_makes_the_algorithm_agents_repeat() {
    let state = array![0.1, -0.2, 0.3, 0.4];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = A2CAgent::new(4, 3, &[16], optimizer, 0.99, 5, 0.01, 0.5);

    // Reseeding the same agent isolates the generator. Two separately constructed
    // agents would also differ in their weights, which is a separate axis: seed the
    // weight initialization as well when a whole run has to reproduce.
    let mut sample = |seed: u64| -> Vec<usize> {
        agent.set_seed(seed);
        (0..50)
            .map(|_| agent.act(state.view()).unwrap().0)
            .collect()
    };

    let first = sample(11);
    let second = sample(11);
    let other = sample(12);

    assert_eq!(first, second);
    assert_ne!(first, other);
}

#[test]
fn replay_sampling_repeats_from_a_seed() {
    let mut buffer = ReplayBuffer::new(100);
    for i in 0..50 {
        buffer.add(Experience {
            state: array![i as f32, 0.0],
            action: i % 3,
            reward: i as f32,
            next_state: array![i as f32 + 1.0, 0.0],
            done: false,
        });
    }

    let rewards = |seed: u64| -> Vec<f32> {
        let mut rng = seeded_rng(seed);
        buffer.sample_with(16, &mut rng).iter().map(|e| e.reward).collect()
    };

    assert_eq!(rewards(5), rewards(5));
    assert_ne!(rewards(5), rewards(6));
}

#[test]
fn action_space_sampling_repeats_from_a_seed() {
    let spaces = [
        ActionSpace::Discrete { n: 6 },
        ActionSpace::MultiDiscrete { nvec: vec![3, 4, 5] },
    ];

    for space in &spaces {
        let mut first = seeded_rng(99);
        let mut second = seeded_rng(99);

        for _ in 0..20 {
            let a = space.sample_with(&mut first);
            let b = space.sample_with(&mut second);
            assert_eq!(format!("{:?}", a), format!("{:?}", b));
        }
    }
}

#[test]
fn weight_initialization_repeats_from_a_seed() {
    let init = WeightInit::HeUniform;

    let mut first = seeded_rng(3);
    let mut second = seeded_rng(3);
    let mut other = seeded_rng(4);

    let a = init.initialize_weights_with((8, 4), &mut first);
    let b = init.initialize_weights_with((8, 4), &mut second);
    let c = init.initialize_weights_with((8, 4), &mut other);

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn a_seeded_network_trains_identically_twice() {
    // Seeded weights plus seeded sampling is what makes a whole run reproducible
    let build = || {
        let mut rng = seeded_rng(21);
        let init = WeightInit::HeUniform;
        let mut network = crate::network::NeuralNetwork::new(
            &[3, 8, 2],
            &[Activation::Relu, Activation::Linear],
            OptimizerWrapper::SGD(SGD::new()),
        );
        for layer in network.layers.iter_mut() {
            let shape = (layer.weights.shape()[0], layer.weights.shape()[1]);
            layer.weights = init.initialize_weights_with(shape, &mut rng);
            layer.biases.fill(0.0);
        }
        network
    };

    let inputs = ndarray::Array2::from_shape_fn((4, 3), |(i, j)| (i * 3 + j) as f32 * 0.1);
    let targets = ndarray::Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f32 * 0.2);

    let mut first = build();
    let mut second = build();

    for _ in 0..20 {
        first.train_minibatch(inputs.view(), targets.view(), 0.01);
        second.train_minibatch(inputs.view(), targets.view(), 0.01);
    }

    assert_eq!(
        first.forward_batch(inputs.view()),
        second.forward_batch(inputs.view())
    );
}
