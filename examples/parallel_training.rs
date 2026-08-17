//! Splitting training across CPU cores with rayon.
//!
//! Run with `cargo run --release --example parallel_training`.
//!
//! Two things are parallel here: collecting episodes from several environments at once,
//! and computing the gradients for one large batch. Both read the network through
//! `predict`, which takes `&self`, so there is one set of weights and no per-thread copy.
//!
//! The batch is 256 rows. Below roughly a hundred, rayon's handoff costs more than the
//! work: the benchmark at the end prints the crossover on this machine.

use athena::network::NeuralNetwork;
use athena::activations::Activation;
use athena::optimizer::OptimizerWrapper;
use athena::parallel::{ParallelNetwork, ParallelGradients, ParallelReplayBuffer, ParallelAugmentation};
use athena::replay_buffer::Experience;
use athena::metrics::MetricsTracker;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Array4};
use std::time::Instant;

/// Simple CartPole-like environment for demonstration
struct SimpleEnv {
    state: Array1<f32>,
    steps: usize,
}

impl SimpleEnv {
    fn new() -> Self {
        SimpleEnv {
            state: Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            steps: 0,
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        self.state = Array1::from_vec(vec![
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
        ]);
        self.steps = 0;
        
        self.state.clone()
    }
    
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Simple dynamics
        let force = if action == 1 { 1.0 } else { -1.0 };
        
        self.state[0] += self.state[1] * 0.02;
        self.state[1] += force * 0.1 + rng.gen_range(-0.01..0.01);
        self.state[2] += self.state[3] * 0.02;
        self.state[3] += self.state[2] * 0.1 + rng.gen_range(-0.01..0.01);
        
        self.steps += 1;
        
        // Reward for staying upright
        let reward = 1.0;
        let done = self.steps >= 200 || self.state[0].abs() > 2.4 || self.state[2].abs() > 0.2;
        
        (self.state.clone(), reward, done)
    }
}

fn main() {
    println!("Parallel Training Example");
    println!("========================\n");
    
    // Set number of threads
    let num_threads = num_cpus::get();
    println!("Using {} CPU threads for parallel computation\n", num_threads);
    
    // Create Q-network
    let mut q_network = NeuralNetwork::new(
        &[4, 128, 128, 2],
        &[Activation::Relu, Activation::Relu, Activation::Linear],
        OptimizerWrapper::SGD(athena::optimizer::SGD::new())
    );
    
    // Create parallel replay buffer
    let mut replay_buffer = ParallelReplayBuffer::new(10000);
    
    // Training parameters
    let num_episodes = 100;
    let batch_size = 256;
    let gamma = 0.99;
    let epsilon_start = 1.0;
    let epsilon_end = 0.01;
    let epsilon_decay = 0.995;
    let mut epsilon = epsilon_start;
    
    // Create multiple environments for parallel data collection
    let num_envs = num_threads;
    let mut envs: Vec<SimpleEnv> = (0..num_envs).map(|_| SimpleEnv::new()).collect();
    
    // Metrics tracking
    let mut metrics = MetricsTracker::new(3, 1000);
    
    println!("Starting parallel training with {} environments...\n", num_envs);

    for episode in 0..num_episodes {
        let episode_start = Instant::now();

        // Collect one episode per environment, across threads. Every thread reads the
        // same network: predict takes &self and writes no caches, so there is nothing to
        // clone and nothing to keep in sync.
        let network_for_workers = &q_network;
        let experiences: Vec<Vec<Experience>> = envs.par_iter_mut()
            .map(|env| {
                let q_network = network_for_workers;
                let mut episode_experiences = Vec::new();
                let mut state = env.reset();
                let mut _total_reward = 0.0;
                
                loop {
                    // Epsilon-greedy action selection
                    let action = if rand::random::<f32>() < epsilon {
                        rand::random::<usize>() % 2
                    } else {
                        let q_values = q_network.predict(state.view());
                        let mut max_idx = 0;
                        let mut max_val = q_values[0];
                        for (i, &val) in q_values.iter().enumerate().skip(1) {
                            if val > max_val {
                                max_val = val;
                                max_idx = i;
                            }
                        }
                        max_idx
                    };
                    
                    let (next_state, reward, done) = env.step(action);
                    _total_reward += reward;
                    
                    episode_experiences.push(Experience {
                        state: state.clone(),
                        action,
                        reward,
                        next_state: next_state.clone(),
                        done,
                    });
                    
                    if done {
                        break;
                    }
                    
                    state = next_state;
                }
                
                episode_experiences
            })
            .collect();
        
        // Add all experiences to replay buffer
        let total_steps: usize = experiences.iter().map(|e| e.len()).sum();
        for env_experiences in experiences {
            for exp in env_experiences {
                replay_buffer.add(exp);
            }
        }
        
        // Parallel batch training
        if replay_buffer.buffer.len() >= batch_size {
            let train_start = Instant::now();
            
            // Sample batch in parallel
            let batch = replay_buffer.sample_parallel(batch_size);
            
            // Prepare batch data
            let mut states = Array2::zeros((batch_size, 4));
            let mut actions = Vec::with_capacity(batch_size);
            let mut rewards = Vec::with_capacity(batch_size);
            let mut next_states = Array2::zeros((batch_size, 4));
            let mut dones = Vec::with_capacity(batch_size);
            
            for (i, exp) in batch.iter().enumerate() {
                states.row_mut(i).assign(&exp.state);
                actions.push(exp.action);
                rewards.push(exp.reward);
                next_states.row_mut(i).assign(&exp.next_state);
                dones.push(exp.done);
            }
            
            // Both passes are inference, so both go through the parallel forward path
            let parallel_network = ParallelNetwork::from_network(&q_network, num_threads);
            let next_q_values = parallel_network.forward_batch_parallel(next_states.view());
            let current_q_values = parallel_network.forward_batch_parallel(states.view());

            // The target matches the prediction on every action except the one taken,
            // so only that column carries an error
            let mut targets = current_q_values.clone();
            for i in 0..batch_size {
                let max_next_q = next_q_values.row(i).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let target_value = if dones[i] {
                    rewards[i]
                } else {
                    rewards[i] + gamma * max_next_q
                };
                
                targets[[i, actions[i]]] = target_value;
            }
            
            // Parallel gradient computation
            let (weight_grads, bias_grads) = ParallelGradients::compute_batch_gradients(
                &q_network,
                states.view(),
                targets.view()
            );
            
            // Apply them. The old version built an updated_network and then dropped it,
            // so this loop never actually trained anything.
            let pairs: Vec<_> = weight_grads.into_iter().zip(bias_grads).collect();
            q_network.apply_gradients(pairs, 0.001);
            
            let train_time = train_start.elapsed();
            
            if episode % 10 == 0 {
                println!("Episode {}: {} steps collected, batch training took {:?}",
                        episode, total_steps, train_time);
            }
        }
        
        // Decay epsilon
        epsilon = (epsilon * epsilon_decay).max(epsilon_end);
        
        let _episode_time = episode_start.elapsed();
        // Track episode completion
        metrics.start_episode();
        metrics.step(episode as f32);  // Just track something for now
        metrics.end_episode();
    }
    
    println!("\nTraining complete!");
    
    // Benchmark: Compare parallel vs sequential processing
    println!("\nBenchmarking parallel vs sequential processing...");
    benchmark_parallel_vs_sequential(&q_network, num_threads);
}

fn benchmark_parallel_vs_sequential(network: &NeuralNetwork, num_threads: usize) {
    let parallel_network = ParallelNetwork::from_network(network, num_threads);
    let batch_sizes = vec![32, 64, 128, 256, 512, 2048];

    println!("\nBatch Size | Sequential Time | Parallel Time | Speedup");
    println!("-----------|-----------------|---------------|--------");
    
    for &batch_size in &batch_sizes {
        let input = Array2::ones((batch_size, 4));
        
        // Sequential timing, through the same cache-free path so the comparison is
        // between one thread and several rather than between two different code paths
        let seq_start = Instant::now();
        let _seq_outputs = network.predict_batch(input.view());
        let seq_time = seq_start.elapsed();
        
        // Parallel timing
        let par_start = Instant::now();
        let _par_outputs = parallel_network.forward_batch_parallel(input.view());
        let par_time = par_start.elapsed();
        
        let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64();
        
        println!("{:10} | {:15.3?} | {:13.3?} | {:.2}x",
                batch_size, seq_time, par_time, speedup);
    }
    
    // Test parallel data augmentation
    println!("\nTesting parallel image augmentation...");
    let images = Array4::ones((16, 3, 32, 32));
    
    let aug_start = Instant::now();
    let _augmented = ParallelAugmentation::augment_batch(images.view());
    let aug_time = aug_start.elapsed();
    
    println!("Augmented 16 images in {:?}", aug_time);

    println!("\nA speedup below 1.0 means rayon's handoff cost more than the work.");
}