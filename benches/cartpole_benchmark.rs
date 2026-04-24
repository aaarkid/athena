//! Faster CartPole benchmark comparing DQN with other algorithms
//! 
//! CartPole is simpler and trains much faster than Mountain Car

use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::replay_buffer::{ReplayBuffer, Experience};
use ndarray::Array1;
use rand::Rng;
use std::time::Instant;

/// Simple CartPole environment
struct CartPole {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    done: bool,
    steps: usize,
}

impl CartPole {
    fn new() -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            done: false,
            steps: 0,
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        self.x = rng.gen_range(-0.05..0.05);
        self.x_dot = rng.gen_range(-0.05..0.05);
        self.theta = rng.gen_range(-0.05..0.05);
        self.theta_dot = rng.gen_range(-0.05..0.05);
        self.done = false;
        self.steps = 0;
        
        self.get_state()
    }
    
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        // CartPole dynamics
        let force = if action == 1 { 10.0 } else { -10.0 };
        let gravity = 9.8;
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let total_mass = mass_cart + mass_pole;
        let length = 0.5;
        let pole_mass_length = mass_pole * length;
        let dt = 0.02;
        
        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();
        
        let temp = (force + pole_mass_length * self.theta_dot.powi(2) * sin_theta) / total_mass;
        let theta_acc = (gravity * sin_theta - cos_theta * temp) 
            / (length * (4.0/3.0 - mass_pole * cos_theta.powi(2) / total_mass));
        let x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;
        
        // Update state
        self.x += self.x_dot * dt;
        self.x_dot += x_acc * dt;
        self.theta += self.theta_dot * dt;
        self.theta_dot += theta_acc * dt;
        self.steps += 1;
        
        // Check if done
        self.done = self.x.abs() > 2.4 || self.theta.abs() > 0.209 || self.steps >= 200;
        
        let reward = if self.done && self.steps < 200 { 0.0 } else { 1.0 };
        let state = self.get_state();
        
        (state, reward, self.done)
    }
    
    fn get_state(&self) -> Array1<f32> {
        Array1::from_vec(vec![self.x, self.x_dot, self.theta, self.theta_dot])
    }
}

/// Benchmark results
#[derive(Clone)]
struct BenchmarkResult {
    algorithm: String,
    episodes_to_solve: Option<usize>,
    final_avg_reward: f32,
    training_time_ms: u128,
    inference_time_us: u128,
}

/// Run DQN benchmark
fn benchmark_dqn(episodes: usize) -> BenchmarkResult {
    let start = Instant::now();
    let mut env = CartPole::new();
    
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(
        &[4, 64, 64, 2],
        1.0, // epsilon
        optimizer,
        100, // target update frequency
        true, // double DQN
    );
    
    let mut replay_buffer = ReplayBuffer::new(10000);
    let mut episode_rewards = Vec::new();
    let mut solved_episode = None;
    
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        for _ in 0..200 {
            let action = agent.act(state.view()).unwrap();
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;
            
            replay_buffer.add(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            if replay_buffer.len() >= 32 {
                let batch = replay_buffer.sample(32);
                let _ = agent.train_on_batch(&batch, 0.99, 0.001);
            }
            
            state = next_state;
            if done { break; }
        }
        
        episode_rewards.push(episode_reward);
        agent.epsilon = (agent.epsilon * 0.995).max(0.01);
        
        // Check if solved (average reward > 195 over 100 episodes)
        if episode >= 100 && solved_episode.is_none() {
            let avg_reward: f32 = episode_rewards[episode-99..=episode].iter().sum::<f32>() / 100.0;
            if avg_reward >= 195.0 {
                solved_episode = Some(episode);
            }
        }
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
    agent.epsilon = 0.0;
    let inference_start = Instant::now();
    let state = env.reset();
    for _ in 0..1000 {
        let _ = agent.act(state.view());
    }
    let inference_time = inference_start.elapsed() / 1000;
    
    let final_avg_reward = if episode_rewards.len() >= 100 {
        episode_rewards.iter().rev().take(100).sum::<f32>() / 100.0
    } else {
        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
    };
    
    BenchmarkResult {
        algorithm: "DQN".to_string(),
        episodes_to_solve: solved_episode,
        final_avg_reward,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
    }
}

/// Run DQN with different hyperparameters
fn benchmark_dqn_tuned(episodes: usize) -> BenchmarkResult {
    let start = Instant::now();
    let mut env = CartPole::new();
    
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(
        &[4, 128, 64, 2], // Larger network
        1.0, // epsilon
        optimizer,
        50, // More frequent target updates
        true, // double DQN
    );
    
    let mut replay_buffer = ReplayBuffer::new(50000);
    let mut episode_rewards = Vec::new();
    let mut solved_episode = None;
    
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        for _ in 0..200 {
            let action = agent.act(state.view()).unwrap();
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;
            
            replay_buffer.add(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            if replay_buffer.len() >= 64 {
                let batch = replay_buffer.sample(64);
                let _ = agent.train_on_batch(&batch, 0.99, 0.0005); // Lower learning rate
            }
            
            state = next_state;
            if done { break; }
        }
        
        episode_rewards.push(episode_reward);
        agent.epsilon = (agent.epsilon * 0.99).max(0.01); // Slower epsilon decay
        
        // Check if solved
        if episode >= 100 && solved_episode.is_none() {
            let avg_reward: f32 = episode_rewards[episode-99..=episode].iter().sum::<f32>() / 100.0;
            if avg_reward >= 195.0 {
                solved_episode = Some(episode);
            }
        }
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
    agent.epsilon = 0.0;
    let inference_start = Instant::now();
    let state = env.reset();
    for _ in 0..1000 {
        let _ = agent.act(state.view());
    }
    let inference_time = inference_start.elapsed() / 1000;
    
    let final_avg_reward = if episode_rewards.len() >= 100 {
        episode_rewards.iter().rev().take(100).sum::<f32>() / 100.0
    } else {
        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
    };
    
    BenchmarkResult {
        algorithm: "DQN-Tuned".to_string(),
        episodes_to_solve: solved_episode,
        final_avg_reward,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
    }
}

/// Run baseline random agent
fn benchmark_random(episodes: usize) -> BenchmarkResult {
    let start = Instant::now();
    let mut env = CartPole::new();
    let mut rng = rand::thread_rng();
    
    let mut episode_rewards = Vec::new();
    
    for _ in 0..episodes.min(100) { // Only run 100 episodes for random
        let _ = env.reset();
        let mut episode_reward = 0.0;
        
        for _ in 0..200 {
            let action = rng.gen_range(0..2);
            let (_, reward, done) = env.step(action);
            episode_reward += reward;
            
            if done { break; }
        }
        
        episode_rewards.push(episode_reward);
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
    let inference_start = Instant::now();
    let _state = env.reset();
    for _ in 0..1000 {
        let _ = rng.gen_range(0..2);
    }
    let inference_time = inference_start.elapsed() / 1000;
    
    let final_avg_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
    
    BenchmarkResult {
        algorithm: "Random".to_string(),
        episodes_to_solve: None,
        final_avg_reward,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
    }
}

fn main() {
    println!("CartPole Algorithm Comparison Benchmark");
    println!("======================================\n");
    
    let episodes = 500;
    
    println!("Running Random baseline...");
    let random_result = benchmark_random(episodes);
    
    println!("Running DQN...");
    let dqn_result = benchmark_dqn(episodes);
    
    println!("Running DQN-Tuned...");
    let dqn_tuned_result = benchmark_dqn_tuned(episodes);
    
    // Store results in a vector
    let results = vec![random_result, dqn_result, dqn_tuned_result];
    
    // Print results
    println!("\nResults:");
    println!("--------");
    
    for result in &results {
        println!("\n{} Algorithm:", result.algorithm);
        match result.episodes_to_solve {
            Some(ep) => println!("  Solved in {} episodes", ep),
            None => println!("  Not solved within {} episodes", episodes),
        }
        println!("  Final average reward: {:.2}", result.final_avg_reward);
        println!("  Training time: {:.2}s", result.training_time_ms as f64 / 1000.0);
        println!("  Inference time: {}μs", result.inference_time_us);
    }
    
    // Create comparison table
    println!("\n\nComparison Table:");
    println!("| Algorithm  | Episodes to Solve | Final Avg Reward | Training Time (s) | Speedup |");
    println!("|------------|-------------------|------------------|-------------------|---------|");
    
    let baseline_time = results[0].training_time_ms as f64; // Random is first
    
    for result in &results {
        let episodes_str = match result.episodes_to_solve {
            Some(ep) => ep.to_string(),
            None => "Not solved".to_string(),
        };
        let speedup = baseline_time / result.training_time_ms as f64;
        println!("| {:10} | {:17} | {:16.2} | {:17.3} | {:7.2}x |",
            result.algorithm,
            episodes_str,
            result.final_avg_reward,
            result.training_time_ms as f64 / 1000.0,
            speedup
        );
    }
    
    // Summary
    println!("\nSummary:");
    println!("--------");
    if results[1].episodes_to_solve.is_some() || results[2].episodes_to_solve.is_some() {
        println!("✓ DQN successfully learns to balance the CartPole");
        
        if let (Some(dqn_ep), Some(tuned_ep)) = (results[1].episodes_to_solve, results[2].episodes_to_solve) {
            if tuned_ep < dqn_ep {
                println!("✓ Hyperparameter tuning improves learning speed by {:.0}%", 
                        (1.0 - tuned_ep as f64 / dqn_ep as f64) * 100.0);
            }
        }
    } else {
        println!("✗ Neither DQN variant solved CartPole within {} episodes", episodes);
        println!("  Consider increasing episode limit or adjusting hyperparameters");
    }
}