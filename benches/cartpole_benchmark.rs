//! CartPole benchmark comparing DQN, PPO, and SAC
//! 
//! CartPole is simpler and trains much faster than Mountain Car

use athena::agent::DqnAgent;
use athena::algorithms::{PPOBuilder, PPORolloutBuffer, SACBuilder, SACExperience, A2CBuilder, A2CExperience, TD3Builder, TD3Experience};
use athena::optimizer::{OptimizerWrapper, SGD, Adam};
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
    
    // Continuous action version for SAC
    fn step_continuous(&mut self, action: &Array1<f32>) -> (Array1<f32>, f32, bool) {
        // Convert continuous action [-1, 1] to discrete
        let discrete_action = if action[0] > 0.0 { 1 } else { 0 };
        self.step(discrete_action)
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
        
        // Progress tracking
        if episode % 50 == 0 {
            println!("DQN Episode {}/{}", episode, episodes);
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

/// Run PPO benchmark
fn benchmark_ppo(episodes: usize) -> BenchmarkResult {
    let start = Instant::now();
    let mut env = CartPole::new();
    
    // Create PPO agent
    let optimizer = OptimizerWrapper::Adam(Adam::default(&[]));
    let mut agent = PPOBuilder::new(4, 2)
        .hidden_sizes(vec![64, 64])
        .optimizer(optimizer)
        .clip_param(0.2)
        .ppo_epochs(10)
        .build()
        .unwrap();
    
    let mut episode_rewards = Vec::new();
    let mut solved_episode = None;
    
    // PPO uses batch collection
    let rollout_size = 2048;
    let mut total_episodes = 0;
    
    while total_episodes < episodes {
        let mut rollout_buffer = PPORolloutBuffer::new();
        let mut rollout_steps = 0;
        
        while rollout_steps < rollout_size && total_episodes < episodes {
            let mut state = env.reset();
            let mut episode_reward = 0.0;
            
            for _ in 0..200 {
                let (action, log_prob, value) = agent.act(state.view()).unwrap();
                let (next_state, reward, done) = env.step(action);
                episode_reward += reward;
                
                rollout_buffer.add(state.clone(), action, reward, value, log_prob, done);
                rollout_steps += 1;
                
                state = next_state;
                if done { break; }
            }
            
            episode_rewards.push(episode_reward);
            total_episodes += 1;
            
            // Check if solved
            if total_episodes >= 100 && solved_episode.is_none() {
                let start_idx = total_episodes.saturating_sub(100);
                let avg_reward: f32 = episode_rewards[start_idx..total_episodes].iter().sum::<f32>() / 100.0;
                if avg_reward >= 195.0 {
                    solved_episode = Some(total_episodes);
                }
            }
            
            // Progress tracking
            if total_episodes % 50 == 0 {
                println!("PPO Episode {}/{}", total_episodes, episodes);
            }
        }
        
        // Train on collected rollout
        if !rollout_buffer.is_empty() {
            let last_value = agent.value.forward(env.get_state().view())[0];
            agent.compute_gae(&mut rollout_buffer, last_value);
            let _ = agent.update(&rollout_buffer, 3e-4);
        }
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
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
        algorithm: "PPO".to_string(),
        episodes_to_solve: solved_episode,
        final_avg_reward,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
    }
}

/// Run A2C benchmark
fn benchmark_a2c(episodes: usize) -> BenchmarkResult {
    let start = Instant::now();
    let mut env = CartPole::new();
    
    // Create A2C agent
    let optimizer = OptimizerWrapper::Adam(Adam::default(&[]));
    let mut agent = A2CBuilder::new(4, 2)  // 2 discrete actions
        .hidden_sizes(vec![64, 64])
        .optimizer(optimizer)
        .gamma(0.99)
        .n_steps(5)
        .entropy_coeff(0.01)
        .value_coeff(0.5)
        .build()
        .unwrap();
    
    let mut episode_rewards = Vec::new();
    let mut solved_episode = None;
    let mut experiences: Vec<A2CExperience> = Vec::new();
    
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut episode_experiences = Vec::new();
        
        for _ in 0..200 {
            let value = agent.get_value(state.view());
            let (action, log_prob) = agent.act(state.view()).unwrap();
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;
            
            episode_experiences.push(A2CExperience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
                log_prob,
                value,
            });
            
            state = next_state;
            if done { break; }
        }
        
        experiences.extend(episode_experiences);
        
        // Train when we have enough experiences
        if experiences.len() >= agent.n_steps * 4 {
            let _ = agent.train(&experiences, 3e-4);
            experiences.clear();
        }
        
        episode_rewards.push(episode_reward);
        
        // Check if solved (average reward > 190 over 100 episodes)
        if episode >= 100 && solved_episode.is_none() {
            let avg_reward: f32 = episode_rewards[episode-99..=episode].iter().sum::<f32>() / 100.0;
            if avg_reward >= 190.0 {
                solved_episode = Some(episode);
            }
        }
        
        // Progress tracking
        if episode % 50 == 0 {
            println!("A2C Episode {}/{}", episode, episodes);
        }
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
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
        algorithm: "A2C".to_string(),
        episodes_to_solve: solved_episode,
        final_avg_reward,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
    }
}

/// Run SAC benchmark
fn benchmark_sac(episodes: usize) -> BenchmarkResult {
    let start = Instant::now();
    let mut env = CartPole::new();
    
    // Create SAC agent
    let optimizer = OptimizerWrapper::Adam(Adam::default(&[]));
    let mut agent = SACBuilder::new(4, 1)  // 1 continuous action
        .hidden_sizes(vec![64, 64])
        .optimizer(optimizer)
        .gamma(0.99)
        .tau(0.005)
        .alpha(0.2)
        .auto_alpha(true)
        .build()
        .unwrap();
    
    let mut episode_rewards = Vec::new();
    let mut solved_episode = None;
    let mut sac_experiences: Vec<SACExperience> = Vec::new();
    
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        for _ in 0..200 {
            let action = agent.act(state.view(), false).unwrap();
            let (next_state, reward, done) = env.step_continuous(&action);
            episode_reward += reward;
            
            sac_experiences.push(SACExperience {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train when we have enough experiences
            if sac_experiences.len() >= 256 {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let mut batch = sac_experiences.clone();
                batch.shuffle(&mut rng);
                batch.truncate(256);
                
                let _ = agent.update(&batch, 3e-4);
                
                // Keep buffer size manageable
                if sac_experiences.len() > 10000 {
                    sac_experiences.drain(0..5000);
                }
            }
            
            state = next_state;
            if done { break; }
        }
        
        episode_rewards.push(episode_reward);
        
        // Check if solved
        if episode >= 100 && solved_episode.is_none() {
            let avg_reward: f32 = episode_rewards[episode-99..=episode].iter().sum::<f32>() / 100.0;
            if avg_reward >= 195.0 {
                solved_episode = Some(episode);
            }
        }
        
        // Progress tracking
        if episode % 50 == 0 {
            println!("SAC Episode {}/{}", episode, episodes);
        }
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
    let inference_start = Instant::now();
    let state = env.reset();
    for _ in 0..1000 {
        let _ = agent.act(state.view(), true);
    }
    let inference_time = inference_start.elapsed() / 1000;
    
    let final_avg_reward = if episode_rewards.len() >= 100 {
        episode_rewards.iter().rev().take(100).sum::<f32>() / 100.0
    } else {
        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
    };
    
    BenchmarkResult {
        algorithm: "SAC".to_string(),
        episodes_to_solve: solved_episode,
        final_avg_reward,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
    }
}

/// Run TD3 benchmark
fn benchmark_td3(episodes: usize) -> BenchmarkResult {
    let start = Instant::now();
    let mut env = CartPole::new();
    
    // Create TD3 agent
    let optimizer = OptimizerWrapper::Adam(Adam::default(&[]));
    let mut agent = TD3Builder::new(4, 1)  // 1 continuous action
        .hidden_sizes(vec![64, 64])
        .optimizer(optimizer)
        .gamma(0.99)
        .tau(0.005)
        .policy_delay(2)
        .action_bounds(-1.0, 1.0)
        .noise_params(0.2, 0.5, 0.1)
        .build()
        .unwrap();
    
    let mut episode_rewards = Vec::new();
    let mut solved_episode = None;
    let mut td3_experiences: Vec<TD3Experience> = Vec::new();
    
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        for _ in 0..200 {
            let action = agent.act(state.view(), false).unwrap();
            let (next_state, reward, done) = env.step_continuous(&action);
            episode_reward += reward;
            
            td3_experiences.push(TD3Experience {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train when we have enough experiences
            if td3_experiences.len() >= 256 {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let mut batch = td3_experiences.clone();
                batch.shuffle(&mut rng);
                batch.truncate(256);
                
                let _ = agent.update(&batch, 3e-4, 3e-4);
                
                // Keep buffer size manageable
                if td3_experiences.len() > 10000 {
                    td3_experiences.drain(0..5000);
                }
            }
            
            state = next_state;
            if done { break; }
        }
        
        episode_rewards.push(episode_reward);
        
        // Check if solved
        if episode >= 100 && solved_episode.is_none() {
            let avg_reward: f32 = episode_rewards[episode-99..=episode].iter().sum::<f32>() / 100.0;
            if avg_reward >= 190.0 {
                solved_episode = Some(episode);
            }
        }
        
        // Progress tracking
        if episode % 50 == 0 {
            println!("TD3 Episode {}/{}", episode, episodes);
        }
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
    let inference_start = Instant::now();
    let state = env.reset();
    for _ in 0..1000 {
        let _ = agent.act(state.view(), true);
    }
    let inference_time = inference_start.elapsed() / 1000;
    
    let final_avg_reward = if episode_rewards.len() >= 100 {
        episode_rewards.iter().rev().take(100).sum::<f32>() / 100.0
    } else {
        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
    };
    
    BenchmarkResult {
        algorithm: "TD3".to_string(),
        episodes_to_solve: solved_episode,
        final_avg_reward,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
    }
}

fn main() {
    println!("CartPole Algorithm Comparison Benchmark");
    println!("======================================\n");
    
    let episodes = 500;
    
    println!("Running DQN...");
    let dqn_result = benchmark_dqn(episodes);
    
    println!("\nRunning PPO...");
    let ppo_result = benchmark_ppo(episodes);
    
    println!("\nRunning SAC...");
    let sac_result = benchmark_sac(episodes);
    
    println!("\nRunning A2C...");
    let a2c_result = benchmark_a2c(episodes);
    
    println!("\nRunning TD3...");
    let td3_result = benchmark_td3(episodes);
    
    // Store results in a vector
    let results = vec![dqn_result, a2c_result, ppo_result, sac_result, td3_result];
    
    // Print results
    println!("\n\nResults:");
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
    println!("| Algorithm | Episodes to Solve | Final Avg Reward | Training Time (s) | Inference Time (μs) |");
    println!("|-----------|-------------------|------------------|-------------------|---------------------|");
    
    for result in &results {
        let episodes_str = match result.episodes_to_solve {
            Some(ep) => ep.to_string(),
            None => "Not solved".to_string(),
        };
        println!("| {:9} | {:17} | {:16.2} | {:17.3} | {:19} |",
            result.algorithm,
            episodes_str,
            result.final_avg_reward,
            result.training_time_ms as f64 / 1000.0,
            result.inference_time_us
        );
    }
    
    // Summary
    println!("\n\nSummary:");
    println!("--------");
    
    // Check which algorithms solved CartPole
    let solved_algorithms: Vec<_> = results.iter()
        .filter(|r| r.episodes_to_solve.is_some())
        .collect();
    
    if !solved_algorithms.is_empty() {
        println!("✓ The following algorithms successfully solved CartPole:");
        for result in &solved_algorithms {
            println!("  - {} in {} episodes", 
                     result.algorithm, 
                     result.episodes_to_solve.unwrap());
        }
        
        // Find the fastest solver
        if let Some(fastest) = solved_algorithms.iter()
            .min_by_key(|r| r.episodes_to_solve.unwrap()) {
            println!("\n✓ Fastest solver: {} ({} episodes)", 
                     fastest.algorithm, 
                     fastest.episodes_to_solve.unwrap());
        }
    } else {
        println!("✗ No algorithms solved CartPole within {} episodes", episodes);
        println!("  Consider increasing episode limit or adjusting hyperparameters");
    }
    
    // Performance comparison
    if results.len() > 1 {
        let fastest_inference = results.iter()
            .min_by_key(|r| r.inference_time_us)
            .unwrap();
        println!("\n✓ Fastest inference: {} ({}μs per action)", 
                 fastest_inference.algorithm, 
                 fastest_inference.inference_time_us);
    }
}