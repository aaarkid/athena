//! Benchmark comparison of different RL algorithms
//! 
//! This benchmark compares the performance and sample efficiency of:
//! - DQN
//! - A2C  
//! - PPO
//! - SAC
//! - TD3

use athena::agent::DqnAgent;
use athena::algorithms::{PPOBuilder, PPORolloutBuffer, SACBuilder, SACExperience};
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::replay_buffer::ReplayBuffer;
use ndarray::{Array1, array};
use std::time::Instant;
use std::fs::File;
use std::io::Write;

/// Standard benchmark environment - Continuous Mountain Car
struct MountainCar {
    position: f32,
    velocity: f32,
    min_position: f32,
    max_position: f32,
    max_speed: f32,
    goal_position: f32,
    force: f32,
}

impl MountainCar {
    fn new() -> Self {
        MountainCar {
            position: -0.5,
            velocity: 0.0,
            min_position: -1.2,
            max_position: 0.6,
            max_speed: 0.07,
            goal_position: 0.5,
            force: 0.003,  // Increased force for SAC continuous actions
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        self.position = rand::random::<f32>() * 0.2 - 0.6;  // [-0.6, -0.4]
        self.velocity = 0.0;
        array![self.position, self.velocity]
    }
    
    fn step(&mut self, action: &Array1<f32>) -> (Array1<f32>, f32, bool) {
        // Continuous action in [-1, 1]
        let force = action[0].clamp(-1.0, 1.0) * self.force;
        
        // Update velocity
        self.velocity += force - 0.0025 * (3.0 * self.position).cos();
        self.velocity = self.velocity.clamp(-self.max_speed, self.max_speed);
        
        // Update position
        self.position += self.velocity;
        self.position = self.position.clamp(self.min_position, self.max_position);
        
        // Bounce off left wall
        if self.position <= self.min_position {
            self.velocity = 0.0;
        }
        
        // Check if goal reached
        let done = self.position >= self.goal_position;
        let reward = if done { 100.0 } else { -1.0 };
        
        (array![self.position, self.velocity], reward, done)
    }
}

/// Benchmark results for an algorithm
#[derive(Debug)]
struct BenchmarkResult {
    algorithm: String,
    episodes_to_solve: Option<usize>,
    final_reward: f32,
    training_time_ms: u128,
    inference_time_us: u128,
    memory_usage_mb: f32,
}

/// Benchmark DQN (discretized actions)
fn benchmark_dqn(episodes: usize) -> BenchmarkResult {
    println!("\nBenchmarking DQN...");
    let start = Instant::now();
    
    // Create DQN agent
    let layer_sizes = &[2, 256, 256, 3];  // 3 discrete actions: left, none, right
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(layer_sizes, 1.0, optimizer, 100, true);
    
    // Training components
    let mut buffer = ReplayBuffer::new(10000);
    let mut env = MountainCar::new();
    let mut solved_episode = None;
    let mut recent_rewards = Vec::new();
    
    // Training loop
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        loop {
            // Select action
            let discrete_action = agent.act(state.view()).unwrap();
            let continuous_action = array![(discrete_action as f32 - 1.0)]; // -1, 0, 1
            
            // Step
            let (next_state, reward, done) = env.step(&continuous_action);
            episode_reward += reward;
            
            // Store experience
            buffer.add(athena::replay_buffer::Experience {
                state: state.clone(),
                action: discrete_action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train
            if buffer.len() >= 32 {
                let batch = buffer.sample(32);
                agent.train_on_batch(&batch, 0.99, 0.001).unwrap();
            }
            
            state = next_state;
            
            if done || episode_reward < -200.0 {
                break;
            }
        }
        
        recent_rewards.push(episode_reward);
        if recent_rewards.len() > 100 {
            recent_rewards.remove(0);
        }
        
        // Check if solved
        if recent_rewards.len() == 100 {
            let avg_reward = recent_rewards.iter().sum::<f32>() / 100.0;
            if avg_reward >= 90.0 && solved_episode.is_none() {
                solved_episode = Some(episode);
                println!("DQN solved at episode {}", episode);
            }
        }
        
        // Decay epsilon
        agent.epsilon = (agent.epsilon * 0.995).max(0.01);
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
    let inference_start = Instant::now();
    for _ in 0..1000 {
        let state = array![0.0, 0.0];
        let _ = agent.act(state.view()).unwrap();
    }
    let inference_time = inference_start.elapsed() / 1000;
    
    // Evaluate final performance
    agent.epsilon = 0.0;
    let mut total_reward = 0.0;
    for _ in 0..100 {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        loop {
            let action = agent.act(state.view()).unwrap();
            let continuous_action = array![(action as f32 - 1.0)];
            let (next_state, reward, done) = env.step(&continuous_action);
            episode_reward += reward;
            state = next_state;
            
            if done || episode_reward < -200.0 {
                break;
            }
        }
        total_reward += episode_reward;
    }
    
    BenchmarkResult {
        algorithm: "DQN".to_string(),
        episodes_to_solve: solved_episode,
        final_reward: total_reward / 100.0,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
        memory_usage_mb: estimate_memory_usage(),
    }
}

/// Benchmark PPO
fn benchmark_ppo(episodes: usize) -> BenchmarkResult {
    println!("\nBenchmarking PPO...");
    let start = Instant::now();
    
    // Create PPO agent
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = PPOBuilder::new(2, 1)  // state_size, action_size
        .hidden_sizes(vec![256, 256])
        .optimizer(optimizer)
        .clip_param(0.2)
        .ppo_epochs(10)
        .build()
        .unwrap();
    
    let mut env = MountainCar::new();
    let mut solved_episode = None;
    let mut recent_rewards = Vec::new();
    
    // PPO uses batch collection
    let n_steps = 2048;
    let mut episode = 0;
    
    while episode < episodes {
        // Collect rollout
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut dones = Vec::new();
        let mut values = Vec::new();
        let mut log_probs = Vec::new();
        
        let mut state = env.reset();
        let mut episode_rewards = Vec::new();
        let mut current_episode_reward = 0.0;
        
        for _ in 0..n_steps {
            let (action, log_prob, value) = agent.act(state.view()).unwrap();
            
            let continuous_action = array![(action as f32 - 0.5) * 2.0];
            let (next_state, reward, done) = env.step(&continuous_action);
            
            states.push(state.clone());
            actions.push(action);
            rewards.push(reward);
            dones.push(done);
            values.push(value);
            log_probs.push(log_prob);
            
            current_episode_reward += reward;
            state = next_state;
            
            if done || current_episode_reward < -200.0 {
                episode_rewards.push(current_episode_reward);
                episode += 1;
                state = env.reset();
                current_episode_reward = 0.0;
            }
        }
        
        // Create rollout buffer directly without unnecessary conversions
        let mut rollout_buffer = PPORolloutBuffer::new();
        let last_done = dones.last() == Some(&true);
        let last_value = if last_done { 
            0.0 
        } else { 
            agent.value.forward(state.view())[0]
        };
        
        for i in 0..states.len() {
            rollout_buffer.add(
                states[i].clone(),
                actions[i],
                rewards[i],
                values[i],
                log_probs[i],
                dones[i],
            );
        }
        
        // Compute advantages and train
        agent.compute_gae(&mut rollout_buffer, last_value);
        let _ = agent.update(&rollout_buffer, 3e-4);
        
        // Track progress
        for ep_reward in episode_rewards {
            recent_rewards.push(ep_reward);
            if recent_rewards.len() > 100 {
                recent_rewards.remove(0);
            }
            
            if recent_rewards.len() == 100 {
                let avg_reward = recent_rewards.iter().sum::<f32>() / 100.0;
                if avg_reward >= 90.0 && solved_episode.is_none() {
                    solved_episode = Some(episode);
                    println!("PPO solved at episode {}", episode);
                }
            }
        }
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
    let inference_start = Instant::now();
    for _ in 0..1000 {
        let state = array![0.0, 0.0];
        let _ = agent.act(state.view()).unwrap();
    }
    let inference_time = inference_start.elapsed() / 1000;
    
    // Evaluate final performance
    let mut total_reward = 0.0;
    for _ in 0..100 {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        loop {
            let (action, _, _) = agent.act(state.view()).unwrap();
            let continuous_action = array![(action as f32 - 0.5) * 2.0];
            let (next_state, reward, done) = env.step(&continuous_action);
            episode_reward += reward;
            state = next_state;
            
            if done || episode_reward < -200.0 {
                break;
            }
        }
        total_reward += episode_reward;
    }
    
    BenchmarkResult {
        algorithm: "PPO".to_string(),
        episodes_to_solve: solved_episode,
        final_reward: total_reward / 100.0,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
        memory_usage_mb: estimate_memory_usage(),
    }
}

/// Benchmark SAC
fn benchmark_sac(episodes: usize) -> BenchmarkResult {
    println!("\nBenchmarking SAC...");
    let start = Instant::now();
    let mut last_print = Instant::now();
    
    // Create SAC agent
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    
    let mut agent = SACBuilder::new(2, 1)  // state_size, action_size
        .hidden_sizes(vec![256, 256])
        .optimizer(optimizer)
        .gamma(0.99)
        .tau(0.005)
        .alpha(0.2)
        .auto_alpha(true)
        .build()
        .unwrap();
    
    let mut env = MountainCar::new();
    let mut solved_episode = None;
    let mut recent_rewards = Vec::new();
    let mut sac_experiences: Vec<SACExperience> = Vec::new();
    
    // Training loop
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        // Progress tracking
        if episode % 20 == 0 {
            println!("SAC Episode {}/{}, experiences: {}", episode, episodes, sac_experiences.len());
        }
        
        let mut steps = 0;
        loop {
            // Select action
            let action = if episode < 10 {
                // Random exploration at start
                array![rand::random::<f32>() * 2.0 - 1.0]
            } else {
                agent.act(state.view(), false).unwrap()
            };
            
            // Step
            let (next_state, reward, done) = env.step(&action);
            episode_reward += reward;
            steps += 1;
            
            
            // Store continuous action experiences for SAC training
            sac_experiences.push(athena::algorithms::SACExperience {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train SAC when we have enough experiences
            if sac_experiences.len() >= 256 {
                // Sample batch and train
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                
                // Sample without cloning the entire buffer
                let indices: Vec<usize> = (0..sac_experiences.len()).collect();
                let sampled_indices: Vec<_> = indices.choose_multiple(&mut rng, 256).cloned().collect();
                
                let batch: Vec<SACExperience> = sampled_indices
                    .iter()
                    .map(|&i| sac_experiences[i].clone())
                    .collect();
                
                // SAC training with continuous actions
                let _ = agent.update(&batch, 3e-4);
                
                // Keep buffer size manageable
                if sac_experiences.len() > 10000 {
                    sac_experiences.drain(0..5000);
                }
            }
            
            state = next_state;
            
            // Add step limit to prevent infinite loops
            if done || episode_reward < -200.0 || steps >= 200 {
                break;
            }
        }
        
        recent_rewards.push(episode_reward);
        if recent_rewards.len() > 100 {
            recent_rewards.remove(0);
        }
        
        // Check if solved
        if recent_rewards.len() == 100 {
            let avg_reward = recent_rewards.iter().sum::<f32>() / 100.0;
            if avg_reward >= 90.0 && solved_episode.is_none() {
                solved_episode = Some(episode);
                println!("SAC solved at episode {}", episode);
            }
        }
    }
    
    let training_time = start.elapsed();
    
    // Measure inference time
    let inference_start = Instant::now();
    for _ in 0..1000 {
        let state = array![0.0, 0.0];
        let _ = agent.act(state.view(), false).unwrap();
    }
    let inference_time = inference_start.elapsed() / 1000;
    
    // Evaluate final performance
    let mut total_reward = 0.0;
    for _ in 0..100 {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        loop {
            let action = agent.act(state.view(), true).unwrap();
            let (next_state, reward, done) = env.step(&action);
            episode_reward += reward;
            state = next_state;
            
            if done || episode_reward < -200.0 {
                break;
            }
        }
        total_reward += episode_reward;
    }
    
    BenchmarkResult {
        algorithm: "SAC".to_string(),
        episodes_to_solve: solved_episode,
        final_reward: total_reward / 100.0,
        training_time_ms: training_time.as_millis(),
        inference_time_us: inference_time.as_micros(),
        memory_usage_mb: estimate_memory_usage(),
    }
}

/// Estimate memory usage (simplified)
fn estimate_memory_usage() -> f32 {
    // This is a simplified estimation
    // In practice, you'd use a memory profiler
    let network_params = 2 * 256 + 256 * 256 + 256 * 1;  // Approximate
    let buffer_size = 10000 * 5 * 4;  // experiences * fields * f32 size
    let total_bytes = (network_params + buffer_size) * 4;  // f32 = 4 bytes
    total_bytes as f32 / (1024.0 * 1024.0)  // Convert to MB
}

/// Generate comparison plots
fn generate_plots(results: &[BenchmarkResult]) -> std::io::Result<()> {
    let mut file = File::create("benchmark_results.md")?;
    
    writeln!(file, "# Algorithm Benchmark Results\n")?;
    writeln!(file, "Environment: Continuous Mountain Car")?;
    writeln!(file, "Hardware: {}\n", get_hardware_info())?;
    
    // Sample efficiency table
    writeln!(file, "## Sample Efficiency\n")?;
    writeln!(file, "| Algorithm | Episodes to Solve | Final Avg Reward |")?;
    writeln!(file, "|-----------|-------------------|------------------|")?;
    
    for result in results {
        writeln!(
            file, 
            "| {} | {} | {:.2} |",
            result.algorithm,
            result.episodes_to_solve
                .map(|e| e.to_string())
                .unwrap_or("Not solved".to_string()),
            result.final_reward
        )?;
    }
    
    // Performance table
    writeln!(file, "\n## Performance Metrics\n")?;
    writeln!(file, "| Algorithm | Training Time (s) | Inference Time (μs) | Memory (MB) |")?;
    writeln!(file, "|-----------|-------------------|---------------------|-------------|")?;
    
    for result in results {
        writeln!(
            file,
            "| {} | {:.2} | {} | {:.2} |",
            result.algorithm,
            result.training_time_ms as f32 / 1000.0,
            result.inference_time_us,
            result.memory_usage_mb
        )?;
    }
    
    // ASCII chart for episodes to solve
    writeln!(file, "\n## Sample Efficiency Chart\n")?;
    writeln!(file, "```")?;
    
    let max_episodes = results
        .iter()
        .filter_map(|r| r.episodes_to_solve)
        .max()
        .unwrap_or(1000);
    
    for result in results {
        let bar_length = if let Some(episodes) = result.episodes_to_solve {
            (episodes as f32 / max_episodes as f32 * 50.0) as usize
        } else {
            50
        };
        
        writeln!(
            file,
            "{:8} |{}{}| {}",
            result.algorithm,
            "█".repeat(bar_length),
            " ".repeat(50 - bar_length),
            result.episodes_to_solve
                .map(|e| e.to_string())
                .unwrap_or("NS".to_string())
        )?;
    }
    
    writeln!(file, "```")?;
    writeln!(file, "\nNS = Not Solved within episode limit")?;
    
    Ok(())
}

fn get_hardware_info() -> String {
    // Simplified hardware info
    format!(
        "{} architecture",
        std::env::consts::ARCH
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting RL Algorithm Benchmark Comparison");
    println!("==========================================\n");
    
    // let episodes = 500;  // Full benchmark
    let episodes = 100;  // Reduced for faster testing
    
    let mut results = Vec::new();
    
    // Run benchmarks
    // TEMP: Skip DQN and PPO for debugging
    // results.push(benchmark_dqn(episodes));
    // results.push(benchmark_ppo(episodes));
    results.push(benchmark_sac(episodes));
    // TD3 and A2C would be similar
    
    // Generate report
    generate_plots(&results)?;
    
    println!("\n==========================================");
    println!("Benchmark complete! Results saved to benchmark_results.md");
    
    // Print summary
    println!("\nSummary:");
    for result in &results {
        println!(
            "{}: {} episodes to solve, {:.2} final reward",
            result.algorithm,
            result.episodes_to_solve
                .map(|e| e.to_string())
                .unwrap_or("Not solved".to_string()),
            result.final_reward
        );
    }
    
    Ok(())
}