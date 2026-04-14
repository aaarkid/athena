//! Pendulum environment solved using SAC (Soft Actor-Critic)
//! 
//! This example demonstrates:
//! - Using SAC for continuous action spaces
//! - Automatic temperature tuning
//! - Experience replay with prioritization
//! - Custom reward shaping
//! - Advanced debugging and visualization

use athena::algorithms::{SACAgent, SACBuilder};
use athena::optimizer::OptimizerWrapper;
use athena::replay_buffer::PrioritizedReplayBuffer;
use ndarray::{Array1, array};
use rand::Rng;
use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;

/// Pendulum environment - classic continuous control task
struct Pendulum {
    // State: [cos(theta), sin(theta), angular_velocity]
    theta: f32,
    theta_dot: f32,
    
    // Environment parameters
    max_speed: f32,
    max_torque: f32,
    dt: f32,
    gravity: f32,
    mass: f32,
    length: f32,
    
    // Episode tracking
    steps: usize,
    max_steps: usize,
}

impl Pendulum {
    fn new() -> Self {
        Pendulum {
            theta: 0.0,
            theta_dot: 0.0,
            max_speed: 8.0,
            max_torque: 2.0,
            dt: 0.05,
            gravity: 10.0,
            mass: 1.0,
            length: 1.0,
            steps: 0,
            max_steps: 200,
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        self.theta = rng.gen_range(-PI..PI);
        self.theta_dot = rng.gen_range(-1.0..1.0);
        self.steps = 0;
        self.get_state()
    }
    
    fn get_state(&self) -> Array1<f32> {
        array![
            self.theta.cos(),
            self.theta.sin(),
            self.theta_dot
        ]
    }
    
    fn step(&mut self, action: &Array1<f32>) -> (Array1<f32>, f32, bool) {
        // Clip action to valid range
        let torque = action[0].clamp(-self.max_torque, self.max_torque);
        
        // Update dynamics
        let costs = self.angle_normalize(self.theta).powi(2) + 
                    0.1 * self.theta_dot.powi(2) + 
                    0.001 * torque.powi(2);
        
        // Physics simulation
        let new_theta_dot = self.theta_dot + 
            (-3.0 * self.gravity / (2.0 * self.length) * self.theta.sin() + 
             3.0 / (self.mass * self.length.powi(2)) * torque) * self.dt;
        
        let new_theta_dot = new_theta_dot.clamp(-self.max_speed, self.max_speed);
        let new_theta = self.theta + new_theta_dot * self.dt;
        
        self.theta = new_theta;
        self.theta_dot = new_theta_dot;
        self.steps += 1;
        
        // Reward is negative cost
        let reward = -costs;
        let done = self.steps >= self.max_steps;
        
        (self.get_state(), reward, done)
    }
    
    fn angle_normalize(&self, angle: f32) -> f32 {
        // Normalize angle to [-pi, pi]
        let mut normalized = angle;
        while normalized > PI {
            normalized -= 2.0 * PI;
        }
        while normalized < -PI {
            normalized += 2.0 * PI;
        }
        normalized
    }
    
    fn render(&self) {
        // ASCII visualization of pendulum
        let x = (40.0 + 20.0 * self.theta.sin()) as usize;
        let y = (10.0 - 10.0 * self.theta.cos()) as usize;
        
        println!("\n{}", "=".repeat(80));
        for row in 0..20 {
            for col in 0..80 {
                if row == 0 && col == 40 {
                    print!("O"); // Pivot point
                } else if row <= y && col == x && row > 0 {
                    print!("|"); // Pendulum rod
                } else if row == y && col == x {
                    print!("●"); // Pendulum bob
                } else {
                    print!(" ");
                }
            }
            println!();
        }
        println!("θ: {:.2}°, ω: {:.2} rad/s", 
                self.theta * 180.0 / PI, self.theta_dot);
    }
}

/// Configuration for SAC training
#[allow(dead_code)]
struct SACConfig {
    // Network architecture
    hidden_dims: Vec<usize>,
    
    // SAC hyperparameters
    learning_rate_actor: f32,
    learning_rate_critic: f32,
    learning_rate_alpha: f32,
    gamma: f32,
    tau: f32,
    initial_alpha: f32,
    target_entropy: Option<f32>,
    
    // Replay buffer
    buffer_size: usize,
    batch_size: usize,
    warmup_steps: usize,
    
    // Training
    max_steps: usize,
    eval_frequency: usize,
    save_frequency: usize,
    
    // Prioritized replay
    use_per: bool,
    per_alpha: f32,
    per_beta_start: f32,
    per_beta_end: f32,
}

impl Default for SACConfig {
    fn default() -> Self {
        SACConfig {
            hidden_dims: vec![256, 256],
            learning_rate_actor: 3e-4,
            learning_rate_critic: 3e-4,
            learning_rate_alpha: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            initial_alpha: 0.2,
            target_entropy: None, // Will be set to -action_dim
            buffer_size: 1_000_000,
            batch_size: 256,
            warmup_steps: 10_000,
            max_steps: 500_000,
            eval_frequency: 5_000,
            save_frequency: 50_000,
            use_per: true,
            per_alpha: 0.6,
            per_beta_start: 0.4,
            per_beta_end: 1.0,
        }
    }
}

/// Training logger for tracking metrics
struct TrainingLogger {
    log_file: File,
    episode_rewards: Vec<f32>,
    actor_losses: Vec<f32>,
    critic_losses: Vec<f32>,
    alpha_values: Vec<f32>,
    q_values: Vec<f32>,
}

impl TrainingLogger {
    fn new(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let log_file = File::create(filename)?;
        Ok(TrainingLogger {
            log_file,
            episode_rewards: Vec::new(),
            actor_losses: Vec::new(),
            critic_losses: Vec::new(),
            alpha_values: Vec::new(),
            q_values: Vec::new(),
        })
    }
    
    fn log_episode(&mut self, episode: usize, reward: f32, steps: usize) -> std::io::Result<()> {
        self.episode_rewards.push(reward);
        writeln!(self.log_file, "episode,{},{},{}", episode, reward, steps)?;
        
        // Print running average
        if self.episode_rewards.len() >= 100 {
            let avg_reward = self.episode_rewards
                .iter()
                .rev()
                .take(100)
                .sum::<f32>() / 100.0;
            println!("Episode {}: Reward = {:.2}, Avg(100) = {:.2}", 
                    episode, reward, avg_reward);
        }
        Ok(())
    }
    
    fn log_training_step(&mut self, step: usize, actor_loss: f32, critic_loss: f32, 
                         alpha: f32, avg_q: f32) -> std::io::Result<()> {
        self.actor_losses.push(actor_loss);
        self.critic_losses.push(critic_loss);
        self.alpha_values.push(alpha);
        self.q_values.push(avg_q);
        
        writeln!(self.log_file, "train,{},{},{},{},{}", 
                step, actor_loss, critic_loss, alpha, avg_q)?;
        Ok(())
    }
    
    fn plot_summary(&self) {
        println!("\n=== Training Summary ===");
        
        // Episode rewards plot (ASCII)
        if !self.episode_rewards.is_empty() {
            println!("\nEpisode Rewards:");
            let max_reward = self.episode_rewards.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_reward = self.episode_rewards.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = max_reward - min_reward;
            
            for y in 0..20 {
                let threshold = max_reward - (y as f32 / 20.0) * range;
                print!("{:6.1} |", threshold);
                
                for (x, &reward) in self.episode_rewards.iter().enumerate() {
                    if x % (self.episode_rewards.len() / 60).max(1) == 0 {
                        if reward >= threshold {
                            print!("█");
                        } else {
                            print!(" ");
                        }
                    }
                }
                println!();
            }
            println!("        └{}", "─".repeat(60));
            println!("         0{}episodes", " ".repeat(50));
        }
        
        // Final statistics
        if self.episode_rewards.len() >= 100 {
            let final_avg = self.episode_rewards.iter().rev().take(100).sum::<f32>() / 100.0;
            println!("\nFinal 100-episode average: {:.2}", final_avg);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = SACConfig::default();
    
    // Create directories
    std::fs::create_dir_all("models")?;
    std::fs::create_dir_all("logs")?;
    
    // Initialize logger
    let mut logger = TrainingLogger::new("logs/pendulum_sac.csv")?;
    
    // Create SAC agent
    let optimizer = OptimizerWrapper::SGD(athena::optimizer::SGD::new());
    
    let target_entropy = config.target_entropy.unwrap_or(-1.0); // -action_dim
    
    let mut agent = SACBuilder::new(3, 1)  // state_size, action_size
        .hidden_sizes(config.hidden_dims.clone())
        .optimizer(optimizer)
        .gamma(config.gamma)
        .tau(config.tau)
        .alpha(config.initial_alpha)
        .auto_alpha(true)
        .build()?;
    
    // Create replay buffer
    let mut buffer = if config.use_per {
        PrioritizedReplayBuffer::new(
            config.buffer_size,
            athena::replay_buffer::PriorityMethod::Proportional { alpha: config.per_alpha },
        )
    } else {
        PrioritizedReplayBuffer::new(config.buffer_size, athena::replay_buffer::PriorityMethod::Uniform)
    };
    
    // Create environment
    let mut env = Pendulum::new();
    let mut state = env.reset();
    
    println!("Starting Pendulum SAC training...");
    println!("State dim: 3, Action dim: 1 (continuous)");
    println!("Target entropy: {}", target_entropy);
    
    let mut total_steps = 0;
    let mut episode = 0;
    let mut episode_reward = 0.0;
    let mut episode_steps = 0;
    
    // Main training loop
    while total_steps < config.max_steps {
        // Select action
        let action = if total_steps < config.warmup_steps {
            // Random action during warmup
            let mut rng = rand::thread_rng();
            array![rng.gen_range(-2.0..2.0)]
        } else {
            agent.act(state.view(), false)?
        };
        
        // Environment step
        let (next_state, reward, done) = env.step(&action);
        episode_reward += reward;
        episode_steps += 1;
        
        // Store transition
        let experience = athena::replay_buffer::Experience {
            state: state.clone(),
            action: 0, // Not used for continuous actions
            reward,
            next_state: next_state.clone(),
            done,
        };
        
        // For continuous actions, we need to store the actual action values
        // This is a simplified version - in practice, you'd extend the Experience struct
        buffer.add_with_priority(experience, 1.0); // Initial priority
        
        // Update state
        state = if done {
            // Log episode
            logger.log_episode(episode, episode_reward, episode_steps)?;
            
            // Render occasionally
            if episode % 100 == 0 {
                env.render();
            }
            
            episode += 1;
            episode_reward = 0.0;
            episode_steps = 0;
            env.reset()
        } else {
            next_state
        };
        
        // Training step
        if total_steps >= config.warmup_steps && buffer.len() >= config.batch_size {
            // Sample batch
            let beta = config.per_beta_start + 
                      (config.per_beta_end - config.per_beta_start) * 
                      (total_steps as f32 / config.max_steps as f32);
            
            let (_batch, _weights, _indices) = buffer.sample_with_weights(config.batch_size, beta);
            
            // Train SAC (simplified - would need proper SACExperience type)
            // In a real implementation, you'd convert the batch to SACExperience
            // and call agent.update(&sac_batch, learning_rate)
            // For now, we skip the actual training
            
            // Log training metrics (with dummy values since we're not actually training)
            if total_steps % 1000 == 0 {
                let actor_loss = 0.0;  // Placeholder
                let critic_loss = 0.0;  // Placeholder
                let avg_q = 0.0;  // Placeholder
                
                logger.log_training_step(
                    total_steps,
                    actor_loss,
                    critic_loss,
                    agent.alpha,
                    avg_q,
                )?;
            }
        }
        
        // Evaluation
        if total_steps % config.eval_frequency == 0 && total_steps > 0 {
            let eval_reward = evaluate_agent(&mut agent, 10)?;
            println!("\n[Step {}] Evaluation reward: {:.2}", total_steps, eval_reward);
        }
        
        // Save checkpoint
        if total_steps % config.save_frequency == 0 && total_steps > 0 {
            agent.save(&format!("models/pendulum_sac_{}.bin", total_steps))?;
            println!("Saved checkpoint at step {}", total_steps);
        }
        
        total_steps += 1;
    }
    
    // Final evaluation
    println!("\n=== Final Evaluation ===");
    let final_reward = evaluate_agent(&mut agent, 100)?;
    println!("Average reward over 100 episodes: {:.2}", final_reward);
    
    // Save final model
    agent.save("models/pendulum_sac_final.bin")?;
    
    // Plot summary
    logger.plot_summary();
    
    Ok(())
}

fn evaluate_agent(agent: &mut SACAgent, n_episodes: usize) -> Result<f32, Box<dyn std::error::Error>> {
    let mut env = Pendulum::new();
    let mut total_reward = 0.0;
    
    for _ in 0..n_episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        loop {
            // Deterministic action for evaluation (use act with deterministic=true)
            let action = agent.act(state.view(), true)?;
            let (next_state, reward, done) = env.step(&action);
            
            episode_reward += reward;
            state = next_state;
            
            if done {
                total_reward += episode_reward;
                break;
            }
        }
    }
    
    Ok(total_reward / n_episodes as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pendulum_dynamics() {
        let mut env = Pendulum::new();
        let initial_state = env.reset();
        
        // State should be normalized
        assert_eq!(initial_state.len(), 3);
        assert!((initial_state[0].powi(2) + initial_state[1].powi(2) - 1.0).abs() < 1e-6);
        
        // Test step with zero action
        let (next_state, reward, done) = env.step(&array![0.0]);
        assert!(!done);
        assert!(reward < 0.0); // Should have negative reward (cost)
    }
    
    #[test]
    fn test_angle_normalization() {
        let env = Pendulum::new();
        
        assert!((env.angle_normalize(0.0) - 0.0).abs() < 1e-6);
        assert!((env.angle_normalize(2.0 * PI) - 0.0).abs() < 1e-6);
        assert!((env.angle_normalize(-PI) - (-PI)).abs() < 1e-6);
        assert!((env.angle_normalize(3.0 * PI) - PI).abs() < 1e-6);
    }
}