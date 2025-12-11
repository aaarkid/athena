//! CartPole environment solved using PPO (Proximal Policy Optimization)
//! 
//! This example demonstrates:
//! - Using PPO for continuous control
//! - Custom environment implementation
//! - Proper state preprocessing
//! - Training monitoring and visualization
//! - Model checkpointing

use athena::algorithms::{PPOAgent, PPOBuilder};
use athena::optimizer::{OptimizerWrapper, Adam, LearningRateScheduler};
use athena::metrics::MetricsTracker;
use ndarray::{Array1, Array2, array};
use rand::Rng;
use std::f32::consts::PI;

/// CartPole environment implementation
struct CartPole {
    // State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    state: Array1<f32>,
    
    // Environment parameters
    gravity: f32,
    cart_mass: f32,
    pole_mass: f32,
    pole_length: f32,
    force_magnitude: f32,
    dt: f32,
    
    // Episode tracking
    steps: usize,
    max_steps: usize,
}

impl CartPole {
    fn new() -> Self {
        CartPole {
            state: Array1::zeros(4),
            gravity: 9.8,
            cart_mass: 1.0,
            pole_mass: 0.1,
            pole_length: 0.5,
            force_magnitude: 10.0,
            dt: 0.02,
            steps: 0,
            max_steps: 500,
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        self.state = array![
            rng.gen_range(-0.05..0.05),  // cart position
            rng.gen_range(-0.05..0.05),  // cart velocity
            rng.gen_range(-0.05..0.05),  // pole angle
            rng.gen_range(-0.05..0.05),  // pole angular velocity
        ];
        self.steps = 0;
        self.state.clone()
    }
    
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        // Convert discrete action to force
        let force = if action == 1 { self.force_magnitude } else { -self.force_magnitude };
        
        // Unpack state
        let x = self.state[0];
        let x_dot = self.state[1];
        let theta = self.state[2];
        let theta_dot = self.state[3];
        
        // Physics simulation
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        
        let total_mass = self.cart_mass + self.pole_mass;
        let pole_mass_length = self.pole_mass * self.pole_length;
        
        let temp = (force + pole_mass_length * theta_dot * theta_dot * sin_theta) / total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp) /
            (self.pole_length * (4.0 / 3.0 - self.pole_mass * cos_theta * cos_theta / total_mass));
        let x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;
        
        // Update state
        self.state[0] += x_dot * self.dt;
        self.state[1] += x_acc * self.dt;
        self.state[2] += theta_dot * self.dt;
        self.state[3] += theta_acc * self.dt;
        
        self.steps += 1;
        
        // Check termination conditions
        let done = x.abs() > 2.4 ||
                   theta.abs() > 12.0 * PI / 180.0 ||
                   self.steps >= self.max_steps;
        
        // Reward is 1 for every step the pole is upright
        let reward = if done && self.steps < self.max_steps { 0.0 } else { 1.0 };
        
        (self.state.clone(), reward, done)
    }
    
    fn render(&self) {
        // Simple ASCII rendering
        let cart_pos = (self.state[0] * 10.0 + 40.0) as usize;
        let pole_angle = self.state[2];
        
        println!("\n{}", "=".repeat(80));
        
        // Draw pole
        let pole_x = cart_pos as f32 + (pole_angle.sin() * 10.0);
        let pole_y = 5.0 - (pole_angle.cos() * 5.0);
        
        for y in 0..10 {
            for x in 0..80 {
                if y == 9 {
                    // Ground
                    print!("-");
                } else if x == cart_pos && y >= 7 {
                    // Cart
                    print!("█");
                } else if (x as f32 - pole_x).abs() < 1.0 && (y as f32 - pole_y).abs() < 1.0 {
                    // Pole
                    print!("│");
                } else {
                    print!(" ");
                }
            }
            println!();
        }
        
        println!("Step: {} | Angle: {:.2}°", self.steps, self.state[2] * 180.0 / PI);
    }
}

/// State preprocessor for normalization
struct StateNormalizer {
    mean: Array1<f32>,
    std: Array1<f32>,
    count: f32,
}

impl StateNormalizer {
    fn new(state_dim: usize) -> Self {
        StateNormalizer {
            mean: Array1::zeros(state_dim),
            std: Array1::ones(state_dim),
            count: 0.0,
        }
    }
    
    fn update(&mut self, state: &Array1<f32>) {
        self.count += 1.0;
        let delta = state - &self.mean;
        self.mean = &self.mean + &delta / self.count;
        let delta2 = state - &self.mean;
        
        if self.count > 1.0 {
            let variance = (&self.std * &self.std * (self.count - 1.0) + &delta * &delta2) / self.count;
            self.std = variance.mapv(f32::sqrt);
        }
    }
    
    fn normalize(&self, state: &Array1<f32>) -> Array1<f32> {
        if self.count > 1.0 {
            (state - &self.mean) / (&self.std + 1e-8)
        } else {
            state.clone()
        }
    }
}

/// Training configuration
struct Config {
    // PPO hyperparameters
    hidden_dims: Vec<usize>,
    learning_rate: f32,
    gamma: f32,
    gae_lambda: f32,
    clip_epsilon: f32,
    value_coeff: f32,
    entropy_coeff: f32,
    
    // Training parameters
    n_envs: usize,
    n_steps: usize,
    n_epochs: usize,
    minibatch_size: usize,
    max_episodes: usize,
    
    // Other
    render_frequency: usize,
    save_frequency: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            hidden_dims: vec![64, 64],
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            value_coeff: 0.5,
            entropy_coeff: 0.01,
            n_envs: 8,
            n_steps: 128,
            n_epochs: 10,
            minibatch_size: 32,
            max_episodes: 1000,
            render_frequency: 100,
            save_frequency: 100,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::default();
    
    // Create PPO agent
    let optimizer = OptimizerWrapper::Adam(Adam::new(
        config.learning_rate,
        0.9,
        0.999,
        1e-8
    ));
    
    let mut agent = PPOBuilder::new()
        .input_dim(4)  // CartPole has 4-dimensional state
        .action_dim(2) // 2 discrete actions (left/right)
        .hidden_dims(config.hidden_dims.clone())
        .optimizer(optimizer)
        .gamma(config.gamma)
        .gae_lambda(config.gae_lambda)
        .clip_epsilon(config.clip_epsilon)
        .value_coeff(config.value_coeff)
        .entropy_coeff(config.entropy_coeff)
        .build()?;
    
    // Create environments
    let mut envs: Vec<CartPole> = (0..config.n_envs)
        .map(|_| CartPole::new())
        .collect();
    
    // Initialize state normalizer
    let mut normalizer = StateNormalizer::new(4);
    
    // Metrics tracking
    let mut metrics = MetricsTracker::new();
    let mut episode_count = 0;
    let mut total_steps = 0;
    
    // Learning rate scheduler
    let lr_scheduler = LearningRateScheduler::cosine(
        config.learning_rate,
        config.learning_rate * 0.1,
        config.max_episodes * config.n_steps,
    );
    
    println!("Starting CartPole PPO training...");
    println!("Environment: {} parallel instances", config.n_envs);
    println!("PPO Config: clip_eps={}, value_coeff={}, entropy_coeff={}",
             config.clip_epsilon, config.value_coeff, config.entropy_coeff);
    
    // Main training loop
    while episode_count < config.max_episodes {
        // Collect rollouts
        let mut rollout_states = Vec::new();
        let mut rollout_actions = Vec::new();
        let mut rollout_rewards = Vec::new();
        let mut rollout_dones = Vec::new();
        let mut rollout_values = Vec::new();
        
        // Reset environments
        let mut states: Vec<_> = envs.iter_mut()
            .map(|env| env.reset())
            .collect();
        
        // Collect experience
        for _ in 0..config.n_steps {
            // Normalize states
            let normalized_states: Vec<_> = states.iter()
                .map(|s| {
                    normalizer.update(s);
                    normalizer.normalize(s)
                })
                .collect();
            
            // Get actions and values from agent
            let mut actions = Vec::new();
            let mut values = Vec::new();
            
            for state in &normalized_states {
                let (action, value) = agent.act_with_value(state)?;
                actions.push(action);
                values.push(value);
            }
            
            // Step environments
            let mut next_states = Vec::new();
            let mut rewards = Vec::new();
            let mut dones = Vec::new();
            
            for (i, env) in envs.iter_mut().enumerate() {
                let (next_state, reward, done) = env.step(actions[i]);
                
                if done {
                    let episode_reward = env.steps as f32;
                    metrics.add_episode_reward(episode_reward);
                    episode_count += 1;
                    
                    if episode_count % config.render_frequency == 0 {
                        println!("\nEpisode {}: Reward = {}", episode_count, episode_reward);
                        env.render();
                    }
                    
                    next_states.push(env.reset());
                } else {
                    next_states.push(next_state);
                }
                
                rewards.push(reward);
                dones.push(done);
            }
            
            // Store rollout data
            rollout_states.extend(normalized_states);
            rollout_actions.extend(actions);
            rollout_rewards.extend(rewards);
            rollout_dones.extend(dones);
            rollout_values.extend(values);
            
            states = next_states;
            total_steps += config.n_envs;
        }
        
        // Get final values for bootstrap
        let final_values: Vec<_> = states.iter()
            .map(|s| {
                let normalized = normalizer.normalize(s);
                agent.value(&normalized).unwrap_or(0.0)
            })
            .collect();
        
        // Convert to arrays for training
        let states_array = Array2::from_shape_vec(
            (rollout_states.len(), 4),
            rollout_states.into_iter().flatten().collect()
        )?;
        
        let actions_array = Array1::from_vec(rollout_actions);
        let rewards_array = Array1::from_vec(rollout_rewards);
        let dones_array = Array1::from_vec(
            rollout_dones.into_iter().map(|d| if d { 1.0 } else { 0.0 }).collect()
        );
        let values_array = Array1::from_vec(rollout_values);
        let final_values_array = Array1::from_vec(final_values);
        
        // Update learning rate
        let current_lr = lr_scheduler.get_lr(total_steps);
        
        // Train PPO
        agent.train(
            states_array.view(),
            actions_array.view(),
            rewards_array.view(),
            dones_array.view(),
            values_array.view(),
            final_values_array.view(),
            current_lr,
        )?;
        
        // Save checkpoint
        if episode_count % config.save_frequency == 0 && episode_count > 0 {
            agent.save(&format!("models/cartpole_ppo_{}.bin", episode_count))?;
            println!("Saved checkpoint at episode {}", episode_count);
        }
        
        // Print statistics
        if episode_count % 10 == 0 && episode_count > 0 {
            let recent_rewards = metrics.get_episode_rewards();
            let avg_reward = recent_rewards.iter().rev().take(100).sum::<f32>() / 100.0.min(recent_rewards.len() as f32);
            
            println!("\n=== Training Progress ===");
            println!("Episodes: {}/{}", episode_count, config.max_episodes);
            println!("Total Steps: {}", total_steps);
            println!("Average Reward (last 100): {:.2}", avg_reward);
            println!("Learning Rate: {:.6}", current_lr);
            
            if avg_reward >= 490.0 {
                println!("\n🎉 CartPole solved! Average reward >= 490");
                break;
            }
        }
    }
    
    // Final evaluation
    println!("\n=== Final Evaluation ===");
    let mut eval_env = CartPole::new();
    let mut eval_rewards = Vec::new();
    
    for _ in 0..100 {
        let mut state = eval_env.reset();
        let mut episode_reward = 0.0;
        
        loop {
            let normalized_state = normalizer.normalize(&state);
            let action = agent.act(&normalized_state)?;
            let (next_state, reward, done) = eval_env.step(action);
            
            episode_reward += reward;
            state = next_state;
            
            if done {
                eval_rewards.push(episode_reward);
                break;
            }
        }
    }
    
    let mean_reward = eval_rewards.iter().sum::<f32>() / eval_rewards.len() as f32;
    let std_reward = (eval_rewards.iter()
        .map(|r| (r - mean_reward).powi(2))
        .sum::<f32>() / eval_rewards.len() as f32)
        .sqrt();
    
    println!("Evaluation over 100 episodes:");
    println!("Mean Reward: {:.2} ± {:.2}", mean_reward, std_reward);
    println!("Max Reward: {:.0}", eval_rewards.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("Min Reward: {:.0}", eval_rewards.iter().cloned().fold(f32::INFINITY, f32::min));
    
    // Save final model
    agent.save("models/cartpole_ppo_final.bin")?;
    println!("\nFinal model saved to models/cartpole_ppo_final.bin");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cartpole_physics() {
        let mut env = CartPole::new();
        let initial_state = env.reset();
        
        // Test that reset gives small values
        assert!(initial_state.iter().all(|&x| x.abs() < 0.05));
        
        // Test step function
        let (next_state, reward, done) = env.step(1);
        assert!(!done);
        assert_eq!(reward, 1.0);
        assert_ne!(initial_state, next_state);
    }
    
    #[test]
    fn test_state_normalizer() {
        let mut normalizer = StateNormalizer::new(4);
        
        // Add some states
        for i in 0..100 {
            let state = array![i as f32, i as f32 * 2.0, i as f32 * 3.0, i as f32 * 4.0];
            normalizer.update(&state);
        }
        
        // Check normalization
        let test_state = array![50.0, 100.0, 150.0, 200.0];
        let normalized = normalizer.normalize(&test_state);
        
        // Should be close to zero mean
        assert!(normalized.mean().unwrap().abs() < 0.1);
    }
}