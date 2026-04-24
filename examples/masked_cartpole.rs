//! Example: CartPole with Action Masking
//! 
//! This example demonstrates how to use action masking with DQN agent.
//! We simulate a modified CartPole environment where certain actions
//! become unavailable based on the cart position.

#[cfg(feature = "action-masking")]
use athena::agent::{DqnAgent, MaskedAgent};
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::replay_buffer::{ReplayBuffer, Experience};
use ndarray::{Array1, array};
use rand::Rng;

/// Simple CartPole environment with action masking
struct MaskedCartPole {
    state: Array1<f32>,
    done: bool,
    step_count: usize,
    max_steps: usize,
    boundary: f32,
}

impl MaskedCartPole {
    fn new() -> Self {
        Self {
            state: array![0.0, 0.0, 0.0, 0.0],
            done: false,
            step_count: 0,
            max_steps: 200,
            boundary: 2.4,
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        self.state = Array1::from_vec(vec![
            rng.gen_range(-0.05..0.05),   // cart position
            0.0,                            // cart velocity
            rng.gen_range(-0.05..0.05),   // pole angle
            0.0,                            // pole velocity
        ]);
        self.done = false;
        self.step_count = 0;
        self.state.clone()
    }
    
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        if self.done {
            return (self.state.clone(), 0.0, true);
        }
        
        // Simple physics simulation
        let force = if action == 0 { -10.0 } else { 10.0 };
        let cart_pos = self.state[0];
        let cart_vel = self.state[1];
        let pole_angle = self.state[2];
        let pole_vel = self.state[3];
        
        // Update dynamics (simplified)
        let new_cart_vel = cart_vel + 0.01 * force;
        let new_cart_pos = cart_pos + 0.01 * new_cart_vel;
        let new_pole_vel = pole_vel + 0.01 * pole_angle * 9.8;
        let new_pole_angle = pole_angle + 0.01 * new_pole_vel;
        
        self.state = array![new_cart_pos, new_cart_vel, new_pole_angle, new_pole_vel];
        self.step_count += 1;
        
        // Check termination
        let done = new_cart_pos.abs() > self.boundary || 
                  new_pole_angle.abs() > 0.2 || 
                  self.step_count >= self.max_steps;
        self.done = done;
        
        let reward = if done && self.step_count < self.max_steps { 
            -1.0 
        } else { 
            1.0 
        };
        
        (self.state.clone(), reward, done)
    }
    
    /// Get action mask based on cart position
    /// Prevent moving further when near boundaries
    fn get_action_mask(&self) -> Array1<bool> {
        let cart_pos = self.state[0];
        let cart_vel = self.state[1];
        
        // Near left boundary and moving left: can't go left
        let can_go_left = !(cart_pos < -self.boundary * 0.8 && cart_vel < 0.0);
        
        // Near right boundary and moving right: can't go right  
        let can_go_right = !(cart_pos > self.boundary * 0.8 && cart_vel > 0.0);
        
        array![can_go_left, can_go_right]
    }
}

#[cfg(feature = "action-masking")]
fn main() {
    println!("CartPole with Action Masking Example");
    println!("=====================================");
    
    // Create environment
    let mut env = MaskedCartPole::new();
    
    // Create DQN agent with masked actions
    let layer_sizes = &[4, 64, 64, 2];  // 4 inputs, 2 actions
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(layer_sizes, 0.1, optimizer, 10000, true);
    
    // Create replay buffer
    let mut buffer = ReplayBuffer::new(10000);
    
    // Training parameters
    let episodes = 100;
    let batch_size = 32;
    let update_frequency = 4;
    let target_update_frequency = 100;
    
    let mut total_steps = 0;
    let mut episode_rewards = Vec::new();
    
    for episode in 0usize..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut episode_length = 0;
        let mut masked_actions = 0;
        
        loop {
            // Get action mask from environment
            let action_mask = env.get_action_mask();
            
            // Track when actions are masked
            if !action_mask[0] || !action_mask[1] {
                masked_actions += 1;
            }
            
            // Select action with masking
            let action = agent.act_masked(state.view(), &action_mask);
            
            // Take step
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;
            episode_length += 1;
            
            // Store experience
            buffer.add(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train agent
            if buffer.len() >= batch_size && total_steps % update_frequency == 0 {
                let batch = buffer.sample(batch_size);
                let _ = agent.train_on_batch(&batch, 0.99, 0.001);
            }
            
            // Update target network
            if total_steps % target_update_frequency == 0 {
                agent.update_target_network();
            }
            
            state = next_state;
            total_steps += 1;
            
            if done {
                break;
            }
        }
        
        episode_rewards.push(episode_reward);
        
        // Decay epsilon
        agent.epsilon = (agent.epsilon * 0.99).max(0.01);
        
        if episode % 10 == 0 {
            let avg_reward: f32 = episode_rewards.iter()
                .skip(episode.saturating_sub(10))
                .sum::<f32>() / 10.0_f32.min(episode_rewards.len() as f32);
            println!(
                "Episode {}: length={}, reward={:.1}, avg_reward={:.1}, masked_actions={}, epsilon={:.3}",
                episode, episode_length, episode_reward, avg_reward, masked_actions, agent.epsilon
            );
        }
    }
    
    println!("\nTraining complete!");
    
    // Test the trained agent
    println!("\nTesting trained agent (no exploration):");
    agent.epsilon = 0.0;
    
    for test_ep in 0..5 {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut masked_count = 0;
        
        while !env.done {
            let action_mask = env.get_action_mask();
            if !action_mask[0] || !action_mask[1] {
                masked_count += 1;
            }
            
            let action = agent.act_masked(state.view(), &action_mask);
            let (next_state, reward, _) = env.step(action);
            episode_reward += reward;
            state = next_state;
        }
        
        println!("Test episode {}: reward={:.1}, length={}, masked_actions={}", 
                 test_ep, episode_reward, env.step_count, masked_count);
    }
}

#[cfg(not(feature = "action-masking"))]
fn main() {
    println!("This example requires the 'action-masking' feature.");
    println!("Run with: cargo run --example masked_cartpole --features action-masking");
}