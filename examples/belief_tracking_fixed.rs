//! Fixed Belief State Tracking Example
//! 
//! This example demonstrates belief state tracking in a partially observable grid world.

use athena::belief::{HistoryBelief, BeliefState};
use athena::belief::belief_agent::BeliefDqnAgent;
use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::replay_buffer::{ReplayBuffer, Experience};
use ndarray::Array1;
use rand::Rng;

/// Partially Observable GridWorld
struct POGridWorld {
    size: usize,
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    observation_radius: usize,
}

impl POGridWorld {
    fn new(size: usize) -> Self {
        Self {
            size,
            agent_pos: (0, 0),
            goal_pos: (size - 1, size - 1),
            observation_radius: 1, // Can only see adjacent cells
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        self.agent_pos = (0, 0);
        self.get_observation()
    }
    
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        // Move agent based on action (0: up, 1: right, 2: down, 3: left)
        let (x, y) = self.agent_pos;
        let new_pos = match action {
            0 if y > 0 => (x, y - 1),
            1 if x < self.size - 1 => (x + 1, y),
            2 if y < self.size - 1 => (x, y + 1),
            3 if x > 0 => (x - 1, y),
            _ => (x, y), // Invalid move
        };
        
        self.agent_pos = new_pos;
        
        // Calculate reward
        let reward = if self.agent_pos == self.goal_pos {
            10.0 // Reward for reaching goal
        } else {
            -0.1 // Small penalty per step
        };
        
        let done = self.agent_pos == self.goal_pos;
        let obs = self.get_observation();
        
        (obs, reward, done)
    }
    
    /// Get partial observation (only see adjacent cells)
    fn get_observation(&self) -> Array1<f32> {
        let (ax, ay) = self.agent_pos;
        let mut obs = Array1::zeros(9); // 3x3 grid around agent
        
        // Encode what agent can see
        let mut idx = 0;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let y = (ay as i32 + dy).max(0).min(self.size as i32 - 1) as usize;
                let x = (ax as i32 + dx).max(0).min(self.size as i32 - 1) as usize;
                
                if (x, y) == self.goal_pos {
                    obs[idx] = 1.0; // Goal visible
                } else if (x, y) == self.agent_pos {
                    obs[idx] = 0.5; // Agent position
                }
                // Otherwise 0.0 for empty
                
                idx += 1;
            }
        }
        
        obs
    }
}

fn main() {
    println!("Belief State Tracking Example");
    println!("=============================\n");
    
    // Create environment
    let mut env = POGridWorld::new(5);
    
    // Create base DQN agent
    let belief_encoding_size = 32;
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let base_agent = DqnAgent::new(
        &[belief_encoding_size, 64, 32, 4], // 4 actions
        1.0, // Start with high exploration
        optimizer,
        100, // Update target network frequency
        true, // Use double DQN
    );
    
    // Create belief state (history-based)
    let belief = HistoryBelief::new(5, 9 * 5); // 5 steps history, 9 features per step
    
    // Create belief-tracking agent
    let mut agent = BeliefDqnAgent::new(
        base_agent,
        belief,
        9, // observation size
        belief_encoding_size,
    );
    
    // Training parameters
    let episodes = 200;
    let max_steps = 50;
    let batch_size = 32;
    
    // Create replay buffer for regular experiences
    let mut replay_buffer = ReplayBuffer::new(5000);
    let mut episode_rewards = Vec::new();
    
    for episode in 0..episodes {
        let mut obs = env.reset();
        agent.belief_agent.reset();
        
        let mut episode_reward = 0.0;
        let mut steps = 0;
        
        for _ in 0..max_steps {
            // Get action from belief-based agent
            let action = agent.act(&obs).unwrap();
            
            // Take step in environment
            let (next_obs, reward, done) = env.step(action);
            episode_reward += reward;
            steps += 1;
            
            // Store experience with encoded belief states
            let belief_vec = agent.belief_agent.belief.to_feature_vector();
            let encoded_state = agent.belief_agent.belief_encoder.forward(belief_vec.view());
            
            // Update belief for next state encoding
            agent.belief_agent.belief.update(action, &next_obs);
            let next_belief_vec = agent.belief_agent.belief.to_feature_vector();
            let encoded_next_state = agent.belief_agent.belief_encoder.forward(next_belief_vec.view());
            
            replay_buffer.add(Experience {
                state: encoded_state,
                action,
                reward,
                next_state: encoded_next_state,
                done,
            });
            
            // Train if we have enough samples
            if replay_buffer.len() >= batch_size {
                let batch = replay_buffer.sample(batch_size);
                let _ = agent.base_agent.train_on_batch(&batch, 0.99, 0.001);
            }
            
            obs = next_obs;
            
            if done {
                break;
            }
        }
        
        episode_rewards.push(episode_reward);
        
        // Decay epsilon
        agent.base_agent.epsilon = (agent.base_agent.epsilon * 0.995).max(0.01);
        
        // Print progress
        if (episode + 1) % 20 == 0 {
            let avg_reward: f32 = episode_rewards
                .iter()
                .skip(episode.saturating_sub(19))
                .sum::<f32>() / 20.0;
            
            println!(
                "Episode {}: steps={}, reward={:.2}, avg_reward={:.2}, epsilon={:.3}, entropy={:.3}",
                episode + 1, steps, episode_reward, avg_reward, 
                agent.base_agent.epsilon, agent.belief_agent.get_entropy()
            );
        }
    }
    
    println!("\nTraining complete!");
}