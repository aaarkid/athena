//! Simple Grid World example that actually works
//! 
//! This example shows a basic DQN agent learning to navigate to a goal.

use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::replay_buffer::{ReplayBuffer, Experience};
use ndarray::Array1;

/// Simple GridWorld environment
struct GridWorld {
    size: usize,
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
}

impl GridWorld {
    fn new(size: usize) -> Self {
        Self {
            size,
            agent_pos: (0, 0),
            goal_pos: (size - 1, size - 1),
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        self.agent_pos = (0, 0);
        self.get_state()
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
            100.0 // Big reward for reaching goal
        } else {
            // Small negative reward to encourage finding goal quickly
            -1.0
        };
        
        let done = self.agent_pos == self.goal_pos;
        let state = self.get_state();
        
        (state, reward, done)
    }
    
    /// Get full state (agent knows its exact position)
    fn get_state(&self) -> Array1<f32> {
        let mut state = Array1::zeros(self.size * self.size);
        let idx = self.agent_pos.0 * self.size + self.agent_pos.1;
        state[idx] = 1.0;
        
        // Also encode goal position
        let goal_idx = self.goal_pos.0 * self.size + self.goal_pos.1;
        state[goal_idx] = 0.5;
        
        state
    }
}

fn main() {
    println!("Simple Grid World DQN Example");
    println!("=============================\n");
    
    // Create environment
    let grid_size = 5;
    let mut env = GridWorld::new(grid_size);
    
    // Create DQN agent
    let state_size = grid_size * grid_size;
    let action_size = 4;
    let layer_sizes = &[state_size, 64, 32, action_size];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    
    let mut agent = DqnAgent::new(
        layer_sizes,
        1.0, // Start with full exploration
        optimizer,
        100, // Update target network every 100 steps
        true, // Use double DQN
    );
    
    // Create replay buffer
    let mut replay_buffer = ReplayBuffer::new(10000);
    
    // Training parameters
    let episodes = 300;
    let max_steps = 50;
    let batch_size = 32;
    let learning_rate = 0.01;
    let gamma = 0.95;
    
    let mut episode_rewards = Vec::new();
    let mut total_steps = 0;
    
    for episode in 0usize..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        for step in 0..max_steps {
            // Get action
            let action = agent.act(state.view()).unwrap();
            
            // Take step
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;
            
            // Store experience
            replay_buffer.add(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train if we have enough samples
            if replay_buffer.len() >= batch_size {
                let batch = replay_buffer.sample(batch_size);
                agent.train_on_batch(&batch, gamma, learning_rate).unwrap();
            }
            
            state = next_state;
            total_steps += 1;
            
            if done {
                break;
            }
        }
        
        episode_rewards.push(episode_reward);
        
        // Decay epsilon
        agent.epsilon = (agent.epsilon * 0.995).max(0.01);
        
        // Print progress
        if (episode + 1) % 50 == 0 {
            let avg_reward: f32 = episode_rewards
                .iter()
                .skip(episode.saturating_sub(49))
                .sum::<f32>() / 50.0_f32.min(episode_rewards.len() as f32);
            
            println!(
                "Episode {}: avg_reward={:.2}, epsilon={:.3}",
                episode + 1, avg_reward, agent.epsilon
            );
        }
    }
    
    println!("\nTraining complete!");
    println!("\nTesting trained agent...");
    
    // Test the trained agent
    agent.epsilon = 0.0; // No exploration
    let mut test_env = GridWorld::new(grid_size);
    let mut state = test_env.reset();
    let mut steps = 0;
    let mut path = vec![test_env.agent_pos];
    
    println!("Starting at: {:?}", test_env.agent_pos);
    
    for _ in 0..max_steps {
        let action = agent.act(state.view()).unwrap();
        let action_name = match action {
            0 => "up",
            1 => "right",
            2 => "down",
            3 => "left",
            _ => "unknown",
        };
        println!("Position: {:?}, Action: {} ({})", test_env.agent_pos, action, action_name);
        
        let (next_state, _reward, done) = test_env.step(action);
        steps += 1;
        path.push(test_env.agent_pos);
        
        state = next_state;
        
        if done {
            println!("Reached goal in {} steps!", steps);
            println!("Path: {:?}", path);
            break;
        }
    }
    
    if test_env.agent_pos != test_env.goal_pos {
        println!("Failed to reach goal :(");
    }
}