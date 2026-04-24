//! Simple Belief State Tracking Example
//! 
//! This example demonstrates belief state tracking in a partially observable environment
//! where the agent can only see limited information about its surroundings.

use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::replay_buffer::{ReplayBuffer, Experience};
use ndarray::Array1;

/// Simple belief tracker that maintains a history of observations
struct SimpleBeliefTracker {
    history_size: usize,
    observation_size: usize,
    history: Vec<Array1<f32>>,
}

impl SimpleBeliefTracker {
    fn new(history_size: usize, observation_size: usize) -> Self {
        Self {
            history_size,
            observation_size,
            history: Vec::new(),
        }
    }
    
    fn update(&mut self, observation: &Array1<f32>) {
        self.history.push(observation.clone());
        if self.history.len() > self.history_size {
            self.history.remove(0);
        }
    }
    
    fn get_belief_state(&self) -> Array1<f32> {
        // Concatenate all observations in history
        let mut belief = Array1::zeros(self.history_size * self.observation_size);
        
        for (i, obs) in self.history.iter().enumerate() {
            let start = i * self.observation_size;
            let end = start + self.observation_size;
            belief.slice_mut(ndarray::s![start..end]).assign(obs);
        }
        
        belief
    }
    
    fn reset(&mut self) {
        self.history.clear();
    }
}

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
    
    /// Get partial observation (what agent can see in radius)
    fn get_observation(&self) -> Array1<f32> {
        let (ax, ay) = self.agent_pos;
        let mut obs = Array1::zeros(5); // 5 features: can_go_up, can_go_right, can_go_down, can_go_left, see_goal
        
        // Check if agent can move in each direction
        obs[0] = if ay > 0 { 1.0 } else { 0.0 };
        obs[1] = if ax < self.size - 1 { 1.0 } else { 0.0 };
        obs[2] = if ay < self.size - 1 { 1.0 } else { 0.0 };
        obs[3] = if ax > 0 { 1.0 } else { 0.0 };
        
        // Check if goal is visible (within observation radius)
        let (gx, gy) = self.goal_pos;
        let dist = ((ax as i32 - gx as i32).abs() + (ay as i32 - gy as i32).abs()) as usize;
        obs[4] = if dist <= self.observation_radius { 1.0 } else { 0.0 };
        
        obs
    }
}

fn main() {
    println!("Simple Belief State Tracking Example");
    println!("===================================\n");
    
    // Create environment
    let mut env = POGridWorld::new(5);
    
    // Create belief tracker
    let history_size = 5;
    let observation_size = 5;
    let mut belief_tracker = SimpleBeliefTracker::new(history_size, observation_size);
    
    // Create DQN agent that uses belief states
    let belief_state_size = history_size * observation_size;
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(
        &[belief_state_size, 64, 32, 4], // 4 actions
        1.0, // Start with high exploration
        optimizer,
        100, // Update target network frequency
        true, // Use double DQN
    );
    
    // Training parameters
    let episodes = 300;
    let max_steps = 50;
    let batch_size = 32;
    
    // Create replay buffer
    let mut replay_buffer = ReplayBuffer::new(5000);
    let mut episode_rewards = Vec::new();
    
    for episode in 0usize..episodes {
        let obs = env.reset();
        belief_tracker.reset();
        belief_tracker.update(&obs);
        
        let mut episode_reward = 0.0;
        let mut steps = 0;
        
        for _ in 0..max_steps {
            // Get belief state and act
            let belief_state = belief_tracker.get_belief_state();
            let action = agent.act(belief_state.view()).unwrap();
            
            // Take step in environment
            let (next_obs, reward, done) = env.step(action);
            episode_reward += reward;
            steps += 1;
            
            // Update belief
            belief_tracker.update(&next_obs);
            let next_belief_state = belief_tracker.get_belief_state();
            
            // Store experience
            replay_buffer.add(Experience {
                state: belief_state,
                action,
                reward,
                next_state: next_belief_state,
                done,
            });
            
            // Train if we have enough samples
            if replay_buffer.len() >= batch_size {
                let batch = replay_buffer.sample(batch_size);
                let _ = agent.train_on_batch(&batch, 0.99, 0.001);
            }
            
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
                "Episode {}: steps={}, reward={:.2}, avg_reward={:.2}, epsilon={:.3}",
                episode + 1, steps, episode_reward, avg_reward, agent.epsilon
            );
        }
    }
    
    println!("\nTraining complete!");
    
    // Test the trained agent
    println!("\nTesting trained agent with belief tracking...");
    agent.epsilon = 0.0; // No exploration
    
    let mut test_env = POGridWorld::new(5);
    let mut test_belief = SimpleBeliefTracker::new(history_size, observation_size);
    
    let obs = test_env.reset();
    test_belief.reset();
    test_belief.update(&obs);
    
    let mut total_reward = 0.0;
    let mut steps = 0;
    
    println!("Starting at position: {:?}", test_env.agent_pos);
    
    for _ in 0..max_steps {
        let belief_state = test_belief.get_belief_state();
        let action = agent.act(belief_state.view()).unwrap();
        
        let action_name = match action {
            0 => "up",
            1 => "right", 
            2 => "down",
            3 => "left",
            _ => "unknown",
        };
        
        let (next_obs, reward, done) = test_env.step(action);
        total_reward += reward;
        steps += 1;
        
        println!("Step {}: Action={}, Position={:?}, Reward={:.1}", 
                 steps, action_name, test_env.agent_pos, reward);
        
        test_belief.update(&next_obs);
        
        if done {
            println!("\nReached goal in {} steps! Total reward: {:.1}", steps, total_reward);
            break;
        }
    }
    
    if test_env.agent_pos != test_env.goal_pos {
        println!("\nFailed to reach goal. Final position: {:?}", test_env.agent_pos);
    }
}