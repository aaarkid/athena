//! Example: Belief State Tracking for Partially Observable Environment
//! 
//! This example demonstrates how to use belief states with a partially
//! observable version of GridWorld where the agent can only see nearby cells.

#[cfg(feature = "belief-states")]
use athena::belief::{BeliefState, HistoryBelief, ParticleFilter, BeliefDqnAgent};
use ndarray::Array1;
use rand::Rng;

/// Partially Observable GridWorld
struct POGridWorld {
    size: usize,
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    obstacles: Vec<(usize, usize)>,
    observation_radius: usize,
}

impl POGridWorld {
    fn new(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        // Random obstacles
        let mut obstacles = Vec::new();
        for _ in 0..size {
            obstacles.push((rng.gen_range(0..size), rng.gen_range(0..size)));
        }
        
        Self {
            size,
            agent_pos: (0, 0),
            goal_pos: (size - 1, size - 1),
            obstacles,
            observation_radius: 2,
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        
        // Reset agent to random valid position
        loop {
            self.agent_pos = (rng.gen_range(0..self.size), rng.gen_range(0..self.size));
            if !self.obstacles.contains(&self.agent_pos) && self.agent_pos != self.goal_pos {
                break;
            }
        }
        
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
        
        // Check if new position is valid
        if !self.obstacles.contains(&new_pos) {
            self.agent_pos = new_pos;
        }
        
        // Calculate reward
        let reward = if self.agent_pos == self.goal_pos {
            10.0
        } else {
            -0.1 // Small penalty for each step
        };
        
        let done = self.agent_pos == self.goal_pos;
        let obs = self.get_observation();
        
        (obs, reward, done)
    }
    
    /// Get partial observation around agent
    fn get_observation(&self) -> Array1<f32> {
        let (ax, ay) = self.agent_pos;
        let radius = self.observation_radius;
        let obs_size = (2 * radius + 1) * (2 * radius + 1) * 3; // 3 channels
        let mut obs = Array1::zeros(obs_size);
        
        let mut idx = 0;
        for dy in 0..=(2 * radius) {
            for dx in 0..=(2 * radius) {
                let y = (ay as i32 + dy as i32 - radius as i32).max(0).min(self.size as i32 - 1) as usize;
                let x = (ax as i32 + dx as i32 - radius as i32).max(0).min(self.size as i32 - 1) as usize;
                
                // Channel 0: walls/obstacles
                if self.obstacles.contains(&(x, y)) {
                    obs[idx] = 1.0;
                }
                
                // Channel 1: goal
                if (x, y) == self.goal_pos {
                    obs[idx + obs_size / 3] = 1.0;
                }
                
                // Channel 2: agent
                if (x, y) == self.agent_pos {
                    obs[idx + 2 * obs_size / 3] = 1.0;
                }
                
                idx += 1;
            }
        }
        
        obs
    }
}

#[cfg(feature = "belief-states")]
fn main() {
    println!("Belief State Tracking Example");
    println!("=============================");
    
    // Create environment
    let mut env = POGridWorld::new(10);
    
    // Create base DQN agent
    let observation_size = (2 * env.observation_radius + 1).pow(2) * 3;
    let belief_encoding_size = 64;
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let base_agent = DqnAgent::new(
        &[belief_encoding_size, 128, 128, 4], // 4 actions
        0.1,
        optimizer,
        10000,
        true,
    );
    
    // Create belief state (history-based)
    let belief = HistoryBelief::new(10, belief_encoding_size * 2);
    
    // Create belief-tracking agent
    let mut agent = BeliefDqnAgent::new(
        base_agent,
        belief,
        observation_size,
        belief_encoding_size,
    );
    
    // Training parameters
    let episodes = 200;
    let max_steps = 100;
    let mut total_rewards = Vec::new();
    
    for episode in 0usize..episodes {
        let mut obs = env.reset();
        agent.belief_agent.reset();
        
        let mut episode_reward = 0.0;
        let mut episode_experiences = Vec::new();
        
        for step in 0..max_steps {
            // Get action from belief-based agent
            let action = agent.act(&obs).unwrap();
            
            // Take step in environment
            let (next_obs, reward, done) = env.step(action);
            episode_reward += reward;
            
            // Store experience
            episode_experiences.push((obs.clone(), action, reward, next_obs.clone(), done));
            
            obs = next_obs;
            
            if done {
                break;
            }
        }
        
        // Train on episode experiences
        if episode_experiences.len() > 10 {
            agent.train_on_belief_batch(&episode_experiences, 0.99, 0.001).unwrap();
        }
        
        total_rewards.push(episode_reward);
        
        // Decay epsilon
        agent.base_agent.epsilon = (agent.base_agent.epsilon * 0.99).max(0.01);
        
        if episode % 20 == 0 {
            let avg_reward: f32 = total_rewards.iter()
                .skip(episode.saturating_sub(20))
                .sum::<f32>() / 20.0_f32.min(total_rewards.len() as f32);
            println!(
                "Episode {}: reward={:.2}, avg_reward={:.2}, epsilon={:.3}",
                episode, episode_reward, avg_reward, agent.base_agent.epsilon
            );
        }
    }
    
    println!("\nTraining complete!");
    println!("\nTesting with particle filter belief state...");
    
    // Test with particle filter
    test_particle_filter_agent();
}

#[cfg(feature = "belief-states")]
fn test_particle_filter_agent() {
    // Simple state representation for particles
    #[derive(Clone, Debug)]
    struct GridState {
        x: usize,
        y: usize,
    }
    
    let _env = POGridWorld::new(10);
    
    // Create particle filter
    let initial_state = || GridState { x: 0, y: 0 };
    let transition = |state: &GridState, action: usize| {
        let mut new_state = state.clone();
        match action {
            0 if new_state.y > 0 => new_state.y -= 1,
            1 if new_state.x < 9 => new_state.x += 1,
            2 if new_state.y < 9 => new_state.y += 1,
            3 if new_state.x > 0 => new_state.x -= 1,
            _ => {}
        }
        new_state
    };
    
    let observe = |_state: &GridState| {
        // In real implementation, this would predict observation from state
        Array1::zeros(75) // Dummy observation
    };
    
    let pf = ParticleFilter::new(100, initial_state, transition, observe);
    
    println!("Particle filter created with 100 particles");
    println!("Effective sample size: {:.1}", pf.effective_sample_size());
    println!("Belief entropy: {:.3}", pf.entropy());
}

#[cfg(not(feature = "belief-states"))]
fn main() {
    println!("This example requires the 'belief-states' feature.");
    println!("Run with: cargo run --example belief_tracking --features belief-states");
}