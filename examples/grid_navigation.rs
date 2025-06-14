/***
# Grid Navigation

* The agent learns to navigate from a starting position to a goal position in a 5x5 grid
* State: 25-dimensional one-hot encoded vector representing the agent's position
* Actions: 4 possible moves (up, down, left, right)
* Rewards: -1 for each step, +10 for reaching the goal, -5 for hitting walls
* Episode ends when the agent reaches the goal or after 50 steps
***/

use athena::{
    agent::DqnAgent,
    replay_buffer::{ReplayBuffer, Experience},
    optimizer::{OptimizerWrapper, SGD},
};
use ndarray::Array1;

const GRID_SIZE: usize = 5;
const STATE_SIZE: usize = GRID_SIZE * GRID_SIZE;
const ACTION_SIZE: usize = 4;
const LEARNING_RATE: f32 = 0.0001;
const EPSILON: f32 = 1.0;
const EPSILON_DECAY: f32 = 0.995;
const EPSILON_MIN: f32 = 0.01;
const GAMMA: f32 = 0.95;
const CAPACITY: usize = 10000;
const BATCH_SIZE: usize = 32;
const EPISODES: usize = 500;
const MAX_STEPS: usize = 50;

#[derive(Clone)]
struct GridWorld {
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    steps: usize,
}

impl GridWorld {
    fn new() -> Self {
        Self {
            agent_pos: (0, 0),  // Start at top-left
            goal_pos: (4, 4),   // Goal at bottom-right
            steps: 0,
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.agent_pos = (0, 0);
        self.steps = 0;
    }

    fn get_state(&self) -> Array1<f32> {
        let mut state = Array1::zeros(STATE_SIZE);
        let index = self.agent_pos.0 * GRID_SIZE + self.agent_pos.1;
        state[index] = 1.0;
        state
    }

    fn step(&mut self, action: usize) -> (f32, bool) {
        self.steps += 1;
        
        let new_pos = match action {
            0 => (self.agent_pos.0.saturating_sub(1), self.agent_pos.1), // Up
            1 => ((self.agent_pos.0 + 1).min(GRID_SIZE - 1), self.agent_pos.1), // Down
            2 => (self.agent_pos.0, self.agent_pos.1.saturating_sub(1)), // Left
            3 => (self.agent_pos.0, (self.agent_pos.1 + 1).min(GRID_SIZE - 1)), // Right
            _ => panic!("Invalid action"),
        };

        let reward = if new_pos == self.agent_pos {
            -5.0  // Hit a wall
        } else if new_pos == self.goal_pos {
            10.0  // Reached goal
        } else {
            -1.0  // Normal step
        };

        self.agent_pos = new_pos;
        let done = self.agent_pos == self.goal_pos || self.steps >= MAX_STEPS;

        (reward, done)
    }
}

fn main() {
    println!("Training Grid Navigation Agent...");
    let start_time = std::time::Instant::now();

    // Create the DQN agent with SGD optimizer
    let layer_sizes = &[STATE_SIZE, 32, 16, ACTION_SIZE];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(layer_sizes, EPSILON, optimizer);
    
    // Create replay buffer
    let mut replay_buffer = ReplayBuffer::new(CAPACITY);
    
    // Training loop
    let mut epsilon = EPSILON;
    let mut episode_rewards = Vec::new();

    for episode in 0..EPISODES {
        let mut env = GridWorld::new();
        let mut state = env.get_state();
        let mut total_reward = 0.0;

        loop {
            // Agent selects action
            let action = agent.act(state.view());
            
            // Environment step
            let (reward, done) = env.step(action);
            total_reward += reward;
            
            // Get next state
            let next_state = env.get_state();
            
            // Store experience
            let experience = Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            };
            replay_buffer.add(experience);
            
            // Train if enough samples
            if replay_buffer.len() >= BATCH_SIZE {
                let experiences = replay_buffer.sample(BATCH_SIZE);
                agent.train_on_batch(&experiences, GAMMA, LEARNING_RATE);
            }
            
            state = next_state;
            
            if done {
                break;
            }
        }
        
        episode_rewards.push(total_reward);
        
        // Decay epsilon
        epsilon = (epsilon * EPSILON_DECAY).max(EPSILON_MIN);
        agent.update_epsilon(epsilon);
        
        // Print progress every 50 episodes
        if (episode + 1) % 50 == 0 {
            let avg_reward: f32 = episode_rewards.iter().skip(episode.saturating_sub(49)).sum::<f32>() / 50.0;
            println!("Episode {}: Average Reward = {:.2}, Epsilon = {:.3}", 
                     episode + 1, avg_reward, epsilon);
        }
    }

    println!("\nTraining completed in {:?}", start_time.elapsed());
    
    // Test the trained agent
    println!("\nTesting trained agent (epsilon = 0)...");
    agent.update_epsilon(0.0);
    
    for test in 0..3 {
        let mut env = GridWorld::new();
        let mut state = env.get_state();
        let mut path = vec![env.agent_pos];
        
        println!("\nTest {}: Starting from {:?}", test + 1, env.agent_pos);
        
        for step in 0..MAX_STEPS {
            let action = agent.act(state.view());
            let (_reward, done) = env.step(action);
            path.push(env.agent_pos);
            state = env.get_state();
            
            if done {
                if env.agent_pos == env.goal_pos {
                    println!("  Reached goal in {} steps!", step + 1);
                    println!("  Path: {:?}", path);
                } else {
                    println!("  Failed to reach goal in {} steps", step + 1);
                }
                break;
            }
        }
    }
}