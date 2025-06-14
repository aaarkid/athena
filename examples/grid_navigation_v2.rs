/***
# Grid Navigation with Enhanced DQN (Target Network + Double DQN)

* Demonstrates the improved DQN agent with target network and Double DQN
* Same environment as grid_navigation.rs for comparison
***/

use athena::{
    agent_v2::{DqnAgentV2, DqnAgentBuilder},
    replay_buffer::{ReplayBuffer, Experience},
    optimizer::{OptimizerWrapper, SGD},
};
use ndarray::Array1;

const GRID_SIZE: usize = 5;
const STATE_SIZE: usize = GRID_SIZE * GRID_SIZE;
const ACTION_SIZE: usize = 4;
const LEARNING_RATE: f32 = 0.001;
const EPSILON: f32 = 1.0;
const EPSILON_DECAY: f32 = 0.995;
const EPSILON_MIN: f32 = 0.01;
const GAMMA: f32 = 0.95;
const CAPACITY: usize = 10000;
const BATCH_SIZE: usize = 32;
const EPISODES: usize = 300;
const MAX_STEPS: usize = 50;
const TARGET_UPDATE_FREQ: usize = 100;

#[derive(Clone)]
struct GridWorld {
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    steps: usize,
}

impl GridWorld {
    fn new() -> Self {
        Self {
            agent_pos: (0, 0),
            goal_pos: (4, 4),
            steps: 0,
        }
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
            0 => (self.agent_pos.0.saturating_sub(1), self.agent_pos.1),
            1 => ((self.agent_pos.0 + 1).min(GRID_SIZE - 1), self.agent_pos.1),
            2 => (self.agent_pos.0, self.agent_pos.1.saturating_sub(1)),
            3 => (self.agent_pos.0, (self.agent_pos.1 + 1).min(GRID_SIZE - 1)),
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
    println!("Training Grid Navigation Agent with Enhanced DQN...");
    println!("Features: Target Network (update every {} steps) + Double DQN", TARGET_UPDATE_FREQ);
    let start_time = std::time::Instant::now();

    // Create the enhanced DQN agent using builder pattern
    let agent_result = DqnAgentBuilder::new()
        .layer_sizes(&[STATE_SIZE, 64, 32, ACTION_SIZE])
        .epsilon(EPSILON)
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .target_update_freq(TARGET_UPDATE_FREQ)
        .use_double_dqn(true)
        .build();
        
    let mut agent = match agent_result {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to create agent: {}", e);
            return;
        }
    };
    
    // Create replay buffer
    let mut replay_buffer = ReplayBuffer::new(CAPACITY);
    
    // Training loop
    let mut epsilon = EPSILON;
    let mut episode_rewards = Vec::new();
    let mut losses = Vec::new();

    for episode in 0..EPISODES {
        let mut env = GridWorld::new();
        let mut state = env.get_state();
        let mut total_reward = 0.0;
        let mut episode_loss = 0.0;
        let mut loss_count = 0;

        loop {
            // Agent selects action
            let action = match agent.act(state.view()) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("Error selecting action: {}", e);
                    0
                }
            };
            
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
                match agent.train_on_batch(&experiences, GAMMA, LEARNING_RATE) {
                    Ok(loss) => {
                        episode_loss += loss;
                        loss_count += 1;
                    }
                    Err(e) => eprintln!("Training error: {}", e),
                }
            }
            
            state = next_state;
            
            if done {
                break;
            }
        }
        
        episode_rewards.push(total_reward);
        if loss_count > 0 {
            losses.push(episode_loss / loss_count as f32);
        }
        
        // Decay epsilon
        epsilon = (epsilon * EPSILON_DECAY).max(EPSILON_MIN);
        agent.update_epsilon(epsilon);
        
        // Print progress every 50 episodes
        if (episode + 1) % 50 == 0 {
            let avg_reward: f32 = episode_rewards.iter()
                .skip(episode.saturating_sub(49))
                .sum::<f32>() / 50.0;
            let avg_loss: f32 = if !losses.is_empty() {
                losses.iter().skip(losses.len().saturating_sub(50)).sum::<f32>() 
                    / losses.len().min(50) as f32
            } else {
                0.0
            };
            println!("Episode {}: Avg Reward = {:.2}, Avg Loss = {:.4}, Epsilon = {:.3}", 
                     episode + 1, avg_reward, avg_loss, epsilon);
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
            let action = match agent.act(state.view()) {
                Ok(a) => a,
                Err(_) => 0,
            };
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
    
    // Save the trained agent
    match agent.save("grid_navigator_v2.agent") {
        Ok(_) => println!("\nAgent saved successfully!"),
        Err(e) => eprintln!("Failed to save agent: {}", e),
    }
}