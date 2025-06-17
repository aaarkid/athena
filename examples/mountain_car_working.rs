//! Working Mountain Car Example with DQN
//! 
//! This example properly trains a DQN agent to solve the Mountain Car problem.

use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, Adam};
use athena::replay_buffer::{ReplayBuffer, Experience};
use athena::layers::Layer;
use athena::activations::Activation;
use ndarray::{Array1, array};
use std::time::Instant;

/// Mountain Car Environment
struct MountainCar {
    position: f32,
    velocity: f32,
    min_position: f32,
    max_position: f32,
    max_speed: f32,
    goal_position: f32,
}

impl MountainCar {
    fn new() -> Self {
        MountainCar {
            position: -0.5,
            velocity: 0.0,
            min_position: -1.2,
            max_position: 0.6,
            max_speed: 0.07,
            goal_position: 0.45, // Slightly easier than 0.5
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        self.position = rand::random::<f32>() * 0.2 - 0.6;  // [-0.6, -0.4]
        self.velocity = 0.0;
        array![self.position, self.velocity]
    }
    
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        // Convert discrete action to force: 0=push left, 1=no push, 2=push right
        let force = (action as f32 - 1.0) * 0.001;
        
        // Update velocity
        self.velocity += force - 0.0025 * (3.0 * self.position).cos();
        self.velocity = self.velocity.clamp(-self.max_speed, self.max_speed);
        
        // Update position
        self.position += self.velocity;
        
        // Bounce off walls
        if self.position <= self.min_position {
            self.position = self.min_position;
            self.velocity = 0.0;
        } else if self.position >= self.max_position {
            self.position = self.max_position;
            self.velocity = 0.0;
        }
        
        // Check if goal reached
        let done = self.position >= self.goal_position && self.velocity.abs() < 0.05;
        
        // Reward shaping to help learning
        let reward = if done {
            100.0  // Big reward for reaching goal
        } else {
            // Small reward for height (helps agent learn to build momentum)
            -1.0 + (self.position + 1.2) * 0.1
        };
        
        (array![self.position, self.velocity], reward, done)
    }
}

fn main() {
    println!("Mountain Car DQN Example");
    println!("=======================\n");
    
    // Create DQN agent with Adam optimizer
    let state_size = 2;
    let action_size = 3;  // left, none, right
    let layer_sizes = &[state_size, 128, 128, action_size];
    
    // Create layers for Adam initialization
    let layers: Vec<Layer> = vec![
        Layer::new(state_size, 128, Activation::Relu),
        Layer::new(128, 128, Activation::Relu),
        Layer::new(128, action_size, Activation::Linear),
    ];
    
    let optimizer = OptimizerWrapper::Adam(Adam::new(&layers, 0.9, 0.999, 1e-8));
    
    let mut agent = DqnAgent::new(
        layer_sizes,
        1.0,  // Start with full exploration
        optimizer,
        500,  // Update target network every 500 steps
        true, // Use double DQN
    );
    
    // Create replay buffer
    let mut replay_buffer = ReplayBuffer::new(50000);
    
    // Training parameters
    let max_episodes = 1000;
    let max_steps = 200;
    let batch_size = 64;
    let learning_rate = 0.001;
    let gamma = 0.99;
    let epsilon_decay = 0.995;
    let epsilon_min = 0.01;
    
    let mut env = MountainCar::new();
    let mut episode_rewards = Vec::new();
    let mut solved = false;
    let start_time = Instant::now();
    
    println!("Training...");
    
    for episode in 0..max_episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut steps = 0;
        
        for _ in 0..max_steps {
            // Select action
            let action = agent.act(state.view()).unwrap();
            
            // Take step
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;
            steps += 1;
            
            // Store experience
            replay_buffer.add(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train when we have enough samples
            if replay_buffer.len() >= batch_size {
                let batch = replay_buffer.sample(batch_size);
                agent.train_on_batch(&batch, gamma, learning_rate).unwrap();
            }
            
            state = next_state;
            
            if done {
                println!("Episode {}: Solved in {} steps! Reward: {:.2}", 
                         episode, steps, episode_reward);
                break;
            }
        }
        
        episode_rewards.push(episode_reward);
        
        // Decay epsilon
        agent.epsilon = (agent.epsilon * epsilon_decay).max(epsilon_min);
        
        // Check if consistently solving
        if episode_rewards.len() >= 100 {
            let recent_avg: f32 = episode_rewards
                .iter()
                .skip(episode_rewards.len() - 100)
                .sum::<f32>() / 100.0;
            
            if recent_avg > 90.0 && !solved {
                solved = true;
                println!("\nSolved at episode {}! Average reward: {:.2}", 
                         episode, recent_avg);
            }
        }
        
        // Progress update
        if (episode + 1) % 100 == 0 {
            let recent_avg: f32 = episode_rewards
                .iter()
                .skip(episode_rewards.len().saturating_sub(100))
                .sum::<f32>() / 100.0f32.min(episode_rewards.len() as f32);
            
            println!("Episode {}: avg_reward={:.2}, epsilon={:.3}", 
                     episode + 1, recent_avg, agent.epsilon);
        }
        
        if solved && episode > 200 {
            break;
        }
    }
    
    let training_time = start_time.elapsed();
    println!("\nTraining completed in {:.2}s", training_time.as_secs_f32());
    
    // Test the trained agent
    println!("\nTesting trained agent...");
    agent.epsilon = 0.0;  // No exploration
    
    let mut test_rewards = Vec::new();
    
    for test in 0..10 {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut positions = vec![env.position];
        let mut solved_test = false;
        
        for step in 0..max_steps {
            let action = agent.act(state.view()).unwrap();
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;
            positions.push(env.position);
            state = next_state;
            
            if done {
                solved_test = true;
                println!("Test {}: Solved in {} steps! Final position: {:.3}", 
                         test + 1, step + 1, env.position);
                break;
            }
        }
        
        if !solved_test {
            println!("Test {}: Failed to reach goal. Max position: {:.3}", 
                     test + 1, 
                     positions.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }
        
        test_rewards.push(episode_reward);
    }
    
    let avg_test_reward: f32 = test_rewards.iter().sum::<f32>() / test_rewards.len() as f32;
    println!("\nAverage test reward: {:.2}", avg_test_reward);
    
    if avg_test_reward > 0.0 {
        println!("Success! Agent learned to solve Mountain Car.");
    } else {
        println!("Agent needs more training to consistently solve the task.");
    }
}