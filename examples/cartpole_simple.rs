//! Simple CartPole Example that Works
//! 
//! This demonstrates a DQN agent learning to balance a pole.

use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::replay_buffer::{ReplayBuffer, Experience};
use ndarray::{Array1, array};
use std::f32::consts::PI;

/// Simple CartPole Environment
struct CartPole {
    x: f32,          // cart position
    x_dot: f32,      // cart velocity
    theta: f32,      // pole angle (radians)
    theta_dot: f32,  // pole angular velocity
    
    // Constants
    gravity: f32,
    mass_cart: f32,
    mass_pole: f32,
    length: f32,
    force_mag: f32,
    dt: f32,
}

impl CartPole {
    fn new() -> Self {
        CartPole {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            length: 0.5,
            force_mag: 10.0,
            dt: 0.02,
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        // Small random initialization
        self.x = (rand::random::<f32>() - 0.5) * 0.1;
        self.x_dot = (rand::random::<f32>() - 0.5) * 0.1;
        self.theta = (rand::random::<f32>() - 0.5) * 0.1;
        self.theta_dot = (rand::random::<f32>() - 0.5) * 0.1;
        
        array![self.x, self.x_dot, self.theta, self.theta_dot]
    }
    
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        // Apply force left or right
        let force = if action == 0 { -self.force_mag } else { self.force_mag };
        
        // Physics simulation
        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();
        let total_mass = self.mass_cart + self.mass_pole;
        let pole_mass_length = self.mass_pole * self.length;
        
        let temp = (force + pole_mass_length * self.theta_dot * self.theta_dot * sin_theta) / total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp)
            / (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / total_mass));
        let x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;
        
        // Update state
        self.x += self.dt * self.x_dot;
        self.x_dot += self.dt * x_acc;
        self.theta += self.dt * self.theta_dot;
        self.theta_dot += self.dt * theta_acc;
        
        // Check termination
        let done = self.x.abs() > 2.4 || self.theta.abs() > PI / 6.0;
        
        // Reward is 1 for each step the pole is balanced
        let reward = if done { 0.0 } else { 1.0 };
        
        let state = array![self.x, self.x_dot, self.theta, self.theta_dot];
        
        (state, reward, done)
    }
}

fn main() {
    println!("CartPole DQN Example");
    println!("===================\n");
    
    // Create DQN agent
    let state_size = 4;
    let action_size = 2;  // left or right
    let layer_sizes = &[state_size, 24, 24, action_size];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    
    let mut agent = DqnAgent::new(
        layer_sizes,
        1.0,   // Start with full exploration
        optimizer,
        100,   // Update target network frequently
        true,  // Use double DQN
    );
    
    // Create replay buffer
    let mut replay_buffer = ReplayBuffer::new(2000);
    
    // Training parameters
    let max_episodes = 500;
    let batch_size = 32;
    let learning_rate = 0.01;
    let gamma = 0.95;
    
    let mut env = CartPole::new();
    let mut episode_lengths = Vec::new();
    let mut solved = false;
    
    println!("Training... (goal: average 195 steps over 100 episodes)");
    
    for episode in 0..max_episodes {
        let mut state = env.reset();
        let mut steps = 0;
        
        loop {
            // Select action
            let action = agent.act(state.view()).unwrap();
            
            // Take step
            let (next_state, reward, done) = env.step(action);
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
            
            if done || steps >= 500 {
                break;
            }
        }
        
        episode_lengths.push(steps);
        
        // Decay epsilon
        agent.epsilon = (agent.epsilon * 0.995).max(0.01);
        
        // Check if solved
        if episode_lengths.len() >= 100 {
            let recent_avg: f32 = episode_lengths
                .iter()
                .skip(episode_lengths.len() - 100)
                .sum::<usize>() as f32 / 100.0;
            
            if recent_avg >= 195.0 && !solved {
                solved = true;
                println!("\nSolved at episode {}! Average steps: {:.1}", 
                         episode, recent_avg);
            }
        }
        
        // Progress update
        if (episode + 1) % 50 == 0 {
            let recent_avg: f32 = episode_lengths
                .iter()
                .skip(episode_lengths.len().saturating_sub(50))
                .sum::<usize>() as f32 / 50.0f32.min(episode_lengths.len() as f32);
            
            println!("Episode {}: avg_steps={:.1}, epsilon={:.3}", 
                     episode + 1, recent_avg, agent.epsilon);
        }
        
        if solved && episode > 150 {
            break;
        }
    }
    
    // Test the trained agent
    println!("\nTesting trained agent (10 episodes)...");
    agent.epsilon = 0.0;  // No exploration
    
    let mut test_lengths = Vec::new();
    
    for test in 0..10 {
        let mut state = env.reset();
        let mut steps = 0;
        
        for _ in 0..500 {
            let action = agent.act(state.view()).unwrap();
            let (next_state, _reward, done) = env.step(action);
            steps += 1;
            state = next_state;
            
            if done {
                break;
            }
        }
        
        println!("Test {}: {} steps", test + 1, steps);
        test_lengths.push(steps);
    }
    
    let avg_test_length: f32 = test_lengths.iter().sum::<usize>() as f32 / test_lengths.len() as f32;
    println!("\nAverage test episode length: {:.1} steps", avg_test_length);
    
    if avg_test_length >= 195.0 {
        println!("Success! Agent learned to balance the pole.");
    } else {
        println!("Agent needs more training.");
    }
}