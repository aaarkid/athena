//! Pendulum benchmark for continuous action algorithms (SAC, TD3)
//!
//! This is a FAIR test for continuous action algorithms.
//! Unlike CartPole (discrete actions), Pendulum requires continuous torque control.
//!
//! Run with: cargo run --example pendulum_continuous --release

use athena::algorithms::{SACBuilder, SACExperience, TD3Builder, TD3Experience};
use athena::optimizer::{Adam, OptimizerWrapper};
use ndarray::Array1;
use rand::Rng;
use std::f32::consts::PI;
use std::time::Instant;

/// Pendulum environment (continuous action space)
///
/// This is the classic continuous control benchmark:
/// - State: [cos(θ), sin(θ), θ_dot]
/// - Action: torque in [-2, 2]
/// - Goal: Swing up and balance at θ=0
struct Pendulum {
    theta: f32,      // Angle (0 = upright)
    theta_dot: f32,  // Angular velocity
    max_speed: f32,
    max_torque: f32,
    dt: f32,
    steps: usize,
    max_steps: usize,
}

impl Pendulum {
    fn new() -> Self {
        Self {
            theta: 0.0,
            theta_dot: 0.0,
            max_speed: 8.0,
            max_torque: 2.0,
            dt: 0.05,
            steps: 0,
            max_steps: 200,
        }
    }

    fn reset(&mut self) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        // Start hanging down (theta = PI) with small random perturbation
        self.theta = PI + rng.gen_range(-0.1..0.1);
        self.theta_dot = rng.gen_range(-1.0..1.0);
        self.steps = 0;
        self.get_state()
    }

    fn step(&mut self, action: &Array1<f32>) -> (Array1<f32>, f32, bool) {
        // Clamp action to valid range
        let torque = (action[0] * self.max_torque).clamp(-self.max_torque, self.max_torque);

        // Physics constants
        let g = 10.0;
        let m = 1.0;
        let l = 1.0;

        // Compute angular acceleration
        let theta_acc = (3.0 * g / (2.0 * l)) * self.theta.sin() + (3.0 / (m * l * l)) * torque;

        // Update state using Euler integration
        self.theta_dot += theta_acc * self.dt;
        self.theta_dot = self.theta_dot.clamp(-self.max_speed, self.max_speed);
        self.theta += self.theta_dot * self.dt;

        // Normalize angle to [-PI, PI]
        while self.theta > PI {
            self.theta -= 2.0 * PI;
        }
        while self.theta < -PI {
            self.theta += 2.0 * PI;
        }

        self.steps += 1;

        // Reward: encourage upright position (theta=0) and low velocity
        // Standard Pendulum reward from OpenAI Gym
        let angle_cost = self.theta.powi(2);
        let velocity_cost = 0.1 * self.theta_dot.powi(2);
        let action_cost = 0.001 * torque.powi(2);
        let reward = -(angle_cost + velocity_cost + action_cost);

        let done = self.steps >= self.max_steps;

        (self.get_state(), reward, done)
    }

    fn get_state(&self) -> Array1<f32> {
        Array1::from_vec(vec![
            self.theta.cos(),
            self.theta.sin(),
            self.theta_dot / self.max_speed, // Normalize
        ])
    }
}

/// Test SAC on Pendulum
fn test_sac(episodes: usize, verbose: bool) -> f32 {
    let mut env = Pendulum::new();

    // Create SAC agent for continuous control
    let optimizer = OptimizerWrapper::Adam(Adam::default(&[]));
    let mut agent = SACBuilder::new(3, 1) // 3 state dims, 1 action dim
        .hidden_sizes(vec![256, 256])
        .optimizer(optimizer)
        .gamma(0.99)
        .tau(0.005)
        .alpha(0.2)
        .auto_alpha(true)
        .build()
        .unwrap();

    let mut experiences: Vec<SACExperience> = Vec::new();
    let mut episode_rewards = Vec::new();
    let mut best_avg_reward = f32::NEG_INFINITY;

    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;

        for _ in 0..env.max_steps {
            // Get action from policy (with exploration noise when not deterministic)
            let action = agent.act(state.view(), false).unwrap();

            let (next_state, reward, done) = env.step(&action);
            episode_reward += reward;

            experiences.push(SACExperience {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
            });

            state = next_state;
            if done {
                break;
            }
        }

        // Train at end of episode (much faster than per-step)
        if experiences.len() >= 256 {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut batch = experiences.clone();
            batch.shuffle(&mut rng);
            batch.truncate(256);

            // Multiple updates per episode for efficiency
            for _ in 0..5 {
                let _ = agent.update(&batch, 3e-4);
            }

            // Keep buffer size manageable
            if experiences.len() > 10000 {
                experiences.drain(0..5000);
            }
        }

        episode_rewards.push(episode_reward);

        // Track progress
        if episode >= 10 && episode % 10 == 0 {
            let avg_reward: f32 = episode_rewards.iter().rev().take(10).sum::<f32>() / 10.0;
            if avg_reward > best_avg_reward {
                best_avg_reward = avg_reward;
            }
            if verbose {
                println!(
                    "SAC Episode {}: Avg Reward (last 10) = {:.2}, Best = {:.2}",
                    episode, avg_reward, best_avg_reward
                );
            }
        }
    }

    // Return average of last 50 episodes
    let final_avg = if episode_rewards.len() >= 50 {
        episode_rewards.iter().rev().take(50).sum::<f32>() / 50.0
    } else {
        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
    };

    final_avg
}

/// Test TD3 on Pendulum
fn test_td3(episodes: usize, verbose: bool) -> f32 {
    let mut env = Pendulum::new();

    // Create TD3 agent for continuous control
    let optimizer = OptimizerWrapper::Adam(Adam::default(&[]));
    let mut agent = TD3Builder::new(3, 1) // 3 state dims, 1 action dim
        .hidden_sizes(vec![256, 256])
        .optimizer(optimizer)
        .gamma(0.99)
        .tau(0.005)
        .policy_delay(2)
        .action_bounds(-1.0, 1.0) // Normalized actions, scaled by env
        .noise_params(0.2, 0.5, 0.1) // exploration noise, policy noise, noise clip
        .build()
        .unwrap();

    let mut experiences: Vec<TD3Experience> = Vec::new();
    let mut episode_rewards = Vec::new();
    let mut best_avg_reward = f32::NEG_INFINITY;

    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;

        for _ in 0..env.max_steps {
            // Get action from policy (with exploration noise)
            let action = agent.act(state.view(), false).unwrap();

            let (next_state, reward, done) = env.step(&action);
            episode_reward += reward;

            experiences.push(TD3Experience {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
            });

            state = next_state;
            if done {
                break;
            }
        }

        // Train at end of episode (much faster than per-step)
        if experiences.len() >= 256 {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut batch = experiences.clone();
            batch.shuffle(&mut rng);
            batch.truncate(256);

            // Multiple updates per episode for efficiency
            for _ in 0..5 {
                let _ = agent.update(&batch, 3e-4, 3e-4);
            }

            // Keep buffer size manageable
            if experiences.len() > 10000 {
                experiences.drain(0..5000);
            }
        }

        episode_rewards.push(episode_reward);

        // Track progress
        if episode >= 10 && episode % 10 == 0 {
            let avg_reward: f32 = episode_rewards.iter().rev().take(10).sum::<f32>() / 10.0;
            if avg_reward > best_avg_reward {
                best_avg_reward = avg_reward;
            }
            if verbose {
                println!(
                    "TD3 Episode {}: Avg Reward (last 10) = {:.2}, Best = {:.2}",
                    episode, avg_reward, best_avg_reward
                );
            }
        }
    }

    // Return average of last 50 episodes
    let final_avg = if episode_rewards.len() >= 50 {
        episode_rewards.iter().rev().take(50).sum::<f32>() / 50.0
    } else {
        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
    };

    final_avg
}

fn main() {
    println!("=================================================");
    println!("Pendulum Continuous Control Benchmark");
    println!("=================================================");
    println!();
    println!("This is a FAIR test for continuous action algorithms.");
    println!("Pendulum requires continuous torque control [-2, 2].");
    println!("Goal: Swing up and balance (reward approaches 0).");
    println!("Worst case reward: ~-1600 (hanging down, not moving)");
    println!();

    let episodes = 300;

    // Test SAC
    println!("Testing SAC (Soft Actor-Critic)...");
    println!("---------------------------------");
    let start = Instant::now();
    let sac_reward = test_sac(episodes, true);
    let sac_time = start.elapsed();
    println!();

    // Test TD3
    println!("Testing TD3 (Twin Delayed DDPG)...");
    println!("----------------------------------");
    let start = Instant::now();
    let td3_reward = test_td3(episodes, true);
    let td3_time = start.elapsed();
    println!();

    // Summary
    println!("=================================================");
    println!("RESULTS SUMMARY");
    println!("=================================================");
    println!();
    println!("| Algorithm | Final Avg Reward | Time     |");
    println!("|-----------|------------------|----------|");
    println!(
        "| SAC       | {:>16.2} | {:>6.1}s  |",
        sac_reward,
        sac_time.as_secs_f32()
    );
    println!(
        "| TD3       | {:>16.2} | {:>6.1}s  |",
        td3_reward,
        td3_time.as_secs_f32()
    );
    println!();

    // Interpretation
    println!("Interpretation:");
    println!("---------------");
    println!("- Reward near 0: Excellent (balanced upright)");
    println!("- Reward -200 to -500: Good (swinging, learning)");
    println!("- Reward -800 to -1200: Poor (barely learning)");
    println!("- Reward < -1200: Not learning (stuck hanging)");
    println!();

    if sac_reward > -500.0 || td3_reward > -500.0 {
        println!("SUCCESS: At least one algorithm shows good learning!");
    } else if sac_reward > -1000.0 || td3_reward > -1000.0 {
        println!("PARTIAL: Algorithms show some learning, may need more episodes or tuning.");
    } else {
        println!("ISSUE: Algorithms not learning well - may need debugging.");
    }
}
