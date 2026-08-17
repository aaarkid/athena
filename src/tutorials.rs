//! # Athena Tutorials and Guides
//! 
//! Tutorials, guides and examples for getting started with the library.
//! 
//! ## Available Tutorials
//! 
//! - [Getting Started](getting_started) - Basic usage and first steps
//! - [Advanced Features](advanced) - Deep dive into advanced capabilities
//! - [Best Practices](best_practices) - Recommended patterns
//! - [Performance Guide](performance) - Optimization tips
//! - [Algorithm Guide](algorithms) - Overview of RL algorithms
//! 
//! ## Quick Navigation
//! 
//! ### For Beginners
//! Start with the [getting_started] module to learn:
//! - How to create your first agent
//! - Basic training loops
//! - Evaluating performance
//! 
//! ### For Advanced Users
//! Check out [advanced] for:
//! - Custom layer implementations
//! - Advanced training techniques
//! - Model export and deployment
//! 
//! ### For Contributors
//! See [best_practices] for:
//! - Code organization guidelines
//! - Testing strategies
//! - Performance considerations

/// Getting Started Tutorial
/// 
/// An introduction to Athena.
pub mod getting_started {
    //! # Getting Started with Athena
    //! 
    //! This tutorial will guide you through the basics of using Athena for reinforcement learning.
    //! 
    //! ## Installation
    //! 
    //! First, add Athena to your `Cargo.toml`:
    //! 
    //! ```toml
    //! [dependencies]
    //! athena = "0.1.0"
    //! ndarray = "0.15"
    //! rand = "0.8"
    //! ```
    //! 
    //! ## Your First Agent
    //! 
    //! Let's create a simple DQN agent to solve a basic grid world environment:
    //! 
    //! ```rust
    //! use athena::agent::DqnAgent;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! use athena::replay_buffer::{ReplayBuffer, Experience};
    //! use ndarray::{Array1, array};
    //! 
    //! fn main() -> Result<(), Box<dyn std::error::Error>> {
    //!     // Define the problem dimensions
    //!     let state_dim = 4;      // e.g., agent x, y, goal x, y
    //!     let action_dim = 4;     // up, down, left, right
    //!     
    //!     // Create a neural network architecture
    //!     let layer_sizes = &[state_dim, 32, 32, action_dim];
    //!     
    //!     // Choose an optimizer
    //!     let optimizer = OptimizerWrapper::SGD(SGD::new());
    //!     
    //!     // Create the DQN agent
    //!     let mut agent = DqnAgent::new(
    //!         layer_sizes,
    //!         0.1,        // epsilon (exploration rate)
    //!         optimizer,
    //!         1000,       // target network update frequency
    //!         true        // use Double DQN
    //!     );
    //!     
    //!     // Create a replay buffer
    //!     let mut replay_buffer = ReplayBuffer::new(10000);
    //!     
    //!     // Training loop. 30 episodes is enough to show the shape of it and keeps
    //!     // this doctest fast; a real run needs hundreds. examples/game_loop_dqn.rs
    //!     // is the runnable version.
    //!     for episode in 0..30 {
    //!         let mut state = array![0.0, 0.0, 4.0, 4.0]; // start at (0,0), goal at (4,4)
    //!         let mut total_reward = 0.0;
    //!         
    //!         for step in 0..40 {
    //!             // Select action
    //!             let action = agent.act(state.view())?;
    //!             
    //!             // Simulate environment step (you would replace this with your env)
    //!             let (next_state, reward, done) = step_environment(&state, action);
    //!             
    //!             // Store experience
    //!             replay_buffer.add(Experience {
    //!                 state: state.clone(),
    //!                 action,
    //!                 reward,
    //!                 next_state: next_state.clone(),
    //!                 done,
    //!             });
    //!             
    //!             // Train when enough experiences
    //!             if replay_buffer.len() >= 32 {
    //!                 let batch = replay_buffer.sample(32);
    //!                 // (experiences, gamma, learning_rate)
    //!                 agent.train_on_batch(&batch, 0.99, 0.001)?;
    //!             }
    //!             
    //!             total_reward += reward;
    //!             state = next_state;
    //!             
    //!             if done { break; }
    //!         }
    //!         
    //!         let _ = (episode, total_reward);
    //!     }
    //!     
    //!     Ok(())
    //! }
    //! 
    //! fn step_environment(state: &Array1<f32>, action: usize) -> (Array1<f32>, f32, bool) {
    //!     // Dummy environment logic
    //!     // In real use, this would be your environment
    //!     let mut next_state = state.clone();
    //!     
    //!     // Apply action (0=up, 1=down, 2=left, 3=right)
    //!     match action {
    //!         0 => next_state[1] += 1.0,
    //!         1 => next_state[1] -= 1.0,
    //!         2 => next_state[0] -= 1.0,
    //!         3 => next_state[0] += 1.0,
    //!         _ => {}
    //!     }
    //!     
    //!     // Calculate reward
    //!     let distance = ((next_state[0] - next_state[2]).powi(2) + 
    //!                     (next_state[1] - next_state[3]).powi(2)).sqrt();
    //!     let reward = -distance / 10.0; // Negative distance as reward
    //!     
    //!     // Check if done (reached goal)
    //!     let done = distance < 0.5;
    //!     
    //!     (next_state, reward, done)
    //! }
    //! ```
    //! 
    //! ## Understanding the Components
    //! 
    //! ### Neural Network Architecture
    //! 
    //! The network architecture is defined by layer sizes:
    //! - Input layer: Size matches your state dimension
    //! - Hidden layers: Typically 128-512 units
    //! - Output layer: Size matches number of actions
    //! 
    //! ### Exploration vs Exploitation
    //! 
    //! The epsilon parameter controls exploration:
    //! - High epsilon (e.g., 1.0): More random actions
    //! - Low epsilon (e.g., 0.01): More greedy actions
    //! - Typically decay epsilon during training
    //! 
    //! ### Experience Replay
    //! 
    //! Experience replay helps with:
    //! - Breaking correlation between consecutive samples
    //! - Reusing past experiences
    //! - Stabilizing training
    //! 
    //! ## Training Tips
    //! 
    //! 1. **Start with high epsilon**: Begin with epsilon=1.0 and decay it
    //! 2. **Use adequate buffer size**: At least 10,000 experiences
    //! 3. **Batch size**: 32-64 works well for most problems
    //! 4. **Learning rate**: Start with 0.001 and adjust if needed
    //! 5. **Target network updates**: Every 1000-10000 steps
    //! 
    //! ## Next Steps
    //! 
    //! - Try different network architectures
    //! - Implement epsilon decay
    //! - Add visualization of training progress
    //! - Experiment with other algorithms (PPO, SAC)
}

/// Advanced Features Tutorial
pub mod advanced {
    //! # Advanced Features in Athena
    //! 
    //! This guide covers advanced features and techniques for experienced users.
    //! 
    //! ## Custom Layers
    //! 
    //! Implement custom layers by following the Layer trait:
    //! 
    //! ```rust,no_run
    //! use athena::layers::traits::Layer;
    //! use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    //! 
    //! pub struct CustomLayer {
    //!     weights: Array2<f32>,
    //!     biases: Array1<f32>,
    //! }
    //! 
    //! // Implementation would include all required trait methods
    //! // See DenseLayer for a complete example
    //! ```
    //! 
    //! ## Advanced Training Techniques
    //! 
    //! ### Learning Rate Scheduling
    //! 
    //! ```rust
    //! use athena::optimizer::LearningRateScheduler;
    //! 
    //! let scheduler = LearningRateScheduler::CosineAnnealing {
    //!     max_lr: 0.001,
    //!     min_lr: 0.0001,
    //!     period: 1000,
    //! };
    //! ```
    //! 
    //! ### Gradient Clipping
    //! 
    //! Gradient clipping is implemented as part of the optimizer:
    //! 
    //! ```rust
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! 
    //! // Create optimizer with gradient clipping
    //! let mut optimizer = OptimizerWrapper::SGD(SGD::new());
    //! // Gradient clipping is applied during optimization
    //! ```
    //! 
    //! ## Parallel Training
    //!
    //! Split a large batch across CPU cores. Worth it from a few hundred rows upward:
    //! below roughly a hundred, rayon's handoff costs more than the work, and for one
    //! action per frame `NeuralNetwork::predict` on a single thread is faster.
    //!
    //! ```rust
    //! use athena::parallel::ParallelNetwork;
    //! use athena::network::NeuralNetwork;
    //! use athena::activations::Activation;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! use ndarray::Array2;
    //!
    //! let network = NeuralNetwork::new(&[784, 128, 10], &[Activation::Relu, Activation::Linear], OptimizerWrapper::SGD(SGD::new()));
    //!
    //! // Borrows the network: every thread reads the same weights
    //! let parallel_net = ParallelNetwork::from_network(&network, 4);
    //! let inputs = Array2::zeros((1024, 784));
    //! let outputs = parallel_net.forward_batch_parallel(inputs.view());
    //! assert_eq!(outputs.dim(), (1024, 10));
    //! ```
    //!
    //! Gradients for a batch, then applied:
    //!
    //! ```rust
    //! use athena::parallel::ParallelGradients;
    //! use athena::network::NeuralNetwork;
    //! use athena::activations::Activation;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! use ndarray::Array2;
    //!
    //! let mut network = NeuralNetwork::new(&[8, 32, 4], &[Activation::Relu, Activation::Linear], OptimizerWrapper::SGD(SGD::new()));
    //! let inputs = Array2::from_shape_fn((512, 8), |(i, j)| ((i + j) as f32 * 0.01).sin());
    //! let targets = Array2::zeros((512, 4));
    //!
    //! // The same numbers backward_batch would give, computed across threads
    //! let (weight_grads, bias_grads) =
    //!     ParallelGradients::compute_batch_gradients(&network, inputs.view(), targets.view());
    //! let pairs: Vec<_> = weight_grads.into_iter().zip(bias_grads).collect();
    //! network.apply_gradients(pairs, 0.01);
    //! ```
    //! 
    //! ## GPU Acceleration
    //! 
    //! Enable GPU support for significant speedups:
    //! 
    //! ```rust
    //! #[cfg(feature = "gpu")]
    //! use athena::layers::GpuDenseLayer;
    //! #[cfg(feature = "gpu")]
    //! use athena::activations::Activation;
    //! 
    //! #[cfg(feature = "gpu")]
    //! let gpu_layer = GpuDenseLayer::new(512, 256, Activation::Relu).unwrap();
    //! ```
    //! 
    //! ## Model Export
    //! 
    //! Export models for deployment:
    //! 
    //! ```rust,no_run
    //! use athena::export::json::NetworkExporter;
    //! use athena::network::NeuralNetwork;
    //! use athena::activations::Activation;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! use std::path::Path;
    //! 
    //! // Create a simple network
    //! let network = NeuralNetwork::new(&[784, 128, 10], &[Activation::Relu, Activation::Linear], OptimizerWrapper::SGD(SGD::new()));
    //! NetworkExporter::export(&network, Path::new("model.txt")).unwrap();
    //! ```
}

/// Best Practices Guide
pub mod best_practices {
    //! # Best Practices for Athena
    //! 
    //! This guide provides recommendations for getting the most out of Athena.
    //! 
    //! ## Code Organization
    //! 
    //! ### Project Structure
    //! ```text
    //! my_rl_project/
    //! ├── src/
    //! │   ├── main.rs
    //! │   ├── environments/
    //! │   │   └── my_env.rs
    //! │   ├── agents/
    //! │   │   └── custom_agent.rs
    //! │   └── utils/
    //! │       └── visualization.rs
    //! ├── configs/
    //! │   └── hyperparams.toml
    //! └── Cargo.toml
    //! ```
    //! 
    //! ## Performance Tips
    //! 
    //! ### 1. Use Release Mode
    //! Always compile in release mode for performance:
    //! ```bash
    //! cargo build --release
    //! cargo run --release
    //! ```
    //! 
    //! ### 2. Batch Operations
    //! Process multiple samples at once:
    //! ```rust,no_run
    //! # use athena::network::NeuralNetwork;
    //! # use ndarray::Array2;
    //! # let mut network = NeuralNetwork::new(&[784, 128, 10], &[athena::activations::Activation::Relu, athena::activations::Activation::Linear], athena::optimizer::OptimizerWrapper::SGD(athena::optimizer::SGD::new()));
    //! # let batch_inputs = Array2::zeros((32, 784));
    //! // Good: Process batch
    //! let outputs = network.forward_batch(batch_inputs.view());
    //! ```
    //! 
    //! ### 3. Pre-allocate Buffers
    //! ```rust
    //! use ndarray::Array2;
    //! let batch_size = 32;
    //! let hidden_size = 128;
    //! // Pre-allocate arrays for repeated operations
    //! let mut buffer = Array2::<f32>::zeros((batch_size, hidden_size));
    //! ```
    //! 
    //! ## Common Pitfalls
    //! 
    //! ### 1. Exploding Gradients
    //! - Use gradient clipping
    //! - Check weight initialization
    //! - Reduce learning rate
    //! 
    //! ### 2. Poor Exploration
    //! - Start with high epsilon
    //! - Use epsilon decay schedule
    //! - Consider exploration bonuses
    //! 
    //! ### 3. Unstable Training
    //! - Increase replay buffer size
    //! - Use target networks
    //! - Normalize inputs
}

/// Performance Optimization Guide
pub mod performance {
    //! # Performance Guide for Athena
    //! 
    //! Learn how to optimize your Athena applications for maximum performance.
    //! 
    //! ## Benchmarking
    //! 
    //! Always benchmark before optimizing:
    //! 
    //! ```rust
    //! use std::time::Instant;
    //! 
    //! let start = Instant::now();
    //! // Your code here
    //! let duration = start.elapsed();
    //! println!("Time elapsed: {:?}", duration);
    //! ```
    //! 
    //! ## Memory Optimization
    //! 
    //! ### Use Array Pools
    //!
    //! Hand a buffer back when you are done with it and the next request of that size
    //! reuses it instead of allocating. A returned array comes back zeroed.
    //!
    //! ```rust
    //! use athena::memory_optimization::ArrayPool;
    //!
    //! let mut pool = ArrayPool::new(100);
    //! let array = pool.get_array_1d(1024);
    //! // Use array...
    //! pool.return_array_1d(array);
    //!
    //! // Served from the pool this time
    //! let reused = pool.get_array_1d(1024);
    //! assert!(reused.iter().all(|&v| v == 0.0));
    //! ```
    //!
    //! For a network's own forward pass, `athena::network::InferenceBuffers` already does
    //! this and needs no pool.
    //! 
    //! ### Gradient Accumulation
    //! For large batches that don't fit in memory:
    //! ```rust,no_run
    //! use athena::memory_optimization::GradientAccumulator;
    //! use athena::network::NeuralNetwork;
    //! use athena::activations::Activation;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! 
    //! // Create a network
    //! let network = NeuralNetwork::new(&[784, 128, 10], &[Activation::Relu, Activation::Linear], OptimizerWrapper::SGD(SGD::new()));
    //! let mut accumulator = GradientAccumulator::new(&network);
    //! 
    //! // accumulate() once per mini-batch, then get_gradients() takes the mean over
    //! // those calls and clears. It weights each call equally, so keep the mini-batches
    //! // the same size.
    //! let (avg_weight_grads, avg_bias_grads) = accumulator.get_gradients();
    //! let _ = (avg_weight_grads, avg_bias_grads);
    //! ```
    //! 
    //! ## Parallelization
    //! 
    //! ### Data Parallel Training
    //!
    //! `examples/parallel_training.rs` prints the crossover on your machine. Measured
    //! here, against the same cache-free path on one thread: 1.36x at batch 32, 1.92x at
    //! 256, 7.63x at 2048.
    //!
    //! ```rust
    //! use athena::parallel::ParallelNetwork;
    //! use athena::network::NeuralNetwork;
    //! use athena::activations::Activation;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! use ndarray::Array2;
    //!
    //! let network = NeuralNetwork::new(&[784, 128, 10], &[Activation::Relu, Activation::Linear], OptimizerWrapper::SGD(SGD::new()));
    //! let parallel_net = ParallelNetwork::from_network(&network, 4)
    //!     .with_chunk_rows(128);
    //! let inputs = Array2::zeros((1024, 784));
    //! let outputs = parallel_net.forward_batch_parallel(inputs.view());
    //! assert_eq!(outputs.dim(), (1024, 10));
    //! ```
    //! 
    //! ## GPU Acceleration Tips
    //! 
    //! 1. **Batch Size**: Larger batches utilize GPU better
    //! 2. **Layer Size**: GPUs excel with layers > 256 units
    //! 3. **Memory Transfer**: Minimize CPU-GPU transfers
}

/// Algorithm Selection Guide
pub mod algorithms {
    //! # RL Algorithm Guide
    //! 
    //! This guide helps you choose the right algorithm for your problem.
    //! 
    //! ## Algorithm Comparison
    //! 
    //! | Algorithm | Best For | Pros | Cons |
    //! |-----------|----------|------|------|
    //! | DQN | Discrete actions, off-policy | Sample efficient, stable | Only discrete actions |
    //! | PPO | Continuous/discrete, on-policy | Stable, general purpose | Less sample efficient |
    //! | SAC | Continuous actions | Very stable, high performance | Complex implementation |
    //! | A2C | Simple on-policy | Easy to understand | Less stable than PPO |
    //! | TD3 | Continuous actions | Addresses overestimation | Requires careful tuning |
    //! 
    //! ## DQN (Deep Q-Network)
    //! 
    //! Best for discrete action spaces:
    //! ```rust
    //! use athena::agent::DqnAgent;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! 
    //! let state_dim = 4;
    //! let action_dim = 2;
    //! let optimizer = OptimizerWrapper::SGD(SGD::new());
    //! 
    //! let agent = DqnAgent::new(
    //!     &[state_dim, 128, 128, action_dim],
    //!     0.1,  // epsilon
    //!     optimizer,
    //!     1000, // target update frequency
    //!     true  // double DQN
    //! );
    //! ```
    //! 
    //! ## PPO (Proximal Policy Optimization)
    //! 
    //! Great general-purpose algorithm:
    //! ```rust
    //! use athena::algorithms::ppo::PPOAgent;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! 
    //! let optimizer = OptimizerWrapper::SGD(SGD::new());
    //! let ppo = PPOAgent::new(
    //!     4,         // state_size
    //!     2,         // action_size
    //!     &[64, 64], // hidden_sizes
    //!     optimizer,
    //!     0.99,      // gamma
    //!     0.95,      // gae_lambda
    //!     0.2,       // clip_param
    //!     4          // ppo_epochs
    //! );
    //! ```
    //! 
    //! ## SAC (Soft Actor-Critic)
    //! 
    //! Excellent for continuous control:
    //! ```rust
    //! use athena::algorithms::sac::SACAgent;
    //! use athena::optimizer::{OptimizerWrapper, SGD};
    //! 
    //! let optimizer = OptimizerWrapper::SGD(SGD::new());
    //! let sac = SACAgent::new(
    //!     4,           // state_size
    //!     2,           // action_size
    //!     &[256, 256], // hidden_sizes
    //!     optimizer,
    //!     0.99,        // gamma
    //!     0.005,       // tau (soft update)
    //!     0.2,         // alpha (entropy)
    //!     true         // auto_alpha
    //! );
    //! ```
    //! 
    //! ## Choosing an Algorithm
    //! 
    //! 1. **Action Space**:
    //!    - Discrete: DQN, PPO, A2C
    //!    - Continuous: SAC, TD3, PPO
    //! 
    //! 2. **Sample Efficiency**:
    //!    - High: DQN, SAC, TD3 (off-policy)
    //!    - Low: PPO, A2C (on-policy)
    //! 
    //! 3. **Stability**:
    //!    - Most stable: PPO, SAC
    //!    - Moderate: DQN with target network
    //!    - Requires tuning: A2C, TD3
    //! 
    //! 4. **Implementation Complexity**:
    //!    - Simple: DQN, A2C
    //!    - Moderate: PPO
    //!    - Complex: SAC, TD3
}