# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Athena is a high-performance QDN (Quantum Deep Neural) library written in Rust that provides a foundation for building and training deep neural networks with a focus on reinforcement learning. The library includes customizable neural network architectures, DQN agents, replay buffers, and multiple optimizers.

## Development Commands

### Building and Testing
- `cargo build` - Build the project
- `cargo test` - Run all tests
- `cargo test test_name` - Run a specific test
- `cargo check` - Check for compilation errors without building
- `cargo clippy` - Run the Rust linter to catch common mistakes

### Running Examples
- `cargo run --example grid_navigation` - Run the grid navigation example where an agent learns to navigate to a goal

### Documentation
- `cargo doc --open` - Generate and open the documentation

## Architecture

The library is organized into four main modules:

1. **network.rs** - Neural network implementation including:
   - `NeuralNetwork` struct with customizable layers
   - `Layer` struct with weights, biases, and activation functions
   - Activation functions (ReLU, Sigmoid, Tanh, Linear)
   - Forward and backward propagation
   - Training methods

2. **agent.rs** - Reinforcement learning agent implementation:
   - `DqnAgent` struct implementing Deep Q-Network
   - Epsilon-greedy action selection
   - Batch training on experiences

3. **optimizer.rs** - Optimization algorithms:
   - `SGD` (Stochastic Gradient Descent)
   - `Adam` optimizer with momentum
   - `OptimizerWrapper` enum for flexible optimizer selection

4. **replay_buffer.rs** - Experience replay functionality:
   - `ReplayBuffer` struct for storing experiences
   - `Experience` struct containing state, action, reward, next_state
   - Random sampling for batch training

## Key Dependencies

- `ndarray` (0.15) - N-dimensional arrays with rayon parallelization
- `ndarray-rand` (0.14) - Random number generation for arrays
- `rand` (0.8) - Random number generation
- `serde` (1.0) - Serialization/deserialization
- `bincode` (1.3) - Binary encoding

## Commit Guidelines

Commits must be made with files based on scope in the format:
- `fix(NeuralNetwork): Make this change`
- `feat(DqnAgent): Add new feature`
- `refactor(optimizer): Improve performance`

Keep commits simple and human-like without body text or footers.

## Testing Requirements

All changes and new functionality must be tested and verified:
- Run `cargo build` to ensure compilation
- Write unit tests for new functionality
- Run `cargo test` to verify all tests pass
- Test specific functionality with `cargo test test_name`
- Verify examples still work with `cargo run --example grid_navigation`

Tests are located in `src/lib.rs` and cover all major components. Each module has comprehensive unit tests validating functionality.

## Continuous Development Workflow

When working on tasks:
- Always use `/compact` mode during development
- At the end of each TODO, always add these steps:
  1. Update IMPLEMENTATION_STATUS.md with all progress made
  2. Refer to ADVANCED_FEATURES_PLAN.md for next tasks and update the TODO with the next tasks
  3. Continue working on the next task without stopping to explain completion
- The goal is continuous progress through the advanced features plan

## General Development Rules

- Always commit changes following the commit guidelines above
- Keep working continuously through tasks
- Always test and verify progress with `cargo test` and `cargo build`
- Don't stop to explain that you're done - move to the next task
- Refer to ADVANCED_FEATURES_PLAN.md as the source of truth for task order