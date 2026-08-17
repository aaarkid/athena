# Athena Documentation

Documentation for Athena, a deep learning library for Rust with a focus on reinforcement learning.

## Getting Started
- [Tutorial: Getting Started](tutorial_getting_started.md) - Basic usage and first steps
- [Tutorial: Advanced Features](tutorial_advanced.md) - custom layers, multi-agent, performance

## Guides
- [Algorithms Guide](algorithms_guide.md) - Overview of RL algorithms (DQN, PPO, SAC, etc.)
- [Performance Guide](performance_guide.md) - Optimization tips and benchmarking
- [Best Practices](best_practices.md) - Recommended patterns and practices

## Platform-Specific
- [Windows Setup](WINDOWS_SETUP.md) - Installation guide for Windows users

## GPU Support
Build with `--features gpu` for OpenCL, or `--features gpu-mock` to compile against the
same API without OpenCL installed. See the [Windows Setup](WINDOWS_SETUP.md) guide for
driver notes.

## API Reference
Run `cargo doc --open` to view the full API documentation.

## Examples
See the `examples/` directory for working code samples:
- `grid_navigation.rs` - Basic RL agent navigation
- `simple_benchmark.rs` - Performance benchmarking
- `gpu_acceleration.rs` - GPU acceleration demo
- `cartpole_ppo.rs` - PPO algorithm example
- `pendulum_sac.rs` - SAC for continuous control
- And many more...

## Development
Build with `cargo build`, test with `cargo test`, lint with `cargo clippy`.