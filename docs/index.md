# Athena Documentation

Welcome to the Athena deep learning library documentation.

## Getting Started
- [Tutorial: Getting Started](tutorial_getting_started.md) - Basic usage and first steps
- [Tutorial: Advanced Features](tutorial_advanced.md) - Deep dive into advanced capabilities

## Guides
- [Algorithms Guide](algorithms_guide.md) - Overview of RL algorithms (DQN, PPO, SAC, etc.)
- [Performance Guide](performance_guide.md) - Optimization tips and benchmarking
- [Best Practices](best_practices.md) - Recommended patterns and practices

## Platform-Specific
- [Windows Setup](WINDOWS_SETUP.md) - Installation guide for Windows users

## GPU Support
- [Intel Arc GPU Support](gpu/INTEL_ARC_GPU_SUPPORT.md) - Intel Arc GPU acceleration
- [GPU Implementation Summary](gpu/GPU_IMPLEMENTATION_SUMMARY.md) - Technical details
- [GPU Investigation](gpu/GPU_INVESTIGATION.md) - Development notes

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
For contributing and future development plans, see:
- [ADVANCED_FEATURES_PLAN.md](../ADVANCED_FEATURES_PLAN.md) - Roadmap for research features
- [FUTURE_ENHANCEMENTS.md](../FUTURE_ENHANCEMENTS.md) - Planned enhancements
- [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) - Current implementation status