# Athena Documentation

Documentation for Athena, a deep learning library for Rust with a focus on reinforcement learning.

## Start here
- [Quickstart](quickstart.md) - act, learn, save, reload, in one page
- [Conventions](conventions.md) - shapes, weight orientation, what can be stacked, the
  features table, the known limitations

## Guides
- [Getting Started](tutorial_getting_started.md) - the basics at more length
- [Algorithms Guide](algorithms_guide.md) - DQN, A2C, PPO, SAC and TD3, and how each is called
- [Best Practices](best_practices.md) - what to check when an agent will not learn
- [Advanced Tutorial](tutorial_advanced.md) - writing a layer, multi-agent, partial observability
- [Performance Guide](performance_guide.md) - what costs what

Every Rust sample in the guides above is compiled by `cargo test`, so none of them can
drift from the API.

## Platform-Specific
- [Windows Setup](WINDOWS_SETUP.md) - Installation guide for Windows users

## GPU Support
Build with `--features gpu` for OpenCL, or `--features gpu-mock` to compile against the
same API without OpenCL installed. Under `gpu-mock` every operation runs on the CPU and
`device_info` is fabricated, so its timings mean nothing. See the
[Windows Setup](WINDOWS_SETUP.md) guide for driver notes.

## API Reference
Run `cargo doc --open` to view the full API documentation.

## Examples
- `game_loop_dqn.rs` - the canonical path: act, learn, decay, save, reload
- `background_training.rs` - training on a worker thread, off the frame thread
- `grid_navigation.rs` - DQN on a small grid world
- `cartpole_ppo.rs` - PPO
- `pendulum_sac.rs` - SAC on continuous control
- `conv_shapes.rs` - the conv and pooling backward passes
- `parallel_training.rs` - splitting a batch across cores, with the crossover printed

Run with `cargo run --release --example <name>`.

## Development
`cargo test --lib` is the fast loop. `cargo test` adds the doctests. Do not run
`cargo test --all-targets`: it runs the benches in the debug profile and does not finish.
See the development section of [Conventions](conventions.md).