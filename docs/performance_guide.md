# Athena Performance Guide

This guide provides optimization strategies and performance tips for getting the most out of the Athena reinforcement learning library.

## Table of Contents

1. [General Performance Tips](#general-performance-tips)
2. [Neural Network Optimization](#neural-network-optimization)
3. [Memory Management](#memory-management)
4. [Training Optimization](#training-optimization)
5. [Parallelization Strategies](#parallelization-strategies)
6. [Profiling and Benchmarking](#profiling-and-benchmarking)

## General Performance Tips

### 1. Enable Compiler Optimizations

Always compile with optimizations for production:

```bash
# Development (fast compilation, slow runtime)
cargo build

# Production (slow compilation, fast runtime)
cargo build --release

# Maximum optimization (even slower compilation)
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### 2. Use BLAS Backend

Athena uses ndarray with optional BLAS support for faster matrix operations:

```toml
# In Cargo.toml
[dependencies]
ndarray = { version = "0.15", features = ["blas"] }
blas-src = { version = "0.8", features = ["openblas"] }
```

### 3. Profile-Guided Optimization

Use PGO for additional performance:

```bash
# Step 1: Build with profiling
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" cargo build --release

# Step 2: Run your training workload
./target/release/your_training_app

# Step 3: Build with profile data
RUSTFLAGS="-C profile-use=/tmp/pgo-data" cargo build --release
```

## Neural Network Optimization

### 1. Batch Processing

Always use batch processing instead of single samples:

```rust
// Slow: Process one at a time
for state in states {
    let output = network.forward(state.view());
}

// Fast: Process in batches
let batch = stack_states(&states);  // Shape: [batch_size, input_dim]
let outputs = network.forward_batch(batch.view());
```

### 2. Layer Configuration

Optimize layer sizes for your hardware:

```rust
// Powers of 2 often perform better due to memory alignment
let layer_sizes = &[256, 512, 256, 64];  // Good
let layer_sizes = &[250, 500, 250, 60];  // Less optimal

// Consider your hardware's cache sizes
// L1: 32-64KB, L2: 256KB-1MB, L3: 8-32MB
```

### 3. Activation Functions

Performance comparison (fastest to slowest):
1. ReLU - Simple max operation
2. LeakyReLU - One comparison, one multiply
3. Tanh - One exponential
4. Sigmoid - Two exponentials
5. GELU - Most complex

```rust
// For hidden layers, prefer ReLU for speed
let activations = &[Activation::Relu, Activation::Relu, Activation::Linear];
```

### 4. Weight Initialization

Proper initialization prevents vanishing/exploding gradients:

```rust
use athena::layers::WeightInit;

// For ReLU networks
let layer = Layer::new_with_init(
    input_size, output_size, 
    Activation::Relu, 
    WeightInit::HeUniform
);

// For Tanh/Sigmoid networks
let layer = Layer::new_with_init(
    input_size, output_size,
    Activation::Tanh,
    WeightInit::XavierUniform
);
```

## Memory Management

### 1. Replay Buffer Optimization

Choose the right buffer size and implementation:

```rust
// For simple tasks (< 100K experiences)
let buffer = ReplayBuffer::new(50_000);

// For complex tasks, consider memory vs performance
let buffer = ReplayBuffer::new(1_000_000);

// Use prioritized replay for better sample efficiency
let buffer = PrioritizedReplayBuffer::new(100_000, 0.6, 0.4);
```

### 2. Avoid Unnecessary Clones

```rust
// Bad: Cloning large arrays
let state_copy = state.clone();

// Good: Use views when possible
let state_view = state.view();

// Good: Move ownership when appropriate
buffer.add(Experience {
    state,  // Move, not clone
    action,
    reward,
    next_state,
    done,
});
```

### 3. Pre-allocate Buffers

```rust
// Pre-allocate vectors with known capacity
let mut experiences = Vec::with_capacity(batch_size);

// Pre-allocate arrays for batch processing
let mut batch_states = Array2::<f32>::zeros((batch_size, state_dim));
let mut batch_targets = Array2::<f32>::zeros((batch_size, action_dim));
```

## Training Optimization

### 1. Hyperparameter Tuning

Key parameters that affect performance:

```rust
// Batch size: Larger = more stable but slower per step
let batch_size = 256;  // Good default, try 128-512

// Learning rate: Critical for convergence speed
let learning_rate = 3e-4;  // Good starting point

// Update frequency: Balance stability and speed
let update_frequency = 4;  // Update every 4 steps
```

### 2. Learning Rate Scheduling

Adapt learning rate during training:

```rust
use athena::optimizer::LearningRateScheduler;

// Exponential decay
let scheduler = LearningRateScheduler::exponential(
    initial_lr: 1e-3,
    decay_rate: 0.99,
    decay_steps: 1000,
);

// Step decay
let scheduler = LearningRateScheduler::step(
    initial_lr: 1e-3,
    drop_factor: 0.5,
    drop_every: 10000,
);

// Cosine annealing
let scheduler = LearningRateScheduler::cosine(
    initial_lr: 1e-3,
    min_lr: 1e-5,
    period: 50000,
);
```

### 3. Gradient Clipping

Prevent training instability:

```rust
use athena::optimizer::GradientClipper;

// Clip by norm (recommended)
let clipper = GradientClipper::new(max_norm: 0.5);

// Apply during training
let clipped_gradients = clipper.clip_gradients(&gradients);
```

### 4. Early Stopping

Monitor validation performance:

```rust
let mut best_reward = f32::NEG_INFINITY;
let mut patience_counter = 0;
let patience = 10;

for episode in 0..max_episodes {
    let reward = evaluate_agent(&agent);
    
    if reward > best_reward {
        best_reward = reward;
        patience_counter = 0;
        agent.save("best_model.bin")?;
    } else {
        patience_counter += 1;
        if patience_counter >= patience {
            println!("Early stopping at episode {}", episode);
            break;
        }
    }
}
```

## Parallelization Strategies

### 1. Environment Parallelization

Run multiple environments in parallel:

```rust
use rayon::prelude::*;

// Collect experiences from multiple environments
let experiences: Vec<_> = (0..num_envs)
    .into_par_iter()
    .map(|env_id| {
        let mut env = create_env(env_id);
        let mut local_experiences = vec![];
        
        for _ in 0..steps_per_env {
            let action = agent.act(&state)?;
            let (next_state, reward, done) = env.step(action);
            local_experiences.push(Experience { 
                state, action, reward, next_state, done 
            });
        }
        local_experiences
    })
    .flatten()
    .collect();
```

### 2. Batch Processing Parallelization

Use ndarray's parallel features:

```toml
# In Cargo.toml
[dependencies]
ndarray = { version = "0.15", features = ["rayon"] }
```

```rust
// Parallel matrix operations
use ndarray::parallel::prelude::*;

// Automatically parallelize large operations
let result = &large_matrix.dot(&other_matrix);
```

### 3. Data Loading Pipeline

Implement async data loading:

```rust
use std::sync::mpsc;
use std::thread;

// Create channel for experiences
let (tx, rx) = mpsc::channel();

// Collector thread
thread::spawn(move || {
    loop {
        let experience = collect_experience();
        tx.send(experience).unwrap();
    }
});

// Training thread
for batch in rx.iter().chunks(batch_size) {
    agent.train_batch(&batch)?;
}
```

## Profiling and Benchmarking

### 1. Built-in Benchmarks

Use Athena's benchmark suite:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench network_bench

# Save baseline for comparison
cargo bench -- --save-baseline my_baseline

# Compare against baseline
cargo bench -- --baseline my_baseline
```

### 2. Custom Profiling

Profile your training loop:

```rust
use std::time::Instant;

let mut timings = HashMap::new();

// Profile forward pass
let start = Instant::now();
let output = network.forward_batch(batch.view());
timings.insert("forward", start.elapsed());

// Profile backward pass
let start = Instant::now();
let gradients = network.backward_batch(errors.view());
timings.insert("backward", start.elapsed());

// Print timings
for (name, duration) in &timings {
    println!("{}: {:?}", name, duration);
}
```

### 3. Memory Profiling

Track memory usage:

```rust
#[cfg(feature = "memory-stats")]
fn print_memory_usage() {
    use memory_stats::memory_stats;
    
    if let Some(usage) = memory_stats() {
        println!("Physical memory: {} MB", usage.physical_mem / 1024 / 1024);
        println!("Virtual memory: {} MB", usage.virtual_mem / 1024 / 1024);
    }
}
```

### 4. Flame Graphs

Generate flame graphs for detailed profiling:

```bash
# Install flamegraph
cargo install flamegraph

# Run with profiling
cargo flamegraph --bin your_training_app

# Open the generated flamegraph.svg
```

## Algorithm-Specific Optimizations

### DQN Optimizations

```rust
// Use larger replay buffer for better sample diversity
let buffer_size = 1_000_000;

// Update target network less frequently for stability
let target_update_freq = 10_000;

// Use Double DQN to reduce overestimation
let use_double_dqn = true;
```

### PPO Optimizations

```rust
// Reuse collected data multiple times
let n_epochs = 10;

// Use larger batches for stable updates
let minibatch_size = 256;

// Collect more data before updates
let horizon = 2048;
```

### SAC Optimizations

```rust
// Use automatic temperature tuning
let auto_alpha = true;

// Update critics more frequently
let critic_update_freq = 1;
let actor_update_freq = 2;
```

## Hardware Considerations

### CPU Optimization

1. **SIMD Instructions**: Compile with `-C target-cpu=native`
2. **Thread Affinity**: Pin threads to cores
3. **NUMA Awareness**: Keep data local to cores

### GPU Acceleration (Future)

While Athena currently runs on CPU, GPU support is planned:

```rust
// Future API (not yet implemented)
let device = Device::cuda(0);
let network = network.to_device(device);
```

## Common Bottlenecks and Solutions

| Bottleneck | Symptom | Solution |
|-----------|---------|----------|
| Replay Buffer Sampling | Slow training step | Use smaller buffer or optimize sampling |
| Network Forward Pass | Low steps/second | Increase batch size, simplify network |
| Environment Step | Low environment FPS | Parallelize environments |
| Memory Allocation | Frequent GC pauses | Pre-allocate buffers |
| Gradient Computation | Slow backward pass | Use gradient checkpointing |

## Performance Checklist

- [ ] Compile with `--release` flag
- [ ] Use batch processing everywhere
- [ ] Enable BLAS backend
- [ ] Profile before optimizing
- [ ] Monitor memory usage
- [ ] Use appropriate data structures
- [ ] Parallelize where possible
- [ ] Tune hyperparameters
- [ ] Consider hardware limitations
- [ ] Benchmark regularly

## References

1. [Rust Performance Book](https://nnethercote.github.io/perf-book/)
2. [ndarray Performance Tips](https://docs.rs/ndarray/latest/ndarray/)
3. [Rayon Parallelization](https://github.com/rayon-rs/rayon)
4. [BLAS Performance](http://www.netlib.org/blas/)