# Athena Algorithms Guide

This guide provides detailed explanations of the reinforcement learning algorithms implemented in Athena, including their theoretical foundations, implementation details, and practical usage tips.

## Table of Contents

1. [Deep Q-Network (DQN)](#deep-q-network-dqn)
2. [Advantage Actor-Critic (A2C)](#advantage-actor-critic-a2c)
3. [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
4. [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
5. [Twin Delayed DDPG (TD3)](#twin-delayed-ddpg-td3)

## Deep Q-Network (DQN)

### Theory

DQN combines Q-learning with deep neural networks to handle high-dimensional state spaces. The key innovation is using a neural network to approximate the Q-function:

```
Q(s, a; θ) ≈ Q*(s, a)
```

Where `Q*` is the optimal action-value function and `θ` are the network parameters.

### Key Features

1. **Experience Replay**: Stores transitions and samples random minibatches to break correlations
2. **Target Network**: Separate network for computing target values, updated periodically
3. **Double DQN**: Uses online network to select actions and target network to evaluate them

### Implementation in Athena

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, Adam};

// Create DQN agent
let layer_sizes = &[state_dim, 128, 128, action_dim];
let optimizer = OptimizerWrapper::Adam(Adam::new(0.001, 0.9, 0.999, 1e-8));
let agent = DqnAgent::new(
    layer_sizes,
    0.1,        // epsilon (exploration rate)
    optimizer,
    1000,       // target update frequency
    true        // use double DQN
);
```

### Hyperparameters

- **Learning Rate**: 1e-4 to 1e-3
- **Epsilon**: Start at 1.0, decay to 0.01-0.1
- **Target Update**: Every 100-10000 steps
- **Batch Size**: 32-256
- **Replay Buffer**: 10k-1M experiences

### Best Practices

1. Use Double DQN to reduce overestimation bias
2. Decay epsilon gradually (e.g., linear or exponential decay)
3. Clip rewards to [-1, 1] for stability
4. Use Huber loss instead of MSE for robustness

## Advantage Actor-Critic (A2C)

### Theory

A2C is a synchronous variant of the actor-critic algorithm that combines:
- **Actor**: Policy network π(a|s; θ) that selects actions
- **Critic**: Value network V(s; φ) that estimates state values

The advantage function reduces variance:
```
A(s, a) = Q(s, a) - V(s)
```

### Key Features

1. **Generalized Advantage Estimation (GAE)**: Balances bias and variance
2. **Entropy Regularization**: Encourages exploration
3. **Synchronous Updates**: Multiple actors collect experiences in parallel

### Implementation in Athena

```rust
use athena::algorithms::{A2CAgent, A2CBuilder};

let agent = A2CBuilder::new()
    .input_dim(state_dim)
    .action_dim(action_dim)
    .hidden_dims(vec![256, 256])
    .gamma(0.99)           // discount factor
    .n_steps(5)            // steps before update
    .entropy_coeff(0.01)   // exploration bonus
    .value_coeff(0.5)      // value loss weight
    .build()
    .unwrap();
```

### Hyperparameters

- **Learning Rate**: 7e-4 to 3e-3
- **N-steps**: 5-20 steps
- **Entropy Coefficient**: 0.01-0.001
- **Value Coefficient**: 0.5-1.0
- **GAE Lambda**: 0.95-0.99

### Best Practices

1. Use gradient clipping to prevent instability
2. Normalize advantages for stable training
3. Decay entropy coefficient over time
4. Use larger n_steps for better credit assignment

## Proximal Policy Optimization (PPO)

### Theory

PPO improves on vanilla policy gradient by limiting policy updates to prevent destructive large updates:

```
L^CLIP(θ) = min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)
```

Where r_t(θ) is the probability ratio between new and old policies.

### Key Features

1. **Clipped Objective**: Prevents large policy updates
2. **Multiple Epochs**: Reuses collected data efficiently
3. **Value Function Clipping**: Optional additional stability

### Implementation in Athena

```rust
use athena::algorithms::{PPOAgent, PPOBuilder};

let agent = PPOBuilder::new()
    .input_dim(state_dim)
    .action_dim(action_dim)
    .hidden_dims(vec![256, 256])
    .clip_epsilon(0.2)     // clipping parameter
    .n_epochs(10)          // epochs per update
    .minibatch_size(64)    // SGD batch size
    .build()
    .unwrap();
```

### Hyperparameters

- **Learning Rate**: 3e-4
- **Clip Epsilon**: 0.1-0.3
- **Epochs**: 3-10
- **Minibatch Size**: 32-256
- **Horizon**: 128-2048 steps

### Best Practices

1. Use GAE for advantage estimation
2. Normalize observations and rewards
3. Use larger batch sizes for stability
4. Anneal learning rate and clip epsilon

## Soft Actor-Critic (SAC)

### Theory

SAC maximizes both expected reward and policy entropy:

```
J(π) = E[Σ γ^t (r_t + αH(π(·|s_t)))]
```

This encourages exploration and robustness to model errors.

### Key Features

1. **Maximum Entropy Framework**: Balances exploration and exploitation
2. **Twin Q-Networks**: Reduces overestimation like TD3
3. **Automatic Temperature Tuning**: Adapts entropy weight

### Implementation in Athena

```rust
use athena::algorithms::{SACAgent, SACBuilder};

let agent = SACBuilder::new()
    .input_dim(state_dim)
    .action_dim(action_dim)
    .hidden_dims(vec![256, 256])
    .alpha(0.2)            // initial temperature
    .tau(0.005)            // soft update rate
    .auto_alpha(true)      // automatic tuning
    .build()
    .unwrap();
```

### Hyperparameters

- **Learning Rate**: 3e-4 for all networks
- **Alpha**: 0.2 (or auto-tuned)
- **Tau**: 0.005 (soft update)
- **Batch Size**: 256
- **Buffer Size**: 1M

### Best Practices

1. Use automatic alpha tuning
2. Initialize alpha based on action dimension
3. Use same learning rate for all networks
4. Update critic more frequently than actor

## Twin Delayed DDPG (TD3)

### Theory

TD3 addresses overestimation in actor-critic methods through:
1. Twin Q-networks with minimum operation
2. Delayed policy updates
3. Target policy smoothing

### Key Features

1. **Twin Critics**: Takes minimum to reduce overestimation
2. **Delayed Updates**: Policy updated less frequently
3. **Action Noise**: Added to target actions for smoothing

### Implementation in Athena

```rust
use athena::algorithms::{TD3Agent, TD3Builder};

let agent = TD3Builder::new()
    .input_dim(state_dim)
    .action_dim(action_dim)
    .hidden_dims(vec![256, 256])
    .policy_delay(2)       // update actor every 2 critic updates
    .noise_std(0.2)        // target policy noise
    .noise_clip(0.5)       // noise clipping
    .build()
    .unwrap();
```

### Hyperparameters

- **Learning Rate**: 3e-4
- **Policy Delay**: 2
- **Noise Std**: 0.1-0.3
- **Noise Clip**: 0.5
- **Tau**: 0.005

### Best Practices

1. Use exploration noise during training
2. Clip actions to valid range
3. Normalize observations
4. Use larger replay buffers (1M+)

## Algorithm Selection Guide

| Task Type | Recommended Algorithm | Why |
|-----------|---------------------|-----|
| Discrete Actions, Simple | DQN | Fast, stable, well-understood |
| Discrete Actions, Complex | PPO | Better exploration, stable |
| Continuous Control | SAC | Excellent exploration, sample efficient |
| Robotic Control | TD3 | Very stable, handles delays well |
| Multi-Agent | PPO/A2C | Easy to parallelize |
| Sparse Rewards | SAC | Entropy bonus helps exploration |

## Common Implementation Tips

### 1. Observation Normalization
```rust
// Normalize observations to zero mean, unit variance
let normalized = (obs - mean) / (std + 1e-8);
```

### 2. Reward Scaling
```rust
// Scale rewards to reasonable range
let scaled_reward = reward / 100.0;  // environment-specific
```

### 3. Learning Rate Scheduling
```rust
use athena::optimizer::LearningRateScheduler;

let scheduler = LearningRateScheduler::exponential(
    initial_lr: 3e-4,
    decay_rate: 0.99,
    decay_steps: 1000
);
```

### 4. Gradient Clipping
```rust
use athena::optimizer::GradientClipper;

let clipper = GradientClipper::new(max_norm: 0.5);
```

## Debugging Training

### Signs of Issues

1. **Exploding Q-values**: Learning rate too high, use gradient clipping
2. **No Learning**: Exploration too low, learning rate too small
3. **Unstable Training**: Batch size too small, update frequency too high
4. **Slow Learning**: Increase learning rate, decrease target update frequency

### Monitoring Metrics

1. **Episode Reward**: Should increase over time
2. **Loss Values**: Should decrease but may oscillate
3. **Q-values**: Should stabilize, not explode
4. **Entropy**: (PPO/SAC) Should decrease gradually
5. **KL Divergence**: (PPO) Should stay small

## References

1. [DQN Paper](https://www.nature.com/articles/nature14236)
2. [A2C/A3C Paper](https://arxiv.org/abs/1602.01783)
3. [PPO Paper](https://arxiv.org/abs/1707.06347)
4. [SAC Paper](https://arxiv.org/abs/1801.01290)
5. [TD3 Paper](https://arxiv.org/abs/1802.09477)