# Deep Research Prompts for RL Algorithm Implementation

These prompts are designed for deep research into the mathematical foundations needed to implement production-ready RL algorithms.

---

## 1. SAC (Soft Actor-Critic) - CRITICAL PRIORITY

### Research Prompt 1: Reparameterization Trick for SAC

```
I'm implementing Soft Actor-Critic (SAC) for continuous control in Rust. My current implementation doesn't learn effectively on the Pendulum task (achieves -1550 vs random -1600).

I need a deep mathematical explanation of the REPARAMETERIZATION TRICK for SAC:

1. **The Core Problem**:
   - SAC objective: maximize E[Q(s,a) - α * log π(a|s)]
   - Why can't we directly backpropagate through the sampling operation a ~ π(·|s)?
   - What is the "score function" gradient estimator and why does it have high variance?

2. **Reparameterization Solution**:
   - Explain how a = μ(s) + σ(s) * ε (where ε ~ N(0,1)) allows gradient flow
   - Derive the gradient ∇_θ E[Q(s, a_θ(s,ε))] step by step
   - How does the Jacobian of the tanh squashing function factor in?

3. **Implementation Details**:
   - Exact formula for the actor loss gradient with respect to network parameters
   - How to handle the log probability gradient with tanh squashing
   - The correct way to compute: ∇_θ [α * log π(a|s) - Q(s,a)]

4. **Numerical Stability**:
   - Clamping strategies for log_std (why -20 to 2?)
   - Handling atanh for actions near ±1
   - Preventing gradient explosion in early training

Please provide:
- Complete mathematical derivations
- Pseudocode for the actor update step
- Common implementation pitfalls and how to avoid them
```

### Research Prompt 2: SAC Entropy and Temperature Tuning

```
I need to understand SAC's entropy-regularized objective and automatic temperature tuning:

1. **Maximum Entropy Framework**:
   - Derive the soft Bellman equation: Q(s,a) = r + γ E[V(s')] where V(s) = E[Q(s,a) - α log π(a|s)]
   - Why does maximizing entropy improve exploration and robustness?
   - What is the relationship between α (temperature) and exploration?

2. **Automatic Temperature Tuning**:
   - Derive the dual optimization problem for α
   - The loss function: L(α) = E[-α * (log π(a|s) + H_target)]
   - How to set the target entropy H_target (why -dim(A) is common?)
   - Gradient update for log(α) vs α directly

3. **Practical Questions**:
   - Should α be learned per-dimension or globally?
   - How does entropy change during training and what does it indicate?
   - Debugging: what does it mean if entropy collapses to zero?

Provide mathematical derivations and implementation pseudocode.
```

---

## 2. TD3 (Twin Delayed DDPG) - HIGH PRIORITY

### Research Prompt 3: Deterministic Policy Gradient

```
I'm implementing TD3 in Rust. My search-based actor update shows learning (+26% vs random) but isn't optimal.

Explain the DETERMINISTIC POLICY GRADIENT theorem:

1. **DPG Theorem**:
   - For deterministic policy μ(s), derive: ∇_θ J = E[∇_a Q(s,a)|_{a=μ(s)} * ∇_θ μ(s)]
   - Why is this lower variance than stochastic policy gradients?
   - How does this differ from REINFORCE-style gradients?

2. **Chain Rule Application**:
   - The actor update: θ_actor += α * ∇_θ μ(s) * ∇_a Q(s, μ(s))
   - Step-by-step: how gradients flow from Q-network through action to actor
   - Why do we maximize Q (ascent) not minimize (descent)?

3. **TD3 Specific Improvements**:
   - Twin critics: why use min(Q1, Q2) for target but Q1 for actor gradient?
   - Delayed policy updates: mathematical justification for updating actor less frequently
   - Target policy smoothing: why add noise to target actions?

4. **Implementation**:
   - Exact gradient computation for actor network
   - How to implement ∇_a Q when Q network takes [state, action] as input
   - Handling action bounds with tanh and the gradient implications

Provide complete derivations and numpy/pseudocode for the actor update.
```

### Research Prompt 4: TD3 Overestimation Bias

```
Explain the overestimation bias problem in actor-critic methods and how TD3 addresses it:

1. **The Problem**:
   - Why does Q-learning overestimate values?
   - How does this compound in continuous action spaces?
   - Mathematical analysis: E[max(Q1, Q2)] >= max(E[Q1], E[Q2])

2. **Clipped Double Q-Learning**:
   - Derive why y = r + γ * min(Q1_target, Q2_target) reduces overestimation
   - When does this introduce underestimation bias? Is that better?
   - Empirical evidence: how much does this improve learning?

3. **Delayed Policy Updates**:
   - Why does updating the actor every d steps help?
   - Relationship to target network update frequency
   - Optimal values for d (why 2 is common?)

Provide mathematical analysis and experimental insights.
```

---

## 3. PPO (Proximal Policy Optimization) - MEDIUM PRIORITY

### Research Prompt 5: PPO Clipped Objective Deep Dive

```
My PPO implementation scores 21.87 on CartPole (should be 195+). I need to understand the mathematics deeply:

1. **Trust Region Motivation**:
   - What is TRPO's constraint: D_KL(π_old || π_new) <= δ?
   - Why is this constraint important for stable learning?
   - How does PPO approximate this with clipping?

2. **PPO-Clip Objective**:
   - L^CLIP = E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]
   - Where r(θ) = π(a|s) / π_old(a|s)
   - Derive the gradient of this objective
   - When does the clipping activate? (positive vs negative advantage)

3. **Gradient Computation**:
   - For discrete actions: ∇_θ log π(a|s) = one_hot(a) - softmax(logits)
   - Complete gradient: ∇_θ L^CLIP accounting for clipping
   - Why multiply by ratio in gradient? (importance sampling)

4. **GAE (Generalized Advantage Estimation)**:
   - Derive: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
   - Where δ_t = r_t + γV(s_{t+1}) - V(s_t)
   - How does λ trade off bias vs variance?
   - Implementation: compute GAE backwards through trajectory

5. **Hyperparameter Sensitivity**:
   - ε (clip range): typical values 0.1-0.3, why?
   - Number of epochs: why multiple passes through data?
   - Minibatch size effects on gradient variance

Provide derivations, gradient formulas, and debugging checklist.
```

---

## 4. General Policy Gradient Mathematics

### Research Prompt 6: Policy Gradient Foundations

```
Provide a unified mathematical treatment of policy gradients for RL:

1. **The Policy Gradient Theorem**:
   - Derive: ∇_θ J(θ) = E[∑_t ∇_θ log π(a_t|s_t) * G_t]
   - Where G_t is return from time t
   - Prove this from first principles (likelihood ratio trick)

2. **Variance Reduction**:
   - Baseline subtraction: why E[∇ log π * b(s)] = 0?
   - Actor-critic: using V(s) as baseline
   - Advantage function: A(s,a) = Q(s,a) - V(s)

3. **On-Policy vs Off-Policy**:
   - Why is vanilla policy gradient on-policy?
   - Importance sampling for off-policy: π(a|s)/μ(a|s)
   - The "deadly triad" in off-policy learning

4. **Continuous vs Discrete Actions**:
   - Discrete: ∇_θ log π = one_hot - softmax
   - Continuous Gaussian: ∇_θ log N(a; μ, σ)
   - Derive both gradient formulas explicitly

This is foundational - I need to understand this deeply before implementing any policy gradient method.
```

---

## 5. Neural Network Considerations for RL

### Research Prompt 7: Network Architecture for RL

```
I'm implementing RL algorithms with neural networks in Rust. Help me understand the architecture considerations:

1. **Actor Network Design**:
   - Why separate actor and critic networks? When to share layers?
   - Output layer for continuous actions: why tanh? alternatives?
   - For SAC: outputting mean AND log_std vs separate networks

2. **Critic Network Design**:
   - Q(s,a): concatenate state and action at input? or after first layer?
   - TD3's twin critics: share any layers or completely separate?
   - Value network V(s) vs Q-network Q(s,a) tradeoffs

3. **Initialization**:
   - Why is initialization crucial in RL?
   - Orthogonal initialization vs Xavier/He
   - Final layer initialization: small weights, why?

4. **Gradient Flow Issues**:
   - Vanishing gradients in deep critics
   - Gradient clipping strategies
   - Layer normalization vs batch normalization in RL

5. **Numerical Stability**:
   - Float32 vs Float64 for RL
   - When do NaN/Inf appear and how to prevent?
   - Safe log and exp implementations
```

---

## 6. Debugging and Diagnostics

### Research Prompt 8: RL Debugging Methodology

```
My RL implementations compile and run but don't learn well. Provide a systematic debugging methodology:

1. **Sanity Checks**:
   - Can the Q-network fit random Q-values? (supervised learning test)
   - Does the actor loss decrease when training on fixed data?
   - Is reward scaling appropriate for your task?

2. **Diagnostic Metrics**:
   - What should Q-values look like during training?
   - Expected entropy curve for SAC
   - Policy gradient magnitude: too high? too low?
   - Value function prediction accuracy

3. **Common Failure Modes**:
   - Q-values exploding (positive feedback loop)
   - Policy collapsing to deterministic
   - Critic not fitting well
   - Advantage estimates all same sign

4. **Hyperparameter Debugging**:
   - Learning rate: signs it's too high vs too low
   - Batch size effects
   - Target network update frequency
   - Replay buffer size

5. **Environment Issues**:
   - Reward scaling and normalization
   - State normalization importance
   - Episode length considerations

Provide a debugging checklist with specific metrics to monitor.
```

---

## How to Use These Prompts

1. **For Deep Research**: Use with Claude, GPT-4, or Perplexity to get comprehensive mathematical explanations
2. **For Implementation**: Extract pseudocode and convert to Rust
3. **For Debugging**: Use the diagnostic prompts when algorithms aren't learning

## Priority Order

1. **SAC Reparameterization** (Prompt 1) - Critical, current implementation doesn't work
2. **TD3 Deterministic Policy Gradient** (Prompt 3) - High, shows promise but needs proper gradients
3. **PPO Clipped Objective** (Prompt 5) - Medium, algorithm works but underperforms
4. **Foundations** (Prompt 6) - Do this if confused about basics
5. **Architecture/Debugging** (Prompts 7-8) - Reference as needed

## Expected Outcomes After Research

After deeply understanding these topics, you should be able to:
- Implement SAC with proper reparameterization (expect -200 to -500 on Pendulum)
- Implement TD3 with deterministic policy gradient (expect -200 to -400 on Pendulum)
- Tune PPO to solve CartPole (expect 195+ average reward)
