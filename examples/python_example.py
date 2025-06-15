"""
Example of using Athena through Python bindings
"""

import numpy as np
from athena import NeuralNetwork, DqnAgent, ReplayBuffer

def test_neural_network():
    """Test basic neural network functionality"""
    print("Testing Neural Network...")
    
    # Create a simple network
    nn = NeuralNetwork(
        layer_sizes=[4, 32, 32, 2],
        activations=["relu", "relu", "linear"],
        optimizer="adam"
    )
    
    # Test forward pass
    input_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    output = nn.forward(input_data)
    print(f"Input: {input_data}")
    print(f"Output: {output}")
    
    # Test batch forward
    batch_input = np.random.randn(10, 4).astype(np.float32)
    batch_output = nn.forward_batch(batch_input)
    print(f"Batch input shape: {batch_input.shape}")
    print(f"Batch output shape: {batch_output.shape}")
    
    # Test training
    targets = np.random.randn(10, 2).astype(np.float32)
    nn.train_minibatch(batch_input, targets, learning_rate=0.01)
    print("Training step completed")

def test_dqn_agent():
    """Test DQN agent functionality"""
    print("\nTesting DQN Agent...")
    
    # Create agent
    agent = DqnAgent(
        state_size=4,
        action_size=2,
        hidden_sizes=[32, 32],
        epsilon=0.1,
        target_update_freq=100,
        use_double_dqn=True
    )
    
    # Create replay buffer
    buffer = ReplayBuffer(capacity=1000)
    
    # Simulate some experiences
    for i in range(100):
        state = np.random.randn(4).astype(np.float32)
        action = agent.act(state)
        reward = np.random.randn()
        next_state = np.random.randn(4).astype(np.float32)
        done = i % 10 == 0
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Replay buffer size: {len(buffer)}")
    print(f"Current epsilon: {agent.epsilon}")
    
    # Train the agent
    if len(buffer) >= 32:
        agent.train(buffer, batch_size=32, learning_rate=0.001)
        print("Training completed")
    
    # Update target network
    agent.update_target_network()
    print("Target network updated")
    
    # Decay epsilon
    agent.decay_epsilon(0.99)
    print(f"Epsilon after decay: {agent.epsilon}")

def test_replay_buffer():
    """Test replay buffer functionality"""
    print("\nTesting Replay Buffer...")
    
    buffer = ReplayBuffer(capacity=100)
    
    # Add some experiences
    for i in range(50):
        state = np.random.randn(4).astype(np.float32)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4).astype(np.float32)
        done = i % 10 == 0
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Is empty: {buffer.is_empty()}")
    
    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(16)
    print(f"Sampled batch:")
    print(f"  States: {len(states)} items")
    print(f"  Actions: {len(actions)} items")
    print(f"  Rewards: {len(rewards)} items")

if __name__ == "__main__":
    print("Athena Python Bindings Example")
    print("=" * 50)
    
    test_neural_network()
    test_dqn_agent()
    test_replay_buffer()
    
    print("\nAll tests completed!")