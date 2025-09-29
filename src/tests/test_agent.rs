use ndarray::array;
use crate::agent::DqnAgent;
use crate::optimizer::{OptimizerWrapper, SGD};
use crate::replay_buffer::Experience;

fn create_test_agent() -> DqnAgent {
    let layer_sizes = [4, 32, 2];
    let epsilon = 0.5;
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    DqnAgent::new(&layer_sizes, epsilon, optimizer)
}

#[test]
fn test_dqn_agent_new() {
    let agent = create_test_agent();
    assert_eq!(agent.epsilon, 0.5);
    assert_eq!(agent.network.layers.len(), 2);
}

#[test]
fn test_dqn_agent_act() {
    let mut agent = create_test_agent();
    let state = array![0.0, 0.5, 1.0, 0.5];
    let action = agent.act(state.view());
    assert!(action < agent.network.layers.last().unwrap().biases.len());
}

#[test]
fn test_dqn_agent_act_greedy() {
    let mut agent = create_test_agent();
    agent.update_epsilon(0.0); // No exploration
    
    let state = array![0.0, 0.5, 1.0, 0.5];
    let action1 = agent.act(state.view());
    let action2 = agent.act(state.view());
    
    // With epsilon=0, should always choose same action
    assert_eq!(action1, action2);
}

#[test]
fn test_dqn_agent_update_epsilon() {
    let mut agent = create_test_agent();
    agent.update_epsilon(0.1);
    assert_eq!(agent.epsilon, 0.1);
    
    agent.update_epsilon(0.0);
    assert_eq!(agent.epsilon, 0.0);
    
    agent.update_epsilon(1.0);
    assert_eq!(agent.epsilon, 1.0);
}

#[test]
fn test_dqn_agent_train_on_batch() {
    let mut agent = create_test_agent();
    let experience = Experience {
        state: array![0.0, 0.5, 1.0, 0.5],
        action: 0,
        reward: 1.0,
        next_state: array![0.1, 0.6, 0.9, 0.4],
        done: false,
    };
    let gamma = 0.99;
    let learning_rate = 0.001;
    
    // Get Q-values before training
    let q_before = agent.network.forward(experience.state.view()).to_owned();
    
    // Train
    agent.train_on_batch(&[&experience], gamma, learning_rate);
    
    // Get Q-values after training
    let q_after = agent.network.forward(experience.state.view());
    
    // Q-values should have changed
    assert_ne!(q_before, q_after);
}

#[test]
fn test_dqn_agent_train_terminal_state() {
    let mut agent = create_test_agent();
    let experience = Experience {
        state: array![0.0, 0.5, 1.0, 0.5],
        action: 0,
        reward: -1.0,
        next_state: array![0.1, 0.6, 0.9, 0.4],
        done: true, // Terminal state
    };
    let gamma = 0.99;
    let learning_rate = 0.001;
    
    agent.train_on_batch(&[&experience], gamma, learning_rate);
    
    // Training should complete without panic
}

#[test]
fn test_dqn_agent_train_multiple_experiences() {
    let layer_sizes = [2, 32, 2];
    let epsilon = 0.5;
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(&layer_sizes, epsilon, optimizer);
    
    let experiences = vec![
        Experience {
            state: array![0.5, -0.5],
            action: 0,
            reward: 1.0,
            next_state: array![0.6, -0.4],
            done: false,
        },
        Experience {
            state: array![0.6, -0.4],
            action: 1,
            reward: -0.5,
            next_state: array![0.7, -0.3],
            done: false,
        },
    ];
    
    let exp_refs: Vec<&Experience> = experiences.iter().collect();
    let gamma = 0.99;
    let learning_rate = 0.01;
    
    agent.train_on_batch(&exp_refs, gamma, learning_rate);
}

#[test]
fn test_agent_exploration_exploitation() {
    let mut agent = create_test_agent();
    let state = array![0.0, 0.5, 1.0, 0.5];
    
    // With high epsilon, should see variety in actions
    agent.update_epsilon(0.9);
    let mut actions = Vec::new();
    for _ in 0..100 {
        actions.push(agent.act(state.view()));
    }
    
    // Should have both action 0 and 1
    assert!(actions.contains(&0));
    assert!(actions.contains(&1));
}

#[test]
fn test_agent_q_values() {
    let mut agent = create_test_agent();
    let state = array![0.0, 0.5, 1.0, 0.5];
    
    // Get Q-values directly
    let q_values = agent.network.forward(state.view());
    assert_eq!(q_values.len(), 2); // Two actions
    
    // Q-values should be finite
    for &q in q_values.iter() {
        assert!(q.is_finite());
    }
}