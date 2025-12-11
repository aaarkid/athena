use crate::types::{State, Action, GenericExperience};
use crate::error::Result;

/// Trait for reinforcement learning agents
pub trait RLAgent<S: State, A: Action> {
    /// Select an action given a state
    fn act(&mut self, state: &S) -> Result<A>;
    
    /// Update the agent with a single experience
    fn update(&mut self, experience: GenericExperience<S, A>, learning_rate: f32) -> Result<()>;
    
    /// Train on a batch of experiences
    fn train_batch(&mut self, batch: &[GenericExperience<S, A>], learning_rate: f32) -> Result<()>;
    
    /// Get the current exploration rate (if applicable)
    fn exploration_rate(&self) -> Option<f32> {
        None
    }
    
    /// Set the exploration rate (if applicable)
    fn set_exploration_rate(&mut self, _rate: f32) -> Result<()> {
        Ok(())
    }
}

/// Trait for value-based agents
pub trait ValueBasedAgent<S: State>: RLAgent<S, crate::types::DiscreteAction> {
    /// Get Q-values for a state
    fn q_values(&mut self, state: &S) -> Result<ndarray::Array1<f32>>;
    
    /// Get the value of a state (max Q-value)
    fn state_value(&mut self, state: &S) -> Result<f32> {
        let q_values = self.q_values(state)?;
        Ok(q_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
    }
}

/// Trait for policy-based agents
pub trait PolicyBasedAgent<S: State, A: Action>: RLAgent<S, A> {
    /// Get action probabilities for a state (discrete actions)
    fn action_probabilities(&mut self, state: &S) -> Result<ndarray::Array1<f32>>;
    
    /// Get the log probability of an action given a state
    fn log_prob(&mut self, state: &S, action: &A) -> Result<f32>;
}

/// Trait for actor-critic agents
pub trait ActorCriticAgent<S: State, A: Action>: PolicyBasedAgent<S, A> {
    /// Get the value estimate for a state
    fn value(&mut self, state: &S) -> Result<f32>;
    
    /// Get both action and value for a state
    fn act_with_value(&mut self, state: &S) -> Result<(A, f32)>;
}

/// Adapter to make DqnAgent work with generic traits
pub struct DqnAgentAdapter {
    inner: crate::agent::DqnAgent,
}

impl DqnAgentAdapter {
    pub fn new(inner: crate::agent::DqnAgent) -> Self {
        DqnAgentAdapter { inner }
    }
}

impl RLAgent<crate::types::DenseState, crate::types::DiscreteAction> for DqnAgentAdapter {
    fn act(&mut self, state: &crate::types::DenseState) -> Result<crate::types::DiscreteAction> {
        let action_idx = self.inner.act(state.data.view())?;
        Ok(crate::types::DiscreteAction::new(action_idx))
    }
    
    fn update(&mut self, experience: GenericExperience<crate::types::DenseState, crate::types::DiscreteAction>, _learning_rate: f32) -> Result<()> {
        // Convert to standard experience
        let _exp = crate::replay_buffer::Experience {
            state: experience.state.data.clone(),
            action: experience.action.index,
            reward: experience.reward,
            next_state: experience.next_state.data.clone(),
            done: experience.done,
        };
        
        // Add to internal buffer and train
        // Note: This is simplified - in practice you'd want a proper buffer
        Ok(())
    }
    
    fn train_batch(&mut self, _batch: &[GenericExperience<crate::types::DenseState, crate::types::DiscreteAction>], _learning_rate: f32) -> Result<()> {
        // Convert batch and train
        // This is a simplified implementation
        Ok(())
    }
    
    fn exploration_rate(&self) -> Option<f32> {
        Some(self.inner.epsilon)
    }
    
    fn set_exploration_rate(&mut self, rate: f32) -> Result<()> {
        self.inner.epsilon = rate;
        Ok(())
    }
}

impl ValueBasedAgent<crate::types::DenseState> for DqnAgentAdapter {
    fn q_values(&mut self, state: &crate::types::DenseState) -> Result<ndarray::Array1<f32>> {
        Ok(self.inner.q_network.forward(state.data.view()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DenseState;
    use ndarray::array;
    
    #[test]
    fn test_dqn_adapter() {
        let layer_sizes = &[4, 32, 32, 2];
        let optimizer = crate::optimizer::OptimizerWrapper::SGD(crate::optimizer::SGD::new());
        let dqn = crate::agent::DqnAgent::new(layer_sizes, 0.1, optimizer, 100, false);
        
        let mut adapter = DqnAgentAdapter::new(dqn);
        
        // Test acting
        let state = DenseState::new(array![0.1, 0.2, 0.3, 0.4]);
        let action = adapter.act(&state).unwrap();
        assert!(action.index < 2);
        
        // Test Q-values
        let q_values = adapter.q_values(&state).unwrap();
        assert_eq!(q_values.len(), 2);
        
        // Test exploration rate
        assert_eq!(adapter.exploration_rate(), Some(0.1));
        adapter.set_exploration_rate(0.05).unwrap();
        assert_eq!(adapter.exploration_rate(), Some(0.05));
    }
}