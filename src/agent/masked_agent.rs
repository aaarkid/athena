use ndarray::{Array1, ArrayView1};
use rand::Rng;
use crate::agent::dqn::DqnAgent;

/// Extension trait for agents with action masking
pub trait MaskedAgent {
    /// Select action with invalid actions masked out
    fn act_masked(&mut self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> usize;
    
    /// Get Q-values with masking applied
    fn get_masked_q_values(&mut self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> Array1<f32>;
}

/// Implementation for DQN agent
impl MaskedAgent for DqnAgent {
    fn act_masked(&mut self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> usize {
        let q_values = self.q_network.forward(state);
        
        // Apply mask
        let mut masked_q = q_values.clone();
        for (i, &is_valid) in action_mask.iter().enumerate() {
            if !is_valid {
                masked_q[i] = f32::NEG_INFINITY;
            }
        }
        
        // Epsilon-greedy with valid actions only
        if self.rng.gen::<f32>() < self.epsilon {
            // Random valid action
            let valid_actions: Vec<usize> = action_mask.iter()
                .enumerate()
                .filter(|(_, &valid)| valid)
                .map(|(i, _)| i)
                .collect();
                
            if valid_actions.is_empty() {
                panic!("No valid actions available!");
            }
            
            valid_actions[self.rng.gen_range(0..valid_actions.len())]
        } else {
            // Greedy from masked Q-values
            masked_q.iter()
                .enumerate()
                .filter(|(i, _)| action_mask[*i])
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .expect("No valid actions available!")
        }
    }
    
    fn get_masked_q_values(&mut self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> Array1<f32> {
        let mut q_values = self.q_network.forward(state);
        for (i, &is_valid) in action_mask.iter().enumerate() {
            if !is_valid {
                q_values[i] = f32::NEG_INFINITY;
            }
        }
        q_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{SGD, OptimizerWrapper};
    use ndarray::array;
    
    #[test]
    fn test_masked_action_selection() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[4, 32, 32, 3], 0.0, optimizer, 100, false);
        let state = array![1.0, 2.0, 3.0, 4.0];
        let mask = array![true, false, true];  // Only actions 0 and 2 valid
        
        // Should never select action 1
        for _ in 0..100 {
            let action = agent.act_masked(state.view(), &mask);
            assert!(action != 1);
            assert!(action == 0 || action == 2);
        }
    }
    
    #[test]
    fn test_masked_q_values() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[4, 32, 32, 3], 0.1, optimizer, 100, false);
        let state = array![1.0, 2.0, 3.0, 4.0];
        let mask = array![true, false, true];
        
        let masked_q = agent.get_masked_q_values(state.view(), &mask);
        
        // Check masked action has -inf value
        assert_eq!(masked_q[1], f32::NEG_INFINITY);
        
        // Check valid actions have finite values
        assert!(masked_q[0].is_finite());
        assert!(masked_q[2].is_finite());
    }
    
    #[test]
    fn test_masked_epsilon_greedy() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        // High epsilon for random selection
        let mut agent = DqnAgent::new(&[4, 32, 32, 3], 1.0, optimizer, 100, false);
        let state = array![1.0, 2.0, 3.0, 4.0];
        let mask = array![true, false, true];
        
        let mut action_counts = [0, 0, 0];
        
        // With epsilon=1.0, should select randomly from valid actions
        for _ in 0..1000 {
            let action = agent.act_masked(state.view(), &mask);
            action_counts[action] += 1;
        }
        
        // Action 1 should never be selected
        assert_eq!(action_counts[1], 0);
        
        // Actions 0 and 2 should be selected roughly equally
        let ratio = action_counts[0] as f32 / action_counts[2] as f32;
        assert!(ratio > 0.8 && ratio < 1.2);
    }
    
    #[test]
    #[should_panic(expected = "No valid actions available!")]
    fn test_no_valid_actions() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[4, 32, 32, 3], 0.0, optimizer, 100, false);
        let state = array![1.0, 2.0, 3.0, 4.0];
        let mask = array![false, false, false];  // No valid actions
        
        // Should panic
        agent.act_masked(state.view(), &mask);
    }
}