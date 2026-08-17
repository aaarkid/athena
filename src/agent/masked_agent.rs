use ndarray::{Array1, ArrayView1};
use rand::Rng;
use crate::agent::dqn::DqnAgent;
use crate::error::{AthenaError, Result};

/// Extension trait for agents with action masking
pub trait MaskedAgent {
    /// Select action with invalid actions masked out.
    ///
    /// Returns an error rather than panicking when the mask does not match the number of
    /// actions or when it leaves nothing to choose from. A game decides its own mask, so
    /// both are conditions a caller can hit at runtime.
    fn act_masked(&mut self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> Result<usize>;

    /// Get Q-values with masking applied. Masked entries are negative infinity.
    fn get_masked_q_values(
        &mut self,
        state: ArrayView1<f32>,
        action_mask: &Array1<bool>,
    ) -> Result<Array1<f32>>;
}

/// Implementation for DQN agent
impl MaskedAgent for DqnAgent {
    fn act_masked(&mut self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> Result<usize> {
        // Test epsilon first: an exploring step never needs the network
        if self.rng.gen::<f32>() < self.epsilon {
            check_mask(action_mask, self.q_network.output_size())?;

            let valid_actions: Vec<usize> = action_mask
                .iter()
                .enumerate()
                .filter(|(_, &valid)| valid)
                .map(|(i, _)| i)
                .collect();

            return Ok(valid_actions[self.rng.gen_range(0..valid_actions.len())]);
        }

        let masked_q = self.get_masked_q_values(state, action_mask)?;

        masked_q
            .iter()
            .enumerate()
            .filter(|(i, _)| action_mask[*i])
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| {
                AthenaError::invalid_parameter("action_mask", "no valid actions available")
            })
    }

    fn get_masked_q_values(
        &mut self,
        state: ArrayView1<f32>,
        action_mask: &Array1<bool>,
    ) -> Result<Array1<f32>> {
        check_mask(action_mask, self.q_network.output_size())?;

        let mut q_values = self.q_network.try_forward(state)?;
        for (i, &is_valid) in action_mask.iter().enumerate() {
            if !is_valid {
                q_values[i] = f32::NEG_INFINITY;
            }
        }

        Ok(q_values)
    }
}

fn check_mask(action_mask: &Array1<bool>, num_actions: usize) -> Result<()> {
    if action_mask.len() != num_actions {
        return Err(AthenaError::dimension_mismatch(
            format!("action mask of length {}", num_actions),
            format!("length {}", action_mask.len()),
        ));
    }

    if !action_mask.iter().any(|&valid| valid) {
        return Err(AthenaError::invalid_parameter(
            "action_mask",
            "no valid actions available",
        ));
    }

    Ok(())
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
            let action = agent.act_masked(state.view(), &mask).unwrap();
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

        let masked_q = agent.get_masked_q_values(state.view(), &mask).unwrap();

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
            let action = agent.act_masked(state.view(), &mask).unwrap();
            action_counts[action] += 1;
        }

        // Action 1 should never be selected
        assert_eq!(action_counts[1], 0);

        // Actions 0 and 2 should be selected roughly equally
        let ratio = action_counts[0] as f32 / action_counts[2] as f32;
        assert!(ratio > 0.8 && ratio < 1.2);
    }

    #[test]
    fn an_empty_mask_is_an_error_not_a_panic() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[4, 32, 32, 3], 0.0, optimizer, 100, false);
        let state = array![1.0, 2.0, 3.0, 4.0];
        let mask = array![false, false, false];

        assert!(agent.act_masked(state.view(), &mask).is_err());
    }

    #[test]
    fn a_mask_of_the_wrong_length_is_an_error() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[4, 32, 32, 3], 0.0, optimizer, 100, false);
        let state = array![1.0, 2.0, 3.0, 4.0];

        assert!(agent.act_masked(state.view(), &array![true, true]).is_err());
        assert!(agent.act_masked(state.view(), &array![true, true, true, true, true]).is_err());
    }

    #[test]
    fn a_state_of_the_wrong_width_is_an_error() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[4, 32, 32, 3], 0.0, optimizer, 100, false);
        let mask = array![true, false, true];

        let too_wide = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(agent.act_masked(too_wide.view(), &mask).is_err());
    }
}
