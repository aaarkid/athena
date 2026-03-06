use crate::belief::BeliefState;
use crate::network::NeuralNetwork;
use crate::agent::DqnAgent;
use crate::error::Result;
use crate::activations::Activation;
use crate::optimizer::{OptimizerWrapper, SGD};
use ndarray::{Array1, ArrayView1};

/// Agent that maintains belief state
pub struct BeliefAgent<B: BeliefState> {
    belief: B,
    belief_encoder: NeuralNetwork,
    _observation_dim: usize,
}

impl<B: BeliefState> BeliefAgent<B> {
    pub fn new(belief: B, observation_dim: usize, encoding_dim: usize) -> Self {
        let belief_size = belief.to_feature_vector().len();
        let activations = vec![Activation::Relu, Activation::Relu];
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let belief_encoder = NeuralNetwork::new(&[belief_size, 128, encoding_dim], &activations, optimizer);
        
        Self {
            belief,
            belief_encoder,
            _observation_dim: observation_dim,
        }
    }
    
    /// Act based on belief state
    pub fn act_with_belief<A>(&mut self, agent: &mut A, observation: &B::Observation) -> Result<usize>
    where
        A: DqnAgentLike,
    {
        // Get belief representation
        let belief_vector = self.belief.to_feature_vector();
        let encoded_belief = self.belief_encoder.forward(belief_vector.view());
        
        // Use encoded belief as state for base agent
        let action = agent.act(encoded_belief.view())?;
        
        // Update belief with taken action
        self.belief.update(action, observation);
        
        Ok(action)
    }
    
    /// Get Q-values based on belief state
    pub fn get_belief_q_values<A>(&mut self, agent: &mut A) -> Array1<f32>
    where
        A: DqnAgentLike,
    {
        let belief_vector = self.belief.to_feature_vector();
        let encoded_belief = self.belief_encoder.forward(belief_vector.view());
        agent.get_q_values(encoded_belief.view())
    }
    
    /// Reset belief state
    pub fn reset(&mut self) {
        self.belief.reset();
    }
    
    /// Get current belief entropy
    pub fn get_entropy(&self) -> f32 {
        self.belief.entropy()
    }
}

/// Trait for agents that can work with belief states
pub trait DqnAgentLike {
    fn act(&mut self, state: ArrayView1<f32>) -> Result<usize>;
    fn get_q_values(&mut self, state: ArrayView1<f32>) -> Array1<f32>;
}

impl DqnAgentLike for DqnAgent {
    fn act(&mut self, state: ArrayView1<f32>) -> Result<usize> {
        self.act(state)
    }
    
    fn get_q_values(&mut self, state: ArrayView1<f32>) -> Array1<f32> {
        self.q_network.forward(state)
    }
}

/// Combined agent with belief tracking
pub struct BeliefDqnAgent<B: BeliefState + Clone> {
    pub base_agent: DqnAgent,
    pub belief_agent: BeliefAgent<B>,
}

impl<B: BeliefState + Clone> BeliefDqnAgent<B> {
    pub fn new(
        base_agent: DqnAgent,
        belief: B,
        observation_dim: usize,
        encoding_dim: usize,
    ) -> Self {
        let belief_agent = BeliefAgent::new(belief, observation_dim, encoding_dim);
        Self {
            base_agent,
            belief_agent,
        }
    }
    
    /// Act based on current belief
    pub fn act(&mut self, observation: &B::Observation) -> Result<usize> {
        self.belief_agent.act_with_belief(&mut self.base_agent, observation)
    }
    
    /// Train on a batch of belief-based experiences
    pub fn train_on_belief_batch(
        &mut self,
        experiences: &[(B::Observation, usize, f32, B::Observation, bool)],
        gamma: f32,
        learning_rate: f32,
    ) -> Result<()> {
        // Convert observations to belief states
        let mut encoded_experiences = Vec::new();
        
        for (_obs, action, reward, next_obs, done) in experiences {
            // Encode current observation
            let belief_vec = self.belief_agent.belief.to_feature_vector();
            let encoded_state = self.belief_agent.belief_encoder.forward(belief_vec.view());
            
            // Temporarily update belief for next state
            let saved_belief = self.belief_agent.belief.clone();
            self.belief_agent.belief.update(*action, next_obs);
            let next_belief_vec = self.belief_agent.belief.to_feature_vector();
            let encoded_next_state = self.belief_agent.belief_encoder.forward(next_belief_vec.view());
            
            // Restore belief
            self.belief_agent.belief = saved_belief;
            
            encoded_experiences.push(crate::replay_buffer::Experience {
                state: encoded_state,
                action: *action,
                reward: *reward,
                next_state: encoded_next_state,
                done: *done,
            });
        }
        
        // Train base agent on encoded experiences
        // Convert to references for train_on_batch
        let exp_refs: Vec<&crate::replay_buffer::Experience> = encoded_experiences.iter().collect();
        self.base_agent.train_on_batch(&exp_refs, gamma, learning_rate)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::HistoryBelief;
    use crate::optimizer::{SGD, OptimizerWrapper};
    use ndarray::array;
    
    #[test]
    fn test_belief_agent() {
        let belief = HistoryBelief::new(5, 30);
        let mut belief_agent = BeliefAgent::new(belief, 4, 16);
        
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut dqn = DqnAgent::new(&[16, 32, 32, 2], 0.1, optimizer, 100, false);
        
        let obs = array![1.0, 2.0, 3.0, 4.0];
        let action = belief_agent.act_with_belief(&mut dqn, &obs).unwrap();
        
        assert!(action < 2);
    }
    
    #[test]
    fn test_belief_dqn_agent() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let base_agent = DqnAgent::new(&[16, 32, 32, 2], 0.1, optimizer, 100, false);
        let belief = HistoryBelief::new(5, 30);
        
        let mut agent = BeliefDqnAgent::new(base_agent, belief, 4, 16);
        
        let obs = array![1.0, 2.0, 3.0, 4.0];
        let action = agent.act(&obs).unwrap();
        
        assert!(action < 2);
    }
}