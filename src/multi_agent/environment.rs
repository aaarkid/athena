use ndarray::Array1;
use std::collections::HashMap;

/// Multi-agent environment trait
pub trait MultiAgentEnvironment: Send + Sync {
    type State;
    type Action;
    type Observation;
    
    /// Get number of agents
    fn num_agents(&self) -> usize;
    
    /// Get current active agent(s)
    fn active_agents(&self) -> Vec<usize>;
    
    /// Get observation for specific agent
    fn get_observation(&self, agent_id: usize) -> Self::Observation;
    
    /// Get legal actions for agent
    fn legal_actions(&self, agent_id: usize) -> Array1<bool>;
    
    /// Step environment with actions from active agents
    fn step(&mut self, actions: &[(usize, Self::Action)]) -> MultiAgentTransition<Self::Observation>;
    
    /// Check if episode is done
    fn is_terminal(&self) -> bool;
    
    /// Reset environment
    fn reset(&mut self) -> Vec<Self::Observation>;
}

/// Transition information for multi-agent step
#[derive(Clone, Debug)]
pub struct MultiAgentTransition<O> {
    pub observations: HashMap<usize, O>,
    pub rewards: HashMap<usize, f32>,
    pub done: bool,
}

impl<O> MultiAgentTransition<O> {
    pub fn new(done: bool) -> Self {
        Self {
            observations: HashMap::new(),
            rewards: HashMap::new(),
            done,
        }
    }
}

/// Wrapper to use single-agent environments in multi-agent setting
pub struct TurnBasedWrapper<E> {
    env: E,
    num_agents: usize,
    current_player: usize,
    episode_done: bool,
}

impl<E> TurnBasedWrapper<E> {
    pub fn new(env: E, num_agents: usize) -> Self {
        Self {
            env,
            num_agents,
            current_player: 0,
            episode_done: false,
        }
    }
    
    fn next_player(&mut self) {
        self.current_player = (self.current_player + 1) % self.num_agents;
    }
}

// Simple environment trait for the wrapper
pub trait SimpleEnvironment {
    type State;
    type Action;
    type Observation;
    
    fn reset(&mut self) -> Self::Observation;
    fn step(&mut self, action: Self::Action) -> (Self::Observation, f32, bool);
    fn get_state(&self) -> Self::Observation;
    fn legal_actions(&self) -> Array1<bool>;
}

impl<E> MultiAgentEnvironment for TurnBasedWrapper<E>
where
    E: SimpleEnvironment + Send + Sync,
    E::Observation: Clone,
    E::Action: Clone,
{
    type State = E::State;
    type Action = E::Action;
    type Observation = E::Observation;
    
    fn num_agents(&self) -> usize {
        self.num_agents
    }
    
    fn active_agents(&self) -> Vec<usize> {
        if self.episode_done {
            vec![]
        } else {
            vec![self.current_player]
        }
    }
    
    fn get_observation(&self, _agent_id: usize) -> Self::Observation {
        // In turn-based wrapper, return the current state as observation
        self.env.get_state()
    }
    
    fn legal_actions(&self, _agent_id: usize) -> Array1<bool> {
        self.env.legal_actions()
    }
    
    fn step(&mut self, actions: &[(usize, Self::Action)]) -> MultiAgentTransition<Self::Observation> {
        let mut transition = MultiAgentTransition::new(false);
        
        // Should only have one action from current player
        if let Some((agent_id, action)) = actions.first() {
            if *agent_id == self.current_player {
                let (obs, reward, done) = self.env.step(action.clone());
                
                transition.observations.insert(*agent_id, obs.clone());
                transition.rewards.insert(*agent_id, reward);
                transition.done = done;
                self.episode_done = done;
                
                if !done {
                    self.next_player();
                }
            }
        }
        
        transition
    }
    
    fn is_terminal(&self) -> bool {
        self.episode_done
    }
    
    fn reset(&mut self) -> Vec<Self::Observation> {
        self.current_player = 0;
        self.episode_done = false;
        let obs = self.env.reset();
        vec![obs; self.num_agents]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    struct DummyEnv {
        state: i32,
        max_steps: i32,
    }
    
    impl SimpleEnvironment for DummyEnv {
        type State = i32;
        type Action = i32;
        type Observation = i32;
        
        fn reset(&mut self) -> Self::Observation {
            self.state = 0;
            self.state
        }
        
        fn step(&mut self, action: Self::Action) -> (Self::Observation, f32, bool) {
            self.state += action;
            let done = self.state >= self.max_steps;
            (self.state, action as f32, done)
        }
        
        fn get_state(&self) -> Self::State {
            self.state
        }
        
        fn legal_actions(&self) -> Array1<bool> {
            array![true, true, true]
        }
    }
    
    #[test]
    fn test_turn_based_wrapper() {
        let env = DummyEnv { state: 0, max_steps: 10 };
        let mut wrapper = TurnBasedWrapper::new(env, 2);
        
        assert_eq!(wrapper.num_agents(), 2);
        assert_eq!(wrapper.active_agents(), vec![0]);
        
        // Player 0 takes action
        let transition = wrapper.step(&[(0, 3)]);
        assert_eq!(transition.rewards[&0], 3.0);
        assert!(!transition.done);
        
        // Now player 1 should be active
        assert_eq!(wrapper.active_agents(), vec![1]);
        
        // Player 1 takes action
        let transition = wrapper.step(&[(1, 7)]);
        assert_eq!(transition.rewards[&1], 7.0);
        assert!(transition.done); // 3 + 7 = 10, reached max_steps
    }
}