use ndarray::{Array1, Array2, ArrayView2};
use serde::{Serialize, Deserialize};
use std::fmt::Debug;

/// Trait for state representations in reinforcement learning
pub trait State: Clone + Debug + Send + Sync {
    /// Convert state to a dense array representation
    fn to_array(&self) -> Array1<f32>;
    
    /// Get the dimensionality of the state
    fn dim(&self) -> usize;
}

/// Object-safe version of State trait
pub trait StateRef {
    /// Convert state to a dense array representation
    fn to_array(&self) -> Array1<f32>;
    
    /// Get the dimensionality of the state
    fn dim(&self) -> usize;
}

/// Trait for action representations
pub trait Action: Clone + Debug + Send + Sync {
    /// Convert action to integer (for discrete actions)
    fn to_discrete(&self) -> Option<usize>;
    
    /// Convert action to continuous values
    fn to_continuous(&self) -> Option<Array1<f32>>;
    
    /// Check if action is discrete
    fn is_discrete(&self) -> bool;
}

/// Object-safe version of Action trait
pub trait ActionRef {
    /// Convert action to integer (for discrete actions)
    fn to_discrete(&self) -> Option<usize>;
    
    /// Convert action to continuous values
    fn to_continuous(&self) -> Option<Array1<f32>>;
    
    /// Check if action is discrete
    fn is_discrete(&self) -> bool;
}

/// Standard dense state representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseState {
    pub data: Array1<f32>,
}

impl DenseState {
    pub fn new(data: Array1<f32>) -> Self {
        DenseState { data }
    }
    
    pub fn from_vec(data: Vec<f32>) -> Self {
        DenseState {
            data: Array1::from_vec(data),
        }
    }
}

impl State for DenseState {
    fn to_array(&self) -> Array1<f32> {
        self.data.clone()
    }
    
    fn dim(&self) -> usize {
        self.data.len()
    }
}

impl StateRef for DenseState {
    fn to_array(&self) -> Array1<f32> {
        self.data.clone()
    }
    
    fn dim(&self) -> usize {
        self.data.len()
    }
}

/// Discrete action representation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscreteAction {
    pub index: usize,
}

impl DiscreteAction {
    pub fn new(index: usize) -> Self {
        DiscreteAction { index }
    }
}

impl Action for DiscreteAction {
    fn to_discrete(&self) -> Option<usize> {
        Some(self.index)
    }
    
    fn to_continuous(&self) -> Option<Array1<f32>> {
        None
    }
    
    fn is_discrete(&self) -> bool {
        true
    }
}

impl ActionRef for DiscreteAction {
    fn to_discrete(&self) -> Option<usize> {
        Some(self.index)
    }
    
    fn to_continuous(&self) -> Option<Array1<f32>> {
        None
    }
    
    fn is_discrete(&self) -> bool {
        true
    }
}

/// Continuous action representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinuousAction {
    pub values: Array1<f32>,
}

impl ContinuousAction {
    pub fn new(values: Array1<f32>) -> Self {
        ContinuousAction { values }
    }
    
    pub fn from_vec(values: Vec<f32>) -> Self {
        ContinuousAction {
            values: Array1::from_vec(values),
        }
    }
}

impl Action for ContinuousAction {
    fn to_discrete(&self) -> Option<usize> {
        None
    }
    
    fn to_continuous(&self) -> Option<Array1<f32>> {
        Some(self.values.clone())
    }
    
    fn is_discrete(&self) -> bool {
        false
    }
}

impl ActionRef for ContinuousAction {
    fn to_discrete(&self) -> Option<usize> {
        None
    }
    
    fn to_continuous(&self) -> Option<Array1<f32>> {
        Some(self.values.clone())
    }
    
    fn is_discrete(&self) -> bool {
        false
    }
}

/// Generic experience type
#[derive(Clone, Debug)]
pub struct GenericExperience<S: State, A: Action> {
    pub state: S,
    pub action: A,
    pub reward: f32,
    pub next_state: S,
    pub done: bool,
}

/// Transition type for model-based RL
#[derive(Clone, Debug)]
pub struct Transition<S: State, A: Action> {
    pub state: S,
    pub action: A,
    pub next_state: S,
    pub reward: f32,
    pub done: bool,
    pub info: Option<TransitionInfo>,
}

/// Additional information about transitions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionInfo {
    pub probability: Option<f32>,
    pub value_estimate: Option<f32>,
    pub advantage: Option<f32>,
    pub td_error: Option<f32>,
}

/// Batch of states for efficient processing
pub struct StateBatch {
    pub data: Array2<f32>,
}

impl StateBatch {
    pub fn new(states: Vec<impl State>) -> Self {
        if states.is_empty() {
            return StateBatch {
                data: Array2::zeros((0, 0)),
            };
        }
        
        let dim = states[0].dim();
        let mut data = Array2::zeros((states.len(), dim));
        
        for (i, state) in states.iter().enumerate() {
            data.row_mut(i).assign(&state.to_array());
        }
        
        StateBatch { data }
    }
    
    pub fn view(&self) -> ArrayView2<f32> {
        self.data.view()
    }
}

/// One discrete index per dimension, for `ActionSpace::MultiDiscrete`
#[derive(Clone, Debug)]
pub struct MultiDiscreteAction {
    pub indices: Vec<usize>,
}

impl MultiDiscreteAction {
    pub fn new(indices: Vec<usize>) -> Self {
        MultiDiscreteAction { indices }
    }
}

impl Action for MultiDiscreteAction {
    fn to_discrete(&self) -> Option<usize> {
        // Only meaningful when there is a single dimension
        if self.indices.len() == 1 { Some(self.indices[0]) } else { None }
    }

    fn to_continuous(&self) -> Option<Array1<f32>> {
        Some(Array1::from_iter(self.indices.iter().map(|&i| i as f32)))
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

impl ActionRef for MultiDiscreteAction {
    fn to_discrete(&self) -> Option<usize> {
        Action::to_discrete(self)
    }

    fn to_continuous(&self) -> Option<Array1<f32>> {
        Action::to_continuous(self)
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

/// Concrete action type that can be boxed
#[derive(Clone, Debug)]
pub enum AnyAction {
    Discrete(DiscreteAction),
    Continuous(ContinuousAction),
    MultiDiscrete(MultiDiscreteAction),
}

impl Action for AnyAction {
    fn to_discrete(&self) -> Option<usize> {
        match self {
            AnyAction::Discrete(a) => Action::to_discrete(a),
            AnyAction::Continuous(_) => None,
            AnyAction::MultiDiscrete(a) => Action::to_discrete(a),
        }
    }

    fn to_continuous(&self) -> Option<Array1<f32>> {
        match self {
            AnyAction::Discrete(_) => None,
            AnyAction::Continuous(a) => Action::to_continuous(a),
            AnyAction::MultiDiscrete(a) => Action::to_continuous(a),
        }
    }

    fn is_discrete(&self) -> bool {
        match self {
            AnyAction::Discrete(_) => true,
            AnyAction::Continuous(_) => false,
            AnyAction::MultiDiscrete(_) => true,
        }
    }
}

/// Action space definition
#[derive(Clone, Debug)]
pub enum ActionSpace {
    Discrete { n: usize },
    Continuous { low: Array1<f32>, high: Array1<f32> },
    MultiDiscrete { nvec: Vec<usize> },
}

impl ActionSpace {
    /// Sample a random action from the space, using the thread generator.
    pub fn sample(&self) -> AnyAction {
        self.sample_with(&mut rand::thread_rng())
    }

    /// Sample a random action using a caller-supplied generator.
    ///
    /// Pass a seeded generator when the sequence has to repeat.
    pub fn sample_with<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> AnyAction {
        match self {
            ActionSpace::Discrete { n } => {
                AnyAction::Discrete(DiscreteAction::new(rng.gen_range(0..*n)))
            }
            ActionSpace::Continuous { low, high } => {
                let values = Array1::from_vec(
                    low.iter()
                        .zip(high.iter())
                        .map(|(&l, &h)| rng.gen_range(l..h))
                        .collect()
                );
                AnyAction::Continuous(ContinuousAction::new(values))
            }
            ActionSpace::MultiDiscrete { nvec } => {
                let indices = nvec.iter().map(|&n| rng.gen_range(0..n)).collect();
                AnyAction::MultiDiscrete(MultiDiscreteAction::new(indices))
            }
        }
    }
    
    /// Check if an action is valid for this space
    pub fn contains(&self, action: &dyn ActionRef) -> bool {
        match self {
            ActionSpace::Discrete { n } => {
                if let Some(idx) = action.to_discrete() {
                    idx < *n
                } else {
                    false
                }
            }
            ActionSpace::Continuous { low, high } => {
                if let Some(values) = action.to_continuous() {
                    values.iter()
                        .zip(low.iter())
                        .zip(high.iter())
                        .all(|((&v, &l), &h)| v >= l && v <= h)
                } else {
                    false
                }
            }
            ActionSpace::MultiDiscrete { nvec } => {
                // Multi-discrete actions come through as one float per dimension
                match action.to_continuous() {
                    Some(values) if values.len() == nvec.len() => {
                        values.iter().zip(nvec.iter()).all(|(&v, &n)| {
                            v >= 0.0 && v.fract() == 0.0 && (v as usize) < n
                        })
                    }
                    _ => false,
                }
            }
        }
    }
}

/// State space definition  
#[derive(Clone, Debug)]
pub enum StateSpace {
    Box { low: Array1<f32>, high: Array1<f32> },
    Discrete { n: usize },
}

impl StateSpace {
    /// Get the dimensionality of the state space
    pub fn dim(&self) -> usize {
        match self {
            StateSpace::Box { low, .. } => low.len(),
            StateSpace::Discrete { n: _ } => 1, // One-hot encoding would be n
        }
    }
    
    /// Check if a state is valid for this space
    pub fn contains(&self, state: &dyn StateRef) -> bool {
        match self {
            StateSpace::Box { low, high } => {
                let arr = state.to_array();
                arr.iter()
                    .zip(low.iter())
                    .zip(high.iter())
                    .all(|((&v, &l), &h)| v >= l && v <= h)
            }
            StateSpace::Discrete { n } => {
                state.dim() == 1 && state.to_array()[0] < *n as f32
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_dense_state() {
        let state = DenseState::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(State::dim(&state), 3);
        assert_eq!(State::to_array(&state), array![1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_discrete_action() {
        let action = DiscreteAction::new(5);
        assert_eq!(Action::to_discrete(&action), Some(5));
        assert_eq!(Action::to_continuous(&action), None);
        assert!(Action::is_discrete(&action));
    }
    
    #[test]
    fn test_continuous_action() {
        let action = ContinuousAction::from_vec(vec![0.1, -0.5, 0.8]);
        assert_eq!(Action::to_discrete(&action), None);
        assert_eq!(Action::to_continuous(&action), Some(array![0.1, -0.5, 0.8]));
        assert!(!Action::is_discrete(&action));
    }
    
    #[test]
    fn test_state_batch() {
        let states = vec![
            DenseState::from_vec(vec![1.0, 2.0]),
            DenseState::from_vec(vec![3.0, 4.0]),
            DenseState::from_vec(vec![5.0, 6.0]),
        ];
        
        let batch = StateBatch::new(states);
        assert_eq!(batch.data.shape(), &[3, 2]);
        assert_eq!(batch.data.row(0), array![1.0, 2.0]);
    }
    
    #[test]
    fn test_action_space() {
        // Discrete space
        let discrete_space = ActionSpace::Discrete { n: 4 };
        let action = DiscreteAction::new(2);
        assert!(discrete_space.contains(&action));
        
        let invalid_action = DiscreteAction::new(5);
        assert!(!discrete_space.contains(&invalid_action));
        
        // Continuous space
        let continuous_space = ActionSpace::Continuous {
            low: array![-1.0, -1.0],
            high: array![1.0, 1.0],
        };
        let valid_action = ContinuousAction::from_vec(vec![0.5, -0.5]);
        assert!(continuous_space.contains(&valid_action));
        
        let invalid_action = ContinuousAction::from_vec(vec![2.0, 0.0]);
        assert!(!continuous_space.contains(&invalid_action));
    }
}
#[cfg(test)]
mod action_space_tests {
    use super::*;

    #[test]
    fn multi_discrete_samples_every_dimension() {
        let space = ActionSpace::MultiDiscrete { nvec: vec![3, 5, 2] };

        for _ in 0..50 {
            let action = space.sample();
            match &action {
                AnyAction::MultiDiscrete(a) => {
                    assert_eq!(a.indices.len(), 3);
                    assert!(a.indices[0] < 3 && a.indices[1] < 5 && a.indices[2] < 2);
                }
                other => panic!("expected MultiDiscrete, got {other:?}"),
            }
            assert!(space.contains(&MultiDiscreteAction::new(match action {
                AnyAction::MultiDiscrete(a) => a.indices,
                _ => unreachable!(),
            })));
        }
    }

    #[test]
    fn multi_discrete_rejects_out_of_range() {
        let space = ActionSpace::MultiDiscrete { nvec: vec![3, 5] };

        assert!(!space.contains(&MultiDiscreteAction::new(vec![3, 0])));
        assert!(!space.contains(&MultiDiscreteAction::new(vec![0])));
        assert!(space.contains(&MultiDiscreteAction::new(vec![2, 4])));
    }
}
