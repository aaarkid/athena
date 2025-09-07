use crate::network::NeuralNetwork;
use crate::activations::Activation;
use crate::optimizer::OptimizerWrapper;
use crate::replay_buffer::Experience;
use ndarray::parallel::prelude::{IntoParallelRefIterator, ParallelIterator};
use rand::{Rng, rngs::ThreadRng};
use ndarray::{Array1, ArrayView1, Axis};
use serde::{Serialize, Deserialize};

pub type State = Array1<f32>;
pub type Action = usize;

/// A Deep Q-Network (DQN) agent.
///
/// The agent includes a neural network that determines the Q-values
/// (expected returns) for each possible action given a certain state.
///
/// # Examples
///
/// Creating a new agent with a custom network architecture:
///
/// ```
/// use athena::agent::DqnAgent;
/// use athena::optimizer::{Optimizer, OptimizerWrapper, SGD};
/// let state_size = 4;
/// let action_size = 2;
/// let epsilon = 0.1;
/// let optimizer = OptimizerWrapper::SGD(SGD::new());
/// let agent = DqnAgent::new(&[state_size, 64, 32, action_size], epsilon, optimizer);
/// ```
///
/// Creating a new agent with the default network architecture:
///
/// ```
/// use athena::agent::DqnAgent;
/// use athena::optimizer::{Optimizer, OptimizerWrapper, SGD};
/// let state_size = 4;
/// let action_size = 2;
/// let epsilon = 0.1;
/// let optimizer = OptimizerWrapper::SGD(SGD::new());
/// let agent = DqnAgent::new_default(state_size, action_size, epsilon, optimizer);
/// ```
#[derive(Serialize, Deserialize)]
pub struct DqnAgent {
    pub network: NeuralNetwork,
    pub epsilon: f32,
    #[serde(skip)]
    pub rng: ThreadRng,
}

impl DqnAgent {
    /// Creates a new DQN agent with a custom network architecture.
    ///
    /// `layer_sizes` should contain the sizes of all layers of the network,
    /// including the input (state size) and output (action size) layers.
    ///
    /// The activation function for all hidden layers is ReLU. The output layer
    /// uses a linear activation function.
    pub fn new(layer_sizes: &[usize], epsilon: f32, optimizer: OptimizerWrapper) -> Self {
        assert!(layer_sizes.len() >= 2);
        
        // Make sure the number of neurons in each layer is at least 1
        for &size in layer_sizes.iter() {
            assert!(size > 0);
        }
        
        // We'll use ReLU activations for all but the last layer, which will be linear
        let mut activations = vec![Activation::Relu; layer_sizes.len() - 2];
        activations.push(Activation::Linear);

        let network = NeuralNetwork::new(layer_sizes, &activations, optimizer);

        // println!("Layer weight shapes: {:?}", network.layers.iter().map(|layer| layer.weights.shape()).collect::<Vec<_>>());

        let rng = rand::thread_rng();

        DqnAgent { network, epsilon, rng }
    }

    /// Creates a new DQN agent with the default network architecture.
    ///
    /// The default architecture is a network with one hidden layer containing
    /// 32 neurons.
    pub fn new_default(state_size: usize, action_size: usize, epsilon: f32, optimizer: OptimizerWrapper) -> Self {
        Self::new(&[state_size, 32, action_size], epsilon, optimizer)
    }

    pub fn act(&mut self, state: ArrayView1<f32>) -> Action {
        if self.rng.gen::<f32>() < self.epsilon {
            self.rng.gen_range(0..self.network.layers.last().unwrap().biases.len())
        } else {
            let q_values = self.network.forward(state);
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }
    }

    pub fn update_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    pub fn train_on_batch(&mut self, experiences: &[&Experience], gamma: f32, learning_rate: f32) {
        let (states, target_q_values): (Vec<_>, Vec<_>) = experiences
            .par_iter() // parallel iterator
            .map(|experience| {
                let state = experience.state.view();
                let next_state = experience.next_state.view();

                // We need to clone the network for each thread.
                // This should be done in a way that the network doesn't need to be re-initialized each time, for example by using Arc.
                let mut thread_local_network = self.network.clone(); 

                let q_values = thread_local_network.forward(state);
                let next_q_values = thread_local_network.forward(next_state);

                let mut target_q_values = q_values.clone();
                let max_next_q_value = *next_q_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let target_value = experience.reward + gamma * max_next_q_value;
                target_q_values[experience.action] = target_value;

                (state, target_q_values)
            })
            .unzip();

        let states_2d = ndarray::stack(Axis(0), &states).unwrap();
        let target_q_values_2d = ndarray::stack(Axis(0), &target_q_values.iter().map(|array| array.view()).collect::<Vec<_>>()).unwrap();

        self.network.train_minibatch(
            states_2d.view(),
            target_q_values_2d.view(),
            learning_rate,
        );
    }
}
