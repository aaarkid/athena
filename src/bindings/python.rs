use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use crate::network::NeuralNetwork;
use crate::agent::DqnAgent;
use crate::activations::Activation;
use crate::optimizer::{OptimizerWrapper, SGD, Adam, RMSProp};
use crate::replay_buffer::{ReplayBuffer, Experience};
use crate::layers::Layer;

/// Python wrapper for NeuralNetwork
#[pyclass(name = "NeuralNetwork")]
pub struct PyNeuralNetwork {
    inner: NeuralNetwork,
}

#[pymethods]
impl PyNeuralNetwork {
    #[new]
    fn new(
        layer_sizes: Vec<usize>,
        activations: Vec<String>,
        optimizer: String,
    ) -> PyResult<Self> {
        // Parse activations
        let activations = activations
            .into_iter()
            .map(|s| match s.as_str() {
                "relu" => Ok(Activation::Relu),
                "sigmoid" => Ok(Activation::Sigmoid),
                "tanh" => Ok(Activation::Tanh),
                "linear" => Ok(Activation::Linear),
                "leaky_relu" => Ok(Activation::LeakyRelu { alpha: 0.01 }),
                "elu" => Ok(Activation::Elu { alpha: 1.0 }),
                "gelu" => Ok(Activation::Gelu),
                _ => Err(PyValueError::new_err(format!("Unknown activation: {}", s))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        // Parse optimizer
        let optimizer = match optimizer.as_str() {
            "sgd" => OptimizerWrapper::SGD(SGD::new()),
            "adam" => {
                // Create dummy layers for Adam initialization
                let layers = create_dummy_layers(&layer_sizes, &activations);
                OptimizerWrapper::Adam(Adam::default(&layers))
            }
            "rmsprop" => {
                let layers = create_dummy_layers(&layer_sizes, &activations);
                OptimizerWrapper::RMSProp(RMSProp::default(&layers))
            }
            _ => return Err(PyValueError::new_err(format!("Unknown optimizer: {}", optimizer))),
        };
        
        let network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
        Ok(PyNeuralNetwork { inner: network })
    }
    
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<f32>,
    ) -> &'py PyArray1<f32> {
        let input = input.as_array();
        let output = self.inner.forward(input);
        PyArray1::from_array(py, &output)
    }
    
    fn forward_batch<'py>(
        &mut self,
        py: Python<'py>,
        inputs: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        let inputs = inputs.as_array();
        let outputs = self.inner.forward_batch(inputs);
        PyArray2::from_array(py, &outputs)
    }
    
    fn train_minibatch(
        &mut self,
        inputs: PyReadonlyArray2<f32>,
        targets: PyReadonlyArray2<f32>,
        learning_rate: f32,
    ) {
        let inputs = inputs.as_array();
        let targets = targets.as_array();
        self.inner.train_minibatch(inputs, targets, learning_rate);
    }
    
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        NeuralNetwork::load(path)
            .map(|inner| PyNeuralNetwork { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Python wrapper for DqnAgent
#[pyclass(name = "DqnAgent", unsendable)]
pub struct PyDqnAgent {
    inner: DqnAgent,
}

#[pymethods]
impl PyDqnAgent {
    #[new]
    fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        epsilon: f32,
        target_update_freq: usize,
        use_double_dqn: bool,
    ) -> PyResult<Self> {
        // Build layer sizes for the network
        let mut layer_sizes = vec![state_size];
        layer_sizes.extend_from_slice(&hidden_sizes);
        layer_sizes.push(action_size);
        
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        
        let agent = DqnAgent::new(
            &layer_sizes,
            epsilon,
            optimizer,
            target_update_freq,
            use_double_dqn,
        );
        
        Ok(PyDqnAgent { inner: agent })
    }
    
    fn act<'py>(
        &mut self,
        _py: Python<'py>,
        state: PyReadonlyArray1<f32>,
    ) -> PyResult<usize> {
        let state = state.as_array();
        self.inner.act(state)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    fn train(
        &mut self,
        replay_buffer: &PyReplayBuffer,
        batch_size: usize,
        learning_rate: f32,
    ) -> PyResult<f32> {
        let batch = replay_buffer.inner.sample(batch_size);
        self.inner.train_on_batch(&batch, 0.99, learning_rate)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    fn update_target_network(&mut self) {
        self.inner.update_target_network();
    }
    
    fn decay_epsilon(&mut self, decay_rate: f32) {
        self.inner.epsilon *= decay_rate;
    }
    
    #[getter]
    fn epsilon(&self) -> f32 {
        self.inner.epsilon
    }
    
    #[setter]
    fn set_epsilon(&mut self, value: f32) {
        self.inner.epsilon = value;
    }
}

/// Python wrapper for ReplayBuffer
#[pyclass(name = "ReplayBuffer")]
pub struct PyReplayBuffer {
    inner: ReplayBuffer,
}

#[pymethods]
impl PyReplayBuffer {
    #[new]
    fn new(capacity: usize) -> Self {
        PyReplayBuffer {
            inner: ReplayBuffer::new(capacity),
        }
    }
    
    fn add(
        &mut self,
        state: PyReadonlyArray1<f32>,
        action: usize,
        reward: f32,
        next_state: PyReadonlyArray1<f32>,
        done: bool,
    ) {
        let experience = Experience {
            state: state.as_array().to_owned(),
            action,
            reward,
            next_state: next_state.as_array().to_owned(),
            done,
        };
        self.inner.add(experience);
    }
    
    fn sample<'py>(&self, py: Python<'py>, batch_size: usize) -> PyResult<PyObject> {
        let batch = self.inner.sample(batch_size);
        
        // Convert to Python lists
        let states = PyList::empty(py);
        let actions = PyList::empty(py);
        let rewards = PyList::empty(py);
        let next_states = PyList::empty(py);
        let dones = PyList::empty(py);
        
        for exp in batch {
            states.append(PyArray1::from_array(py, &exp.state))?;
            actions.append(exp.action)?;
            rewards.append(exp.reward)?;
            next_states.append(PyArray1::from_array(py, &exp.next_state))?;
            dones.append(exp.done)?;
        }
        
        // Return as tuple
        Ok((states, actions, rewards, next_states, dones).into_py(py))
    }
    
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Helper function to create dummy layers for optimizer initialization
fn create_dummy_layers(layer_sizes: &[usize], activations: &[Activation]) -> Vec<Layer> {
    layer_sizes
        .windows(2)
        .zip(activations.iter())
        .map(|(window, &activation)| {
            Layer::new(window[0], window[1], activation)
        })
        .collect()
}

/// Python module initialization
#[pymodule]
fn athena_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeuralNetwork>()?;
    m.add_class::<PyDqnAgent>()?;
    m.add_class::<PyReplayBuffer>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dummy_layers() {
        let layer_sizes = vec![4, 32, 32, 2];
        let activations = vec![Activation::Relu, Activation::Relu, Activation::Linear];
        let layers = create_dummy_layers(&layer_sizes, &activations);
        
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].input_size(), 4);
        assert_eq!(layers[0].output_size(), 32);
    }
}