use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::from_value;

use crate::network::NeuralNetwork;
use crate::agent::DqnAgent;
use crate::activations::Activation;
use crate::optimizer::{OptimizerWrapper, SGD, Adam};
use crate::replay_buffer::{ReplayBuffer, Experience};
use crate::layers::Layer;

/// WebAssembly bindings for NeuralNetwork
#[wasm_bindgen]
pub struct WasmNeuralNetwork {
    inner: NeuralNetwork,
}

#[wasm_bindgen]
impl WasmNeuralNetwork {
    /// Create a new neural network
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmNeuralNetwork, JsValue> {
        let config: NetworkConfig = from_value(config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let activations = config.activations
            .into_iter()
            .map(|s| parse_activation(&s))
            .collect::<Result<Vec<_>, _>>()?;
        
        let optimizer = parse_optimizer(&config.optimizer, &config.layer_sizes, &activations)?;
        
        let network = NeuralNetwork::new(&config.layer_sizes, &activations, optimizer);
        Ok(WasmNeuralNetwork { inner: network })
    }
    
    /// Perform forward pass
    pub fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let input_array = ndarray::Array1::from_vec(input);
        let output = self.inner.forward(input_array.view());
        output.to_vec()
    }
    
    /// Perform batch forward pass
    pub fn forward_batch(&mut self, inputs: Vec<f32>, batch_size: usize) -> Vec<f32> {
        let input_size = inputs.len() / batch_size;
        let shape = (batch_size, input_size);
        let input_array = ndarray::Array2::from_shape_vec(shape, inputs)
            .expect("Invalid input shape");
        
        let output = self.inner.forward_batch(input_array.view());
        output.into_raw_vec()
    }
    
    /// Train on a minibatch
    pub fn train_minibatch(
        &mut self,
        inputs: Vec<f32>,
        targets: Vec<f32>,
        batch_size: usize,
        learning_rate: f32,
    ) -> Result<(), JsValue> {
        let input_size = inputs.len() / batch_size;
        let output_size = targets.len() / batch_size;
        
        let input_array = ndarray::Array2::from_shape_vec((batch_size, input_size), inputs)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let target_array = ndarray::Array2::from_shape_vec((batch_size, output_size), targets)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        self.inner.train_minibatch(input_array.view(), target_array.view(), learning_rate);
        Ok(())
    }
    
    /// Save network to JSON string
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    /// Load network from JSON string
    pub fn from_json(json: &str) -> Result<WasmNeuralNetwork, JsValue> {
        let inner: NeuralNetwork = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmNeuralNetwork { inner })
    }
}

/// WebAssembly bindings for DqnAgent
#[wasm_bindgen]
pub struct WasmDqnAgent {
    inner: DqnAgent,
}

#[wasm_bindgen]
impl WasmDqnAgent {
    /// Create a new DQN agent
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmDqnAgent, JsValue> {
        let config: DqnConfig = from_value(config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let mut layer_sizes = vec![config.state_size];
        layer_sizes.extend_from_slice(&config.hidden_sizes);
        layer_sizes.push(config.action_size);
        
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        
        let agent = DqnAgent::new(
            &layer_sizes,
            config.epsilon,
            optimizer,
            config.target_update_freq,
            config.use_double_dqn,
        );
        
        Ok(WasmDqnAgent { inner: agent })
    }
    
    /// Select an action
    pub fn act(&mut self, state: Vec<f32>) -> Result<usize, JsValue> {
        let state_array = ndarray::Array1::from_vec(state);
        self.inner.act(state_array.view())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    /// Update target network
    pub fn update_target_network(&mut self) {
        self.inner.update_target_network();
    }
    
    /// Decay epsilon
    pub fn decay_epsilon(&mut self, decay_rate: f32) {
        self.inner.epsilon *= decay_rate;
    }
    
    /// Get current epsilon value
    pub fn get_epsilon(&self) -> f32 {
        self.inner.epsilon
    }
    
    /// Set epsilon value
    pub fn set_epsilon(&mut self, value: f32) {
        self.inner.epsilon = value;
    }
}

/// WebAssembly bindings for ReplayBuffer
#[wasm_bindgen]
pub struct WasmReplayBuffer {
    inner: ReplayBuffer,
}

#[wasm_bindgen]
impl WasmReplayBuffer {
    /// Create a new replay buffer
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> WasmReplayBuffer {
        WasmReplayBuffer {
            inner: ReplayBuffer::new(capacity),
        }
    }
    
    /// Add an experience
    pub fn add(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) {
        let experience = Experience {
            state: ndarray::Array1::from_vec(state),
            action,
            reward,
            next_state: ndarray::Array1::from_vec(next_state),
            done,
        };
        self.inner.add(experience);
    }
    
    /// Get buffer length
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// Configuration structures
#[derive(serde::Deserialize)]
struct NetworkConfig {
    layer_sizes: Vec<usize>,
    activations: Vec<String>,
    optimizer: String,
}

#[derive(serde::Deserialize)]
struct DqnConfig {
    state_size: usize,
    action_size: usize,
    hidden_sizes: Vec<usize>,
    epsilon: f32,
    target_update_freq: usize,
    use_double_dqn: bool,
}

// Helper functions
fn parse_activation(name: &str) -> Result<Activation, JsValue> {
    match name {
        "relu" => Ok(Activation::Relu),
        "sigmoid" => Ok(Activation::Sigmoid),
        "tanh" => Ok(Activation::Tanh),
        "linear" => Ok(Activation::Linear),
        "leaky_relu" => Ok(Activation::LeakyRelu { alpha: 0.01 }),
        "elu" => Ok(Activation::Elu { alpha: 1.0 }),
        "gelu" => Ok(Activation::Gelu),
        _ => Err(JsValue::from_str(&format!("Unknown activation: {}", name))),
    }
}

fn parse_optimizer(
    name: &str,
    layer_sizes: &[usize],
    activations: &[Activation],
) -> Result<OptimizerWrapper, JsValue> {
    match name {
        "sgd" => Ok(OptimizerWrapper::SGD(SGD::new())),
        "adam" => {
            let layers = create_dummy_layers(layer_sizes, activations);
            Ok(OptimizerWrapper::Adam(Adam::default(&layers)))
        }
        _ => Err(JsValue::from_str(&format!("Unknown optimizer: {}", name))),
    }
}

fn create_dummy_layers(layer_sizes: &[usize], activations: &[Activation]) -> Vec<Layer> {
    layer_sizes
        .windows(2)
        .zip(activations.iter())
        .map(|(window, &activation)| {
            Layer::new(window[0], window[1], activation)
        })
        .collect()
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Set panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}