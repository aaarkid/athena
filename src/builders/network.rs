use crate::network::NeuralNetwork;
use crate::activations::Activation;
use crate::optimizer::{OptimizerWrapper, SGD, Adam, RMSProp};
use crate::layers::Layer;
use crate::error::{Result, AthenaError};

/// Builder for constructing neural networks with a fluent API
pub struct NetworkBuilder {
    layers: Vec<Layer>,
    optimizer: Option<OptimizerWrapper>,
    /// Deferred until `build`, so the choice can be made before any layer is added.
    /// Adam and RMSProp size their state from the layers, which do not exist yet when
    /// `with_adam` is the first call in the chain.
    pending_optimizer: Option<PendingOptimizer>,
}

/// An optimizer choice recorded before the layers were known.
#[derive(Clone, Copy)]
enum PendingOptimizer {
    Adam { beta1: f32, beta2: f32, epsilon: f32 },
    RMSProp { beta: f32, epsilon: f32 },
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        NetworkBuilder {
            layers: Vec::new(),
            optimizer: None,
            pending_optimizer: None,
        }
    }
    
    /// Add a dense layer to the network
    pub fn add_dense(mut self, input_size: usize, output_size: usize, activation: Activation) -> Self {
        self.layers.push(Layer::new(input_size, output_size, activation));
        self
    }
    
    /// Add a sequence of dense layers
    pub fn add_layers(mut self, layer_sizes: &[usize], activations: &[Activation]) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(AthenaError::InvalidParameter {
                name: "layer_sizes".to_string(),
                reason: "Must have at least 2 layer sizes".to_string(),
            });
        }
        
        if layer_sizes.len() - 1 != activations.len() {
            return Err(AthenaError::DimensionMismatch {
                expected: format!("{} activations", layer_sizes.len() - 1),
                actual: format!("{} activations", activations.len()),
            });
        }
        
        for (window, &activation) in layer_sizes.windows(2).zip(activations.iter()) {
            self.layers.push(Layer::new(window[0], window[1], activation));
        }
        
        Ok(self)
    }
    
    /// Set the optimizer to SGD
    pub fn with_sgd(mut self) -> Self {
        self.optimizer = Some(OptimizerWrapper::SGD(SGD::new()));
        self.pending_optimizer = None;
        self
    }
    
    /// Set the optimizer to Adam.
    ///
    /// The choice is recorded and the optimizer is built in `build`, so this works
    /// whether it comes before or after the layers.
    pub fn with_adam(mut self, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        self.optimizer = None;
        self.pending_optimizer = Some(PendingOptimizer::Adam { beta1, beta2, epsilon });
        self
    }
    
    /// Set the optimizer to RMSProp.
    ///
    /// Same as `with_adam`: the order in the chain does not matter.
    pub fn with_rmsprop(mut self, beta: f32, epsilon: f32) -> Self {
        self.optimizer = None;
        self.pending_optimizer = Some(PendingOptimizer::RMSProp { beta, epsilon });
        self
    }
    
    /// Set a custom optimizer
    pub fn with_optimizer(mut self, optimizer: OptimizerWrapper) -> Self {
        self.optimizer = Some(optimizer);
        self.pending_optimizer = None;
        self
    }
    
    /// Build the neural network
    pub fn build(self) -> Result<NeuralNetwork> {
        if self.layers.is_empty() {
            return Err(AthenaError::InvalidParameter {
                name: "layers".to_string(),
                reason: "Network must have at least one layer".to_string(),
            });
        }
        
        // Now that the layers are known, an Adam or RMSProp choice can be sized
        let optimizer = match (self.optimizer, self.pending_optimizer) {
            (Some(optimizer), _) => optimizer,
            (None, Some(PendingOptimizer::Adam { beta1, beta2, epsilon })) => {
                OptimizerWrapper::Adam(Adam::new(&self.layers, beta1, beta2, epsilon))
            }
            (None, Some(PendingOptimizer::RMSProp { beta, epsilon })) => {
                OptimizerWrapper::RMSProp(RMSProp::new(&self.layers, beta, epsilon))
            }
            (None, None) => {
                return Err(AthenaError::InvalidParameter {
                    name: "optimizer".to_string(),
                    reason: "Optimizer not specified".to_string(),
                })
            }
        };
        
        Ok(NeuralNetwork::new_empty()
            .with_layers(self.layers)
            .with_optimizer(optimizer))
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension methods for NeuralNetwork
impl NeuralNetwork {
    /// Set the optimizer
    pub fn with_optimizer(mut self, optimizer: OptimizerWrapper) -> Self {
        self.optimizer = optimizer;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_builder() {
        let network = NetworkBuilder::new()
            .add_dense(4, 32, Activation::Relu)
            .add_dense(32, 32, Activation::Relu)
            .add_dense(32, 2, Activation::Linear)
            .with_sgd()
            .build()
            .unwrap();
        
        assert_eq!(network.layers.len(), 3);
    }
    
    #[test]
    fn test_network_builder_with_layers() {
        let network = NetworkBuilder::new()
            .add_layers(
                &[4, 32, 32, 2],
                &[Activation::Relu, Activation::Relu, Activation::Linear],
            )
            .unwrap()
            .with_adam(0.9, 0.999, 1e-8)
            .build()
            .unwrap();
        
        assert_eq!(network.layers.len(), 3);
    }
    
    #[test]
    fn test_builder_errors() {
        // No layers
        let result = NetworkBuilder::new().with_sgd().build();
        assert!(result.is_err());
        
        // No optimizer
        let result = NetworkBuilder::new()
            .add_dense(4, 2, Activation::Linear)
            .build();
        assert!(result.is_err());
        
        // Mismatched layer sizes and activations
        let result = NetworkBuilder::new()
            .add_layers(&[4, 32, 2], &[Activation::Relu]);
        assert!(result.is_err()); // This should fail - 2 layers need 2 activations, only 1 provided
    }
}