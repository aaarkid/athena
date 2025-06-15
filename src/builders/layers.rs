use crate::layers::{DenseLayer, BatchNormLayer, DropoutLayer};
use crate::layers::initialization::WeightInit;
use crate::activations::Activation;
use crate::error::{Result, AthenaError};

/// Builder for DenseLayer
pub struct DenseLayerBuilder {
    input_size: Option<usize>,
    output_size: Option<usize>,
    activation: Activation,
    weight_init: WeightInit,
}

impl DenseLayerBuilder {
    /// Create a new dense layer builder
    pub fn new() -> Self {
        DenseLayerBuilder {
            input_size: None,
            output_size: None,
            activation: Activation::Linear,
            weight_init: WeightInit::XavierUniform,
        }
    }
    
    /// Set input size
    pub fn input_size(mut self, size: usize) -> Self {
        self.input_size = Some(size);
        self
    }
    
    /// Set output size
    pub fn output_size(mut self, size: usize) -> Self {
        self.output_size = Some(size);
        self
    }
    
    /// Set activation function
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
    
    /// Set weight initialization
    pub fn weight_init(mut self, init: WeightInit) -> Self {
        self.weight_init = init;
        self
    }
    
    /// Build the dense layer
    pub fn build(self) -> Result<DenseLayer> {
        let input_size = self.input_size.ok_or_else(|| AthenaError::InvalidParameter {
            name: "input_size".to_string(),
            reason: "Input size not specified".to_string(),
        })?;
        
        let output_size = self.output_size.ok_or_else(|| AthenaError::InvalidParameter {
            name: "output_size".to_string(),
            reason: "Output size not specified".to_string(),
        })?;
        
        if input_size == 0 || output_size == 0 {
            return Err(AthenaError::InvalidParameter {
                name: "size".to_string(),
                reason: "Layer sizes must be greater than 0".to_string(),
            });
        }
        
        Ok(DenseLayer::new_with_init(
            input_size,
            output_size,
            self.activation,
            self.weight_init,
        ))
    }
}

impl Default for DenseLayerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for BatchNormLayer
pub struct BatchNormLayerBuilder {
    features: Option<usize>,
    momentum: f32,
    epsilon: f32,
}

impl BatchNormLayerBuilder {
    /// Create a new batch norm layer builder
    pub fn new() -> Self {
        BatchNormLayerBuilder {
            features: None,
            momentum: 0.9,
            epsilon: 1e-5,
        }
    }
    
    /// Set number of features
    pub fn features(mut self, features: usize) -> Self {
        self.features = Some(features);
        self
    }
    
    /// Set momentum for running statistics
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
    
    /// Set epsilon for numerical stability
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
    
    /// Build the batch norm layer
    pub fn build(self) -> Result<BatchNormLayer> {
        let features = self.features.ok_or_else(|| AthenaError::InvalidParameter {
            name: "features".to_string(),
            reason: "Number of features not specified".to_string(),
        })?;
        
        if features == 0 {
            return Err(AthenaError::InvalidParameter {
                name: "features".to_string(),
                reason: "Features must be greater than 0".to_string(),
            });
        }
        
        if self.momentum <= 0.0 || self.momentum >= 1.0 {
            return Err(AthenaError::InvalidParameter {
                name: "momentum".to_string(),
                reason: "Momentum must be between 0 and 1".to_string(),
            });
        }
        
        if self.epsilon <= 0.0 {
            return Err(AthenaError::InvalidParameter {
                name: "epsilon".to_string(),
                reason: "Epsilon must be positive".to_string(),
            });
        }
        
        Ok(BatchNormLayer::new(features, self.momentum, self.epsilon))
    }
}

impl Default for BatchNormLayerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for DropoutLayer
pub struct DropoutLayerBuilder {
    size: Option<usize>,
    dropout_rate: f32,
}

impl DropoutLayerBuilder {
    /// Create a new dropout layer builder
    pub fn new() -> Self {
        DropoutLayerBuilder {
            size: None,
            dropout_rate: 0.5,
        }
    }
    
    /// Set layer size
    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }
    
    /// Set dropout rate
    pub fn dropout_rate(mut self, rate: f32) -> Self {
        self.dropout_rate = rate;
        self
    }
    
    /// Build the dropout layer
    pub fn build(self) -> Result<DropoutLayer> {
        let size = self.size.ok_or_else(|| AthenaError::InvalidParameter {
            name: "size".to_string(),
            reason: "Layer size not specified".to_string(),
        })?;
        
        if size == 0 {
            return Err(AthenaError::InvalidParameter {
                name: "size".to_string(),
                reason: "Size must be greater than 0".to_string(),
            });
        }
        
        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err(AthenaError::InvalidParameter {
                name: "dropout_rate".to_string(),
                reason: "Dropout rate must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        
        Ok(DropoutLayer::new(size, self.dropout_rate))
    }
}

impl Default for DropoutLayerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dense_layer_builder() {
        let layer = DenseLayerBuilder::new()
            .input_size(10)
            .output_size(5)
            .activation(Activation::Relu)
            .weight_init(WeightInit::He)
            .build()
            .unwrap();
        
        assert_eq!(layer.weights.shape(), &[10, 5]);
        assert_eq!(layer.biases.len(), 5);
    }
    
    #[test]
    fn test_batch_norm_builder() {
        let layer = BatchNormLayerBuilder::new()
            .features(32)
            .momentum(0.95)
            .epsilon(1e-6)
            .build()
            .unwrap();
        
        assert_eq!(layer.gamma.len(), 32);
        assert_eq!(layer.beta.len(), 32);
    }
    
    #[test]
    fn test_dropout_builder() {
        let layer = DropoutLayerBuilder::new()
            .size(64)
            .dropout_rate(0.2)
            .build()
            .unwrap();
        
        assert_eq!(layer.size, 64);
        assert_eq!(layer.dropout_rate, 0.2);
    }
    
    #[test]
    fn test_builder_errors() {
        // Missing required fields
        assert!(DenseLayerBuilder::new().build().is_err());
        assert!(BatchNormLayerBuilder::new().build().is_err());
        assert!(DropoutLayerBuilder::new().build().is_err());
        
        // Invalid values
        assert!(DenseLayerBuilder::new()
            .input_size(0)
            .output_size(10)
            .build()
            .is_err());
        
        assert!(BatchNormLayerBuilder::new()
            .features(10)
            .momentum(1.5)
            .build()
            .is_err());
        
        assert!(DropoutLayerBuilder::new()
            .size(10)
            .dropout_rate(1.0)
            .build()
            .is_err());
    }
}