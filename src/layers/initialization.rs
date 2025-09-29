use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform};

/// Weight initialization strategies
#[derive(Debug, Clone)]
pub enum WeightInit {
    /// Xavier/Glorot uniform initialization
    XavierUniform,
    
    /// Xavier/Glorot normal initialization  
    XavierNormal,
    
    /// He/Kaiming uniform initialization (for ReLU)
    HeUniform,
    
    /// He/Kaiming normal initialization (for ReLU)
    HeNormal,
    
    /// Uniform distribution with custom range
    Uniform { min: f32, max: f32 },
    
    /// Normal distribution with custom mean and std
    Normal { mean: f32, std: f32 },
    
    /// All zeros
    Zeros,
    
    /// All ones
    Ones,
}

impl WeightInit {
    /// Initialize weights for a layer
    pub fn initialize_weights(&self, shape: (usize, usize)) -> Array2<f32> {
        let (fan_in, fan_out) = shape;
        
        match self {
            WeightInit::XavierUniform => {
                let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                Array2::random(shape, Uniform::new(-limit, limit))
            }
            
            WeightInit::XavierNormal => {
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Array2::random(shape, Normal::new(0.0, std).unwrap())
            }
            
            WeightInit::HeUniform => {
                let limit = (6.0 / fan_in as f32).sqrt();
                Array2::random(shape, Uniform::new(-limit, limit))
            }
            
            WeightInit::HeNormal => {
                let std = (2.0 / fan_in as f32).sqrt();
                Array2::random(shape, Normal::new(0.0, std).unwrap())
            }
            
            WeightInit::Uniform { min, max } => {
                Array2::random(shape, Uniform::new(*min, *max))
            }
            
            WeightInit::Normal { mean, std } => {
                Array2::random(shape, Normal::new(*mean, *std).unwrap())
            }
            
            WeightInit::Zeros => {
                Array2::zeros(shape)
            }
            
            WeightInit::Ones => {
                Array2::ones(shape)
            }
        }
    }
    
    /// Initialize biases for a layer
    pub fn initialize_biases(&self, size: usize) -> Array1<f32> {
        match self {
            WeightInit::Zeros | WeightInit::XavierUniform | WeightInit::XavierNormal 
            | WeightInit::HeUniform | WeightInit::HeNormal => {
                Array1::zeros(size)
            }
            
            WeightInit::Ones => {
                Array1::ones(size)
            }
            
            WeightInit::Uniform { min, max } => {
                Array1::random(size, Uniform::new(*min, *max))
            }
            
            WeightInit::Normal { mean, std } => {
                Array1::random(size, Normal::new(*mean, *std).unwrap())
            }
        }
    }
    
    /// Get the recommended initialization for an activation function
    pub fn for_activation(activation: &crate::activations::Activation) -> Self {
        use crate::activations::Activation;
        
        match activation {
            Activation::Relu | Activation::LeakyRelu { .. } | Activation::Elu { .. } => {
                WeightInit::HeNormal
            }
            Activation::Sigmoid | Activation::Tanh => {
                WeightInit::XavierNormal
            }
            Activation::Linear | Activation::Gelu => {
                WeightInit::XavierNormal
            }
        }
    }
}