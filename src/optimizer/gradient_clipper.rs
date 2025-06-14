use ndarray::{Array1, Array2};

/// Gradient clipping methods
#[derive(Clone, Debug)]
pub enum GradientClipper {
    /// Clip gradients by value
    ClipByValue { min: f32, max: f32 },
    
    /// Clip gradients by norm
    ClipByNorm { max_norm: f32 },
    
    /// Clip gradients by global norm (across all parameters)
    ClipByGlobalNorm { max_norm: f32 },
    
    /// No clipping
    None,
}

impl GradientClipper {
    /// Clip weight gradients
    pub fn clip_weights(&self, gradients: &mut Array2<f32>) {
        match self {
            GradientClipper::ClipByValue { min, max } => {
                gradients.mapv_inplace(|g| g.max(*min).min(*max));
            }
            
            GradientClipper::ClipByNorm { max_norm } => {
                let norm = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();
                if norm > *max_norm {
                    let scale = max_norm / norm;
                    gradients.mapv_inplace(|g| g * scale);
                }
            }
            
            GradientClipper::ClipByGlobalNorm { .. } => {
                // Global norm clipping requires access to all gradients at once
                // This is handled separately in the optimizer
            }
            
            GradientClipper::None => {}
        }
    }
    
    /// Clip bias gradients
    pub fn clip_biases(&self, gradients: &mut Array1<f32>) {
        match self {
            GradientClipper::ClipByValue { min, max } => {
                gradients.mapv_inplace(|g| g.max(*min).min(*max));
            }
            
            GradientClipper::ClipByNorm { max_norm } => {
                let norm = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();
                if norm > *max_norm {
                    let scale = max_norm / norm;
                    gradients.mapv_inplace(|g| g * scale);
                }
            }
            
            GradientClipper::ClipByGlobalNorm { .. } => {
                // Global norm clipping requires access to all gradients at once
                // This is handled separately in the optimizer
            }
            
            GradientClipper::None => {}
        }
    }
    
    /// Compute global norm of all gradients
    pub fn compute_global_norm(weight_grads: &[&Array2<f32>], bias_grads: &[&Array1<f32>]) -> f32 {
        let weight_norm_sq: f32 = weight_grads.iter()
            .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
            .sum();
            
        let bias_norm_sq: f32 = bias_grads.iter()
            .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
            .sum();
            
        (weight_norm_sq + bias_norm_sq).sqrt()
    }
    
    /// Apply global norm clipping
    pub fn apply_global_norm_clipping(
        weight_grads: &mut [Array2<f32>], 
        bias_grads: &mut [Array1<f32>], 
        max_norm: f32
    ) {
        // First compute the global norm
        let weight_refs: Vec<&Array2<f32>> = weight_grads.iter().map(|g| g as &Array2<f32>).collect();
        let bias_refs: Vec<&Array1<f32>> = bias_grads.iter().map(|g| g as &Array1<f32>).collect();
        let global_norm = Self::compute_global_norm(&weight_refs, &bias_refs);
        
        // Apply scaling if needed
        if global_norm > max_norm {
            let scale = max_norm / global_norm;
            
            for grad in weight_grads.iter_mut() {
                grad.mapv_inplace(|g| g * scale);
            }
            
            for grad in bias_grads.iter_mut() {
                grad.mapv_inplace(|g| g * scale);
            }
        }
    }
}