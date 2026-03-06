use ndarray::{Array1, ArrayView1};

/// Trait for applying action masks to outputs
pub trait MaskedLayer {
    /// Apply mask to output values
    fn apply_mask(&self, output: Array1<f32>, mask: &Array1<bool>) -> Array1<f32>;
}

/// Masked softmax layer for action selection
#[derive(Clone)]
pub struct MaskedSoftmax {
    temperature: f32,
}

impl MaskedSoftmax {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
    
    /// Apply softmax to input
    pub fn forward(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_values = input.mapv(|x| ((x - max) / self.temperature).exp());
        let sum = exp_values.sum();
        exp_values / sum
    }
    
    /// Apply softmax with mask
    pub fn forward_masked(&self, input: ArrayView1<f32>, mask: Option<&Array1<bool>>) -> Array1<f32> {
        match mask {
            Some(m) => self.apply_mask(input.to_owned(), m),
            None => self.forward(input),
        }
    }
}

impl MaskedLayer for MaskedSoftmax {
    fn apply_mask(&self, input: Array1<f32>, mask: &Array1<bool>) -> Array1<f32> {
        // Set masked actions to -inf before softmax
        let mut masked_input = input;
        for (i, &is_valid) in mask.iter().enumerate() {
            if !is_valid {
                masked_input[i] = f32::NEG_INFINITY;
            }
        }
        
        // Apply softmax with temperature
        let max = masked_input.iter()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(_, &x)| x)
            .fold(f32::NEG_INFINITY, f32::max);
            
        if max == f32::NEG_INFINITY {
            // No valid actions, return uniform distribution over valid actions
            let valid_count = mask.iter().filter(|&&x| x).count() as f32;
            let mut result = Array1::zeros(masked_input.len());
            for (i, &is_valid) in mask.iter().enumerate() {
                if is_valid {
                    result[i] = 1.0 / valid_count;
                }
            }
            result
        } else {
            let exp_values = masked_input.mapv(|x| {
                if x == f32::NEG_INFINITY {
                    0.0
                } else {
                    ((x - max) / self.temperature).exp()
                }
            });
            let sum = exp_values.sum();
            exp_values / sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_masked_softmax() {
        let layer = MaskedSoftmax::new(1.0);
        let input = array![1.0, 2.0, 3.0, 4.0];
        let mask = array![true, false, true, false];
        
        let output = layer.forward_masked(input.view(), Some(&mask));
        
        // Check that masked actions have probability 0
        assert_eq!(output[1], 0.0);
        assert_eq!(output[3], 0.0);
        
        // Check that valid actions sum to 1
        assert!((output[0] + output[2] - 1.0).abs() < 1e-6);
        
        // Check that higher value has higher probability
        assert!(output[2] > output[0]);
    }
    
    #[test]
    fn test_masked_softmax_no_valid_actions() {
        let layer = MaskedSoftmax::new(1.0);
        let input = array![1.0, 2.0, 3.0];
        let mask = array![false, false, false];
        
        let output = layer.forward_masked(input.view(), Some(&mask));
        
        // All outputs should be 0 when no valid actions
        assert_eq!(output.sum(), 0.0);
    }
    
    #[test]
    fn test_masked_softmax_temperature() {
        let layer_low_temp = MaskedSoftmax::new(0.1);
        let layer_high_temp = MaskedSoftmax::new(10.0);
        let input = array![1.0, 2.0, 3.0];
        let mask = array![true, true, true];
        
        let output_low = layer_low_temp.forward_masked(input.view(), Some(&mask));
        let output_high = layer_high_temp.forward_masked(input.view(), Some(&mask));
        
        // Low temperature should make distribution more peaked
        let entropy_low = -output_low.iter().map(|p| p * p.ln()).sum::<f32>();
        let entropy_high = -output_high.iter().map(|p| p * p.ln()).sum::<f32>();
        
        assert!(entropy_low < entropy_high);
    }
}