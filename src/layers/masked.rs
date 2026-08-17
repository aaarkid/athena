use ndarray::{Array1, ArrayView1};

use crate::error::{AthenaError, Result};

/// Trait for applying action masks to outputs
pub trait MaskedLayer {
    /// Apply mask to output values.
    ///
    /// Errors when the mask length does not match the output or when it leaves no
    /// action valid, since there is no distribution over an empty set.
    fn apply_mask(&self, output: Array1<f32>, mask: &Array1<bool>) -> Result<Array1<f32>>;
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
    pub fn forward_masked(
        &self,
        input: ArrayView1<f32>,
        mask: Option<&Array1<bool>>,
    ) -> Result<Array1<f32>> {
        match mask {
            Some(m) => self.apply_mask(input.to_owned(), m),
            None => Ok(self.forward(input)),
        }
    }
}

impl MaskedLayer for MaskedSoftmax {
    fn apply_mask(&self, input: Array1<f32>, mask: &Array1<bool>) -> Result<Array1<f32>> {
        if mask.len() != input.len() {
            return Err(AthenaError::dimension_mismatch(
                format!("mask of length {}", input.len()),
                format!("length {}", mask.len()),
            ));
        }

        // A distribution over no actions does not exist. This used to fall through to a
        // vector of zeros under a comment promising a uniform distribution.
        if !mask.iter().any(|&is_valid| is_valid) {
            return Err(AthenaError::invalid_parameter(
                "mask",
                "no valid actions, there is no distribution to return",
            ));
        }

        // Softmax over the valid entries only, shifted by their max for stability
        let max = input
            .iter()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(_, &x)| x)
            .fold(f32::NEG_INFINITY, f32::max);

        let mut result = Array1::zeros(input.len());
        let mut sum = 0.0;
        for (i, &is_valid) in mask.iter().enumerate() {
            if is_valid {
                let value = ((input[i] - max) / self.temperature).exp();
                result[i] = value;
                sum += value;
            }
        }

        if sum > 0.0 {
            result /= sum;
        }

        Ok(result)
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
        
        let output = layer.forward_masked(input.view(), Some(&mask)).unwrap();
        
        // Check that masked actions have probability 0
        assert_eq!(output[1], 0.0);
        assert_eq!(output[3], 0.0);
        
        // Check that valid actions sum to 1
        assert!((output[0] + output[2] - 1.0).abs() < 1e-6);
        
        // Check that higher value has higher probability
        assert!(output[2] > output[0]);
    }
    
    #[test]
    fn an_all_false_mask_is_an_error() {
        let layer = MaskedSoftmax::new(1.0);
        let input = array![1.0, 2.0, 3.0];
        let mask = array![false, false, false];
        
        // There is no distribution over an empty set of actions
        assert!(layer.forward_masked(input.view(), Some(&mask)).is_err());
    }
    
    #[test]
    fn test_masked_softmax_temperature() {
        let layer_low_temp = MaskedSoftmax::new(0.1);
        let layer_high_temp = MaskedSoftmax::new(10.0);
        let input = array![1.0, 2.0, 3.0];
        let mask = array![true, true, true];
        
        let output_low = layer_low_temp.forward_masked(input.view(), Some(&mask)).unwrap();
        let output_high = layer_high_temp.forward_masked(input.view(), Some(&mask)).unwrap();
        
        // Low temperature should make distribution more peaked
        let entropy_low = -output_low.iter().map(|p| p * p.ln()).sum::<f32>();
        let entropy_high = -output_high.iter().map(|p| p * p.ln()).sum::<f32>();
        
        assert!(entropy_low < entropy_high);
    }
}