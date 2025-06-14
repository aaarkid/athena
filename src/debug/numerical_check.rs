use ndarray::{Array1, Array2, ArrayView2};

/// Types of numerical issues
#[derive(Debug, Clone, PartialEq)]
pub enum NumericalIssue {
    NaN { count: usize },
    Infinity { count: usize },
    Underflow { count: usize },
    Overflow { count: usize },
}

/// Check weights for numerical issues
pub fn check_weights(weights: &[Array2<f32>]) -> Vec<NumericalIssue> {
    let mut issues = Vec::new();
    
    for (layer_idx, weight_matrix) in weights.iter().enumerate() {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut underflow_count = 0;
        let mut overflow_count = 0;
        
        for &value in weight_matrix.iter() {
            if value.is_nan() {
                nan_count += 1;
            } else if value.is_infinite() {
                inf_count += 1;
            } else if value != 0.0 && value.abs() < 1e-38 {
                underflow_count += 1;
            } else if value.abs() > 1e38 {
                overflow_count += 1;
            }
        }
        
        if nan_count > 0 {
            issues.push(NumericalIssue::NaN { count: nan_count });
            eprintln!("Layer {}: Found {} NaN values in weights", layer_idx, nan_count);
        }
        if inf_count > 0 {
            issues.push(NumericalIssue::Infinity { count: inf_count });
            eprintln!("Layer {}: Found {} infinite values in weights", layer_idx, inf_count);
        }
        if underflow_count > 0 {
            issues.push(NumericalIssue::Underflow { count: underflow_count });
            eprintln!("Layer {}: Found {} underflow values in weights", layer_idx, underflow_count);
        }
        if overflow_count > 0 {
            issues.push(NumericalIssue::Overflow { count: overflow_count });
            eprintln!("Layer {}: Found {} overflow risk values in weights", layer_idx, overflow_count);
        }
    }
    
    issues
}

/// Check gradients for numerical issues
pub fn check_gradients(gradients: &[Array2<f32>]) -> Vec<NumericalIssue> {
    let mut issues = Vec::new();
    
    for (layer_idx, grad_matrix) in gradients.iter().enumerate() {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut underflow_count = 0;
        let mut overflow_count = 0;
        
        for &value in grad_matrix.iter() {
            if value.is_nan() {
                nan_count += 1;
            } else if value.is_infinite() {
                inf_count += 1;
            } else if value != 0.0 && value.abs() < 1e-7 {
                underflow_count += 1;
            } else if value.abs() > 1e3 {
                overflow_count += 1;
            }
        }
        
        if nan_count > 0 {
            issues.push(NumericalIssue::NaN { count: nan_count });
            eprintln!("Layer {}: Found {} NaN values in gradients", layer_idx, nan_count);
        }
        if inf_count > 0 {
            issues.push(NumericalIssue::Infinity { count: inf_count });
            eprintln!("Layer {}: Found {} infinite values in gradients", layer_idx, inf_count);
        }
        if underflow_count > 0 {
            issues.push(NumericalIssue::Underflow { count: underflow_count });
            eprintln!("Layer {}: Found {} vanishing gradients", layer_idx, underflow_count);
        }
        if overflow_count > 0 {
            issues.push(NumericalIssue::Overflow { count: overflow_count });
            eprintln!("Layer {}: Found {} exploding gradients", layer_idx, overflow_count);
        }
    }
    
    issues
}

/// Sanitize an array by replacing NaN and Inf values
pub fn sanitize_array(array: &mut Array2<f32>, nan_replacement: f32, inf_replacement: f32) {
    array.mapv_inplace(|x| {
        if x.is_nan() {
            nan_replacement
        } else if x.is_infinite() {
            if x.is_sign_positive() {
                inf_replacement
            } else {
                -inf_replacement
            }
        } else {
            x
        }
    });
}

/// Check if values are in a reasonable range
pub fn check_value_range(array: ArrayView2<f32>, min_val: f32, max_val: f32) -> bool {
    array.iter().all(|&x| x >= min_val && x <= max_val)
}

/// Compute stability metrics for monitoring training
#[derive(Debug, Clone)]
pub struct StabilityMetrics {
    pub weight_update_ratio: f32,
    pub gradient_variance: f32,
    pub activation_saturation_rate: f32,
    pub parameter_norm: f32,
}

/// Compute stability metrics for a network update
pub fn compute_stability_metrics(
    old_weights: &[Array2<f32>],
    new_weights: &[Array2<f32>],
    gradients: &[Array2<f32>],
    activations: &[Array1<f32>],
) -> StabilityMetrics {
    // Weight update ratio
    let mut total_weight_change = 0.0;
    let mut total_weight_norm = 0.0;
    
    for (old, new) in old_weights.iter().zip(new_weights.iter()) {
        let change = (new - old).mapv(|x| x * x).sum();
        let norm = old.mapv(|x| x * x).sum();
        total_weight_change += change;
        total_weight_norm += norm;
    }
    
    let weight_update_ratio = if total_weight_norm > 0.0 {
        (total_weight_change / total_weight_norm).sqrt()
    } else {
        0.0
    };
    
    // Gradient variance
    let mut all_gradients = Vec::new();
    for grad in gradients {
        all_gradients.extend(grad.iter().copied());
    }
    
    let grad_mean = all_gradients.iter().sum::<f32>() / all_gradients.len() as f32;
    let gradient_variance = all_gradients.iter()
        .map(|&x| (x - grad_mean).powi(2))
        .sum::<f32>() / all_gradients.len() as f32;
    
    // Activation saturation (for sigmoid/tanh)
    let mut saturation_count = 0;
    let mut total_activations = 0;
    
    for act_layer in activations {
        for &act in act_layer.iter() {
            if act.abs() > 0.95 {
                saturation_count += 1;
            }
            total_activations += 1;
        }
    }
    
    let activation_saturation_rate = if total_activations > 0 {
        saturation_count as f32 / total_activations as f32
    } else {
        0.0
    };
    
    // Parameter norm
    let parameter_norm = new_weights.iter()
        .map(|w| w.mapv(|x| x * x).sum())
        .sum::<f32>()
        .sqrt();
    
    StabilityMetrics {
        weight_update_ratio,
        gradient_variance,
        activation_saturation_rate,
        parameter_norm,
    }
}

/// Helper to detect if training is unstable
pub fn is_training_stable(metrics: &StabilityMetrics) -> bool {
    metrics.weight_update_ratio < 0.1 &&
    metrics.gradient_variance < 1.0 &&
    metrics.activation_saturation_rate < 0.5 &&
    metrics.parameter_norm < 1000.0
}