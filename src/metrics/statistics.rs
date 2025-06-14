use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Statistics for a collection of values
#[derive(Debug, Clone)]
pub struct Statistics {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub count: usize,
}

impl Statistics {
    /// Compute statistics from a slice of values
    pub fn from_slice(values: &[f32]) -> Self {
        if values.is_empty() {
            return Statistics {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
            };
        }
        
        let count = values.len();
        let sum: f32 = values.iter().sum();
        let mean = sum / count as f32;
        
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / count as f32;
        let std = variance.sqrt();
        
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        Statistics {
            mean,
            std,
            min,
            max,
            count,
        }
    }
    
    /// Compute statistics from an array view
    pub fn from_array(array: ArrayView1<f32>) -> Self {
        Self::from_slice(array.as_slice().unwrap_or(&[]))
    }
    
    /// Compute statistics for a 2D array (across all elements)
    pub fn from_array2d(array: ArrayView2<f32>) -> Self {
        let values: Vec<f32> = array.iter().copied().collect();
        Self::from_slice(&values)
    }
}

/// Running statistics that can be updated incrementally
#[derive(Debug, Clone)]
pub struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    min: f32,
    max: f32,
}

impl RunningStats {
    pub fn new() -> Self {
        RunningStats {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }
    
    /// Update with a new value
    pub fn update(&mut self, value: f32) {
        self.count += 1;
        let delta = value as f64 - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value as f64 - self.mean;
        self.m2 += delta * delta2;
        
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }
    
    /// Update with multiple values
    pub fn update_batch(&mut self, values: &[f32]) {
        for &value in values {
            self.update(value);
        }
    }
    
    /// Get the mean
    pub fn mean(&self) -> f32 {
        self.mean as f32
    }
    
    /// Get the variance
    pub fn variance(&self) -> f32 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count - 1) as f64) as f32
        }
    }
    
    /// Get the standard deviation
    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }
    
    /// Get the minimum value
    pub fn min(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.min
        }
    }
    
    /// Get the maximum value
    pub fn max(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.max
        }
    }
    
    /// Get the count
    pub fn count(&self) -> usize {
        self.count
    }
    
    /// Reset the statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
    
    /// Get all statistics as a Statistics struct
    pub fn to_statistics(&self) -> Statistics {
        Statistics {
            mean: self.mean(),
            std: self.std(),
            min: self.min(),
            max: self.max(),
            count: self.count,
        }
    }
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute gradient statistics for monitoring
pub fn gradient_statistics(gradients: &[Array2<f32>]) -> Vec<Statistics> {
    gradients.iter()
        .map(|grad| Statistics::from_array2d(grad.view()))
        .collect()
}

/// Compute weight statistics for monitoring
pub fn weight_statistics(weights: &[Array2<f32>]) -> Vec<Statistics> {
    weights.iter()
        .map(|w| Statistics::from_array2d(w.view()))
        .collect()
}

/// Check for numerical issues in an array
pub fn check_numerical_issues(array: ArrayView2<f32>) -> (bool, usize, usize) {
    let mut has_nan = false;
    let mut has_inf = false;
    let mut nan_count = 0;
    let mut inf_count = 0;
    
    for &value in array.iter() {
        if value.is_nan() {
            has_nan = true;
            nan_count += 1;
        }
        if value.is_infinite() {
            has_inf = true;
            inf_count += 1;
        }
    }
    
    (has_nan || has_inf, nan_count, inf_count)
}

/// Compute the Frobenius norm of a matrix
pub fn frobenius_norm(array: ArrayView2<f32>) -> f32 {
    array.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Compute the L1 norm of a matrix
pub fn l1_norm(array: ArrayView2<f32>) -> f32 {
    array.iter().map(|&x| x.abs()).sum()
}

/// Compute the L-infinity norm of a matrix
pub fn linf_norm(array: ArrayView2<f32>) -> f32 {
    array.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
}

/// Check for dead neurons (neurons that never activate)
pub fn dead_neuron_check(activations: ArrayView2<f32>, threshold: f32) -> Vec<bool> {
    let (_, num_neurons) = activations.dim();
    let mut dead_neurons = vec![true; num_neurons];
    
    for row in activations.axis_iter(ndarray::Axis(0)) {
        for (i, &value) in row.iter().enumerate() {
            if value.abs() > threshold {
                dead_neurons[i] = false;
            }
        }
    }
    
    dead_neurons
}

/// Compute activation statistics per layer
pub fn activation_statistics(activations: &[Array1<f32>]) -> Vec<Statistics> {
    activations.iter()
        .map(|act| Statistics::from_array(act.view()))
        .collect()
}