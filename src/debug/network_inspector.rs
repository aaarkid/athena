use crate::network::NeuralNetwork;
use crate::metrics::statistics::{Statistics, check_numerical_issues};
use ndarray::{Array1, Array2};

/// Inspector for debugging neural networks
pub struct NetworkInspector {
    /// History of weight statistics per layer
    weight_stats_history: Vec<Vec<Statistics>>,
    
    /// History of activation statistics per layer
    activation_stats_history: Vec<Vec<Statistics>>,
    
    /// Dead neuron tracking
    dead_neurons_per_layer: Vec<Vec<bool>>,
    
    /// Numerical issues tracking
    numerical_issues_count: usize,
}

impl NetworkInspector {
    pub fn new() -> Self {
        NetworkInspector {
            weight_stats_history: Vec::new(),
            activation_stats_history: Vec::new(),
            dead_neurons_per_layer: Vec::new(),
            numerical_issues_count: 0,
        }
    }
    
    /// Inspect network weights
    pub fn inspect_weights(&mut self, network: &NeuralNetwork) -> Vec<Statistics> {
        let stats: Vec<Statistics> = network.layers.iter()
            .map(|layer| Statistics::from_array2d(layer.weights.view()))
            .collect();
        
        self.weight_stats_history.push(stats.clone());
        
        // Check for numerical issues
        for layer in &network.layers {
            let (has_issues, nan_count, inf_count) = check_numerical_issues(layer.weights.view());
            if has_issues {
                self.numerical_issues_count += 1;
                eprintln!("WARNING: Numerical issues detected in weights - NaN: {}, Inf: {}", 
                         nan_count, inf_count);
            }
        }
        
        stats
    }
    
    /// Inspect layer activations
    pub fn inspect_activations(&mut self, activations: &[Array1<f32>], threshold: f32) -> Vec<Statistics> {
        let stats: Vec<Statistics> = activations.iter()
            .map(|act| Statistics::from_array(act.view()))
            .collect();
        
        self.activation_stats_history.push(stats.clone());
        
        // Check for dead neurons
        let mut dead_neurons = Vec::new();
        for (i, act) in activations.iter().enumerate() {
            let dead = act.iter().all(|&x| x.abs() < threshold);
            dead_neurons.push(dead);
            
            if dead {
                eprintln!("WARNING: Layer {} appears to have dead neurons", i);
            }
        }
        
        self.dead_neurons_per_layer.push(dead_neurons);
        
        stats
    }
    
    /// Get weight change statistics between inspections
    pub fn weight_change_stats(&self) -> Option<Vec<Statistics>> {
        if self.weight_stats_history.len() < 2 {
            return None;
        }
        
        let current = self.weight_stats_history.last()?;
        let previous = self.weight_stats_history.get(self.weight_stats_history.len() - 2)?;
        
        let changes: Vec<Statistics> = current.iter().zip(previous.iter())
            .map(|(curr, prev)| {
                Statistics {
                    mean: (curr.mean - prev.mean).abs(),
                    std: (curr.std - prev.std).abs(),
                    min: (curr.min - prev.min).abs(),
                    max: (curr.max - prev.max).abs(),
                    count: curr.count,
                }
            })
            .collect();
        
        Some(changes)
    }
    
    /// Generate a report of network health
    pub fn generate_report(&self) -> NetworkHealthReport {
        let num_dead_neurons: usize = self.dead_neurons_per_layer.last()
            .map(|layer| layer.iter().filter(|&&x| x).count())
            .unwrap_or(0);
        
        let latest_weight_stats = self.weight_stats_history.last().cloned();
        let latest_activation_stats = self.activation_stats_history.last().cloned();
        
        NetworkHealthReport {
            num_inspections: self.weight_stats_history.len(),
            numerical_issues_count: self.numerical_issues_count,
            num_dead_neurons,
            latest_weight_stats,
            latest_activation_stats,
            weight_change_stats: self.weight_change_stats(),
        }
    }
    
    /// Clear inspection history
    pub fn clear_history(&mut self) {
        self.weight_stats_history.clear();
        self.activation_stats_history.clear();
        self.dead_neurons_per_layer.clear();
        self.numerical_issues_count = 0;
    }
    
    /// Check for gradient flow issues
    pub fn check_gradient_flow(&self, gradients: &[Array2<f32>]) -> GradientFlowReport {
        let mut vanishing_layers = Vec::new();
        let mut exploding_layers = Vec::new();
        
        for (i, grad) in gradients.iter().enumerate() {
            let stats = Statistics::from_array2d(grad.view());
            
            if stats.max < 1e-7 {
                vanishing_layers.push(i);
            }
            
            if stats.max > 1e3 {
                exploding_layers.push(i);
            }
        }
        
        let healthy = vanishing_layers.is_empty() && exploding_layers.is_empty();
        
        GradientFlowReport {
            vanishing_layers,
            exploding_layers,
            healthy,
        }
    }
}

/// Report on network health
#[derive(Debug, Clone)]
pub struct NetworkHealthReport {
    pub num_inspections: usize,
    pub numerical_issues_count: usize,
    pub num_dead_neurons: usize,
    pub latest_weight_stats: Option<Vec<Statistics>>,
    pub latest_activation_stats: Option<Vec<Statistics>>,
    pub weight_change_stats: Option<Vec<Statistics>>,
}

/// Report on gradient flow
#[derive(Debug, Clone)]
pub struct GradientFlowReport {
    pub vanishing_layers: Vec<usize>,
    pub exploding_layers: Vec<usize>,
    pub healthy: bool,
}

impl Default for NetworkInspector {
    fn default() -> Self {
        Self::new()
    }
}