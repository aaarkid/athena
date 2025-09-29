use std::collections::VecDeque;
use serde::{Serialize, Deserialize};

/// Stores training metrics over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Loss values over time
    pub losses: VecDeque<f32>,
    
    /// Rewards per episode
    pub episode_rewards: VecDeque<f32>,
    
    /// Episode lengths
    pub episode_lengths: VecDeque<usize>,
    
    /// Q-value estimates
    pub q_values: VecDeque<f32>,
    
    /// Learning rate over time
    pub learning_rates: VecDeque<f32>,
    
    /// Epsilon values (for exploration)
    pub epsilons: VecDeque<f32>,
    
    /// Gradient norms
    pub gradient_norms: VecDeque<f32>,
    
    /// Weight norms by layer
    pub weight_norms: Vec<VecDeque<f32>>,
    
    /// Custom metrics
    pub custom_metrics: std::collections::HashMap<String, VecDeque<f32>>,
}

impl TrainingMetrics {
    pub fn new(num_layers: usize, history_size: usize) -> Self {
        let mut weight_norms = Vec::new();
        for _ in 0..num_layers {
            weight_norms.push(VecDeque::with_capacity(history_size));
        }
        
        TrainingMetrics {
            losses: VecDeque::with_capacity(history_size),
            episode_rewards: VecDeque::with_capacity(history_size),
            episode_lengths: VecDeque::with_capacity(history_size),
            q_values: VecDeque::with_capacity(history_size),
            learning_rates: VecDeque::with_capacity(history_size),
            epsilons: VecDeque::with_capacity(history_size),
            gradient_norms: VecDeque::with_capacity(history_size),
            weight_norms,
            custom_metrics: std::collections::HashMap::new(),
        }
    }
}

/// Tracks metrics during training
pub struct MetricsTracker {
    metrics: TrainingMetrics,
    history_size: usize,
    
    // Episode tracking
    current_episode_reward: f32,
    current_episode_length: usize,
    episode_count: usize,
    
    // Step tracking
    total_steps: usize,
}

impl MetricsTracker {
    pub fn new(num_layers: usize, history_size: usize) -> Self {
        MetricsTracker {
            metrics: TrainingMetrics::new(num_layers, history_size),
            history_size,
            current_episode_reward: 0.0,
            current_episode_length: 0,
            episode_count: 0,
            total_steps: 0,
        }
    }
    
    /// Record a training loss
    pub fn record_loss(&mut self, loss: f32) {
        if self.metrics.losses.len() >= self.history_size {
            self.metrics.losses.pop_front();
        }
        self.metrics.losses.push_back(loss);
    }
    
    /// Record a Q-value estimate
    pub fn record_q_value(&mut self, q_value: f32) {
        if self.metrics.q_values.len() >= self.history_size {
            self.metrics.q_values.pop_front();
        }
        self.metrics.q_values.push_back(q_value);
    }
    
    /// Record learning rate
    pub fn record_learning_rate(&mut self, lr: f32) {
        if self.metrics.learning_rates.len() >= self.history_size {
            self.metrics.learning_rates.pop_front();
        }
        self.metrics.learning_rates.push_back(lr);
    }
    
    /// Record epsilon value
    pub fn record_epsilon(&mut self, epsilon: f32) {
        if self.metrics.epsilons.len() >= self.history_size {
            self.metrics.epsilons.pop_front();
        }
        self.metrics.epsilons.push_back(epsilon);
    }
    
    /// Record gradient norm
    pub fn record_gradient_norm(&mut self, norm: f32) {
        if self.metrics.gradient_norms.len() >= self.history_size {
            self.metrics.gradient_norms.pop_front();
        }
        self.metrics.gradient_norms.push_back(norm);
    }
    
    /// Record weight norm for a specific layer
    pub fn record_weight_norm(&mut self, layer_idx: usize, norm: f32) {
        if layer_idx < self.metrics.weight_norms.len() {
            let layer_norms = &mut self.metrics.weight_norms[layer_idx];
            if layer_norms.len() >= self.history_size {
                layer_norms.pop_front();
            }
            layer_norms.push_back(norm);
        }
    }
    
    /// Record a custom metric
    pub fn record_custom(&mut self, name: &str, value: f32) {
        let metric = self.metrics.custom_metrics
            .entry(name.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.history_size));
        
        if metric.len() >= self.history_size {
            metric.pop_front();
        }
        metric.push_back(value);
    }
    
    /// Start a new episode
    pub fn start_episode(&mut self) {
        self.current_episode_reward = 0.0;
        self.current_episode_length = 0;
    }
    
    /// Record a step within an episode
    pub fn step(&mut self, reward: f32) {
        self.current_episode_reward += reward;
        self.current_episode_length += 1;
        self.total_steps += 1;
    }
    
    /// End the current episode
    pub fn end_episode(&mut self) {
        if self.metrics.episode_rewards.len() >= self.history_size {
            self.metrics.episode_rewards.pop_front();
        }
        self.metrics.episode_rewards.push_back(self.current_episode_reward);
        
        if self.metrics.episode_lengths.len() >= self.history_size {
            self.metrics.episode_lengths.pop_front();
        }
        self.metrics.episode_lengths.push_back(self.current_episode_length);
        
        self.episode_count += 1;
    }
    
    /// Get a reference to the metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }
    
    /// Get episode count
    pub fn episode_count(&self) -> usize {
        self.episode_count
    }
    
    /// Get total steps
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
    
    /// Get recent average loss
    pub fn avg_loss(&self, window: usize) -> Option<f32> {
        if self.metrics.losses.is_empty() {
            return None;
        }
        
        let n = window.min(self.metrics.losses.len());
        let sum: f32 = self.metrics.losses.iter().rev().take(n).sum();
        Some(sum / n as f32)
    }
    
    /// Get recent average episode reward
    pub fn avg_episode_reward(&self, window: usize) -> Option<f32> {
        if self.metrics.episode_rewards.is_empty() {
            return None;
        }
        
        let n = window.min(self.metrics.episode_rewards.len());
        let sum: f32 = self.metrics.episode_rewards.iter().rev().take(n).sum();
        Some(sum / n as f32)
    }
    
    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics = TrainingMetrics::new(self.metrics.weight_norms.len(), self.history_size);
        self.current_episode_reward = 0.0;
        self.current_episode_length = 0;
        self.episode_count = 0;
        self.total_steps = 0;
    }
    
    /// Save metrics to file
    pub fn save(&self, path: &str) -> crate::error::Result<()> {
        let serialized = serde_json::to_string_pretty(&self.metrics)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    /// Load metrics from file
    pub fn load(&mut self, path: &str) -> crate::error::Result<()> {
        let data = std::fs::read_to_string(path)?;
        self.metrics = serde_json::from_str(&data)?;
        Ok(())
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new(0, 1000)
    }
}