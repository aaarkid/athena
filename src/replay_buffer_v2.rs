use crate::replay_buffer::Experience;
use std::collections::VecDeque;
use rand::seq::SliceRandom;
use rand::Rng;

/// Priority method for experiences
#[derive(Clone, Copy, Debug)]
pub enum PriorityMethod {
    /// Uniform random sampling
    Uniform,
    /// Prioritized sampling based on TD error
    Proportional { alpha: f32 },
    /// Rank-based prioritization
    RankBased { alpha: f32 },
}

/// Enhanced replay buffer with prioritization support
pub struct PrioritizedReplayBuffer {
    /// The actual buffer storing experiences
    buffer: VecDeque<Experience>,
    /// Priorities for each experience  
    priorities: VecDeque<f32>,
    /// Maximum capacity
    capacity: usize,
    /// Priority method
    method: PriorityMethod,
    /// Small constant to ensure non-zero probabilities
    epsilon: f32,
    /// Maximum priority seen
    max_priority: f32,
}

impl PrioritizedReplayBuffer {
    /// Create a new prioritized replay buffer
    pub fn new(capacity: usize, method: PriorityMethod) -> Self {
        PrioritizedReplayBuffer {
            buffer: VecDeque::with_capacity(capacity),
            priorities: VecDeque::with_capacity(capacity),
            capacity,
            method,
            epsilon: 0.01,
            max_priority: 1.0,
        }
    }
    
    /// Add an experience with default priority
    pub fn add(&mut self, experience: Experience) {
        self.add_with_priority(experience, self.max_priority);
    }
    
    /// Add an experience with specific priority
    pub fn add_with_priority(&mut self, experience: Experience, priority: f32) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.priorities.pop_front();
        }
        
        self.buffer.push_back(experience);
        self.priorities.push_back(priority);
        
        if priority > self.max_priority {
            self.max_priority = priority;
        }
    }
    
    /// Update priorities for sampled experiences
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f32]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            if idx < self.priorities.len() {
                self.priorities[idx] = priority;
                if priority > self.max_priority {
                    self.max_priority = priority;
                }
            }
        }
    }
    
    /// Sample a batch of experiences with importance sampling weights
    pub fn sample_with_weights(&self, batch_size: usize, beta: f32) -> (Vec<&Experience>, Vec<f32>, Vec<usize>) {
        if self.buffer.is_empty() {
            return (vec![], vec![], vec![]);
        }
        
        let actual_batch_size = batch_size.min(self.buffer.len());
        
        match self.method {
            PriorityMethod::Uniform => {
                // Uniform sampling
                let mut rng = rand::thread_rng();
                let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
                indices.shuffle(&mut rng);
                indices.truncate(actual_batch_size);
                
                let experiences: Vec<&Experience> = indices.iter()
                    .map(|&i| &self.buffer[i])
                    .collect();
                let weights = vec![1.0; actual_batch_size];
                
                (experiences, weights, indices)
            }
            
            PriorityMethod::Proportional { alpha } => {
                // Proportional prioritization
                let priorities: Vec<f32> = self.priorities.iter()
                    .map(|&p| (p + self.epsilon).powf(alpha))
                    .collect();
                    
                let sum_priorities: f32 = priorities.iter().sum();
                let probabilities: Vec<f32> = priorities.iter()
                    .map(|&p| p / sum_priorities)
                    .collect();
                
                // Sample indices based on probabilities
                let mut rng = rand::thread_rng();
                let mut indices = Vec::with_capacity(actual_batch_size);
                let mut experiences = Vec::with_capacity(actual_batch_size);
                let mut weights = Vec::with_capacity(actual_batch_size);
                
                for _ in 0..actual_batch_size {
                    let r: f32 = rng.gen();
                    let mut cumsum = 0.0;
                    
                    for (i, &prob) in probabilities.iter().enumerate() {
                        cumsum += prob;
                        if r <= cumsum {
                            indices.push(i);
                            experiences.push(&self.buffer[i]);
                            
                            // Importance sampling weight
                            let weight = (self.buffer.len() as f32 * prob).powf(-beta);
                            weights.push(weight);
                            break;
                        }
                    }
                }
                
                // Normalize weights
                let max_weight = weights.iter().fold(0.0_f32, |max, &w| max.max(w));
                if max_weight > 0.0 {
                    for w in weights.iter_mut() {
                        *w /= max_weight;
                    }
                }
                
                (experiences, weights, indices)
            }
            
            PriorityMethod::RankBased { alpha } => {
                // Rank-based prioritization
                let mut indexed_priorities: Vec<(usize, f32)> = self.priorities.iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p))
                    .collect();
                    
                // Sort by priority (descending)
                indexed_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                // Calculate rank-based probabilities
                let n = self.buffer.len() as f32;
                let probabilities: Vec<f32> = (1..=self.buffer.len())
                    .map(|rank| 1.0 / (rank as f32).powf(alpha))
                    .collect();
                    
                let sum_probs: f32 = probabilities.iter().sum();
                let normalized_probs: Vec<f32> = probabilities.iter()
                    .map(|&p| p / sum_probs)
                    .collect();
                
                // Sample based on rank probabilities
                let mut rng = rand::thread_rng();
                let mut sampled_indices = Vec::with_capacity(actual_batch_size);
                let mut experiences = Vec::with_capacity(actual_batch_size);
                let mut weights = Vec::with_capacity(actual_batch_size);
                
                for _ in 0..actual_batch_size {
                    let r: f32 = rng.gen();
                    let mut cumsum = 0.0;
                    
                    for (rank, &prob) in normalized_probs.iter().enumerate() {
                        cumsum += prob;
                        if r <= cumsum {
                            let buffer_idx = indexed_priorities[rank].0;
                            sampled_indices.push(buffer_idx);
                            experiences.push(&self.buffer[buffer_idx]);
                            
                            // Importance sampling weight
                            let weight = (n * prob).powf(-beta);
                            weights.push(weight);
                            break;
                        }
                    }
                }
                
                // Normalize weights
                let max_weight = weights.iter().fold(0.0_f32, |max, &w| max.max(w));
                if max_weight > 0.0 {
                    for w in weights.iter_mut() {
                        *w /= max_weight;
                    }
                }
                
                (experiences, weights, sampled_indices)
            }
        }
    }
    
    /// Sample without importance weights (backward compatibility)
    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        let (experiences, _, _) = self.sample_with_weights(batch_size, 1.0);
        experiences
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}