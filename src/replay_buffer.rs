use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq)]
pub struct Experience {
    pub state: Array1<f32>,
    pub next_state: Array1<f32>,
    pub action: usize,
    pub reward: f32,
    pub done: bool,
}

#[derive(Clone)]
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() == self.capacity && self.capacity > 0 {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        let mut rng = thread_rng();
    
        let (slice1, slice2) = self.buffer.as_slices();
        let mut indices = (0..self.buffer.len()).collect::<Vec<usize>>();
        indices.shuffle(&mut rng);
    
        if batch_size > indices.len() {
            // Not enough samples in the buffer yet, return all of them:
            indices.into_iter().map(|i| {
                if i < slice1.len() {
                    &slice1[i]
                } else {
                    &slice2[i - slice1.len()]
                }
            }).collect::<Vec<_>>()
        } else {
            indices.into_iter().take(batch_size).map(|i| {
                if i < slice1.len() {
                    &slice1[i]
                } else {
                    &slice2[i - slice1.len()]
                }
            }).collect::<Vec<_>>()
        }
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
    
    /// Sample without importance weights (simpler interface)
    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        let (experiences, _, _) = self.sample_with_weights(batch_size, 1.0);
        experiences
    }
    
    /// Get the number of experiences in the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_replay_buffer_add_and_sample() {
        let mut buffer = ReplayBuffer::new(5);
        
        for i in 0..7 {
            let exp = Experience {
                state: array![i as f32],
                action: i % 2,
                reward: i as f32,
                next_state: array![(i + 1) as f32],
                done: false,
            };
            buffer.add(exp);
        }
        
        assert_eq!(buffer.len(), 5); // Should only keep last 5
        
        let samples = buffer.sample(3);
        assert_eq!(samples.len(), 3);
    }
    
    #[test]
    fn test_prioritized_replay_buffer() {
        let mut buffer = PrioritizedReplayBuffer::new(5, PriorityMethod::Proportional { alpha: 0.6 });
        
        for i in 0..3 {
            let exp = Experience {
                state: array![i as f32],
                action: i,
                reward: i as f32,
                next_state: array![(i + 1) as f32],
                done: false,
            };
            buffer.add_with_priority(exp, (i + 1) as f32);
        }
        
        let (samples, weights, indices) = buffer.sample_with_weights(2, 0.4);
        assert_eq!(samples.len(), 2);
        assert_eq!(weights.len(), 2);
        assert_eq!(indices.len(), 2);
    }
}