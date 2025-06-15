use crate::replay_buffer::{ReplayBuffer, PrioritizedReplayBuffer, PriorityMethod};
use crate::error::{Result, AthenaError};

/// Builder for ReplayBuffer
pub struct ReplayBufferBuilder {
    capacity: Option<usize>,
}

impl ReplayBufferBuilder {
    /// Create a new replay buffer builder
    pub fn new() -> Self {
        ReplayBufferBuilder { capacity: None }
    }
    
    /// Set the capacity
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }
    
    /// Build the replay buffer
    pub fn build(self) -> Result<ReplayBuffer> {
        let capacity = self.capacity.ok_or_else(|| AthenaError::InvalidParameter {
            name: "capacity".to_string(),
            reason: "Capacity not specified".to_string(),
        })?;
        
        if capacity == 0 {
            return Err(AthenaError::InvalidParameter {
                name: "capacity".to_string(),
                reason: "Capacity must be greater than 0".to_string(),
            });
        }
        
        Ok(ReplayBuffer::new(capacity))
    }
}

impl Default for ReplayBufferBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for PrioritizedReplayBuffer
pub struct PrioritizedReplayBufferBuilder {
    capacity: Option<usize>,
    priority_method: PriorityMethod,
    epsilon: f32,
}

impl PrioritizedReplayBufferBuilder {
    /// Create a new prioritized replay buffer builder
    pub fn new() -> Self {
        PrioritizedReplayBufferBuilder {
            capacity: None,
            priority_method: PriorityMethod::Uniform,
            epsilon: 1e-6,
        }
    }
    
    /// Set the capacity
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }
    
    /// Use uniform sampling (no prioritization)
    pub fn uniform(mut self) -> Self {
        self.priority_method = PriorityMethod::Uniform;
        self
    }
    
    /// Use proportional prioritization
    pub fn proportional(mut self, alpha: f32) -> Self {
        self.priority_method = PriorityMethod::Proportional { alpha };
        self
    }
    
    /// Use rank-based prioritization
    pub fn rank_based(mut self, alpha: f32) -> Self {
        self.priority_method = PriorityMethod::RankBased { alpha };
        self
    }
    
    /// Set epsilon for numerical stability
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
    
    /// Build the prioritized replay buffer
    pub fn build(self) -> Result<PrioritizedReplayBuffer> {
        let capacity = self.capacity.ok_or_else(|| AthenaError::InvalidParameter {
            name: "capacity".to_string(),
            reason: "Capacity not specified".to_string(),
        })?;
        
        if capacity == 0 {
            return Err(AthenaError::InvalidParameter {
                name: "capacity".to_string(),
                reason: "Capacity must be greater than 0".to_string(),
            });
        }
        
        if self.epsilon <= 0.0 {
            return Err(AthenaError::InvalidParameter {
                name: "epsilon".to_string(),
                reason: "Epsilon must be positive".to_string(),
            });
        }
        
        // For now, we'll accept the default epsilon value
        // In the future, we could add a setter method to PrioritizedReplayBuffer
        Ok(PrioritizedReplayBuffer::new(capacity, self.priority_method))
    }
}

impl Default for PrioritizedReplayBufferBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_replay_buffer_builder() {
        let buffer = ReplayBufferBuilder::new()
            .capacity(1000)
            .build()
            .unwrap();
        
        assert_eq!(buffer.capacity(), 1000);
    }
    
    #[test]
    fn test_prioritized_buffer_builder() {
        // Uniform
        let buffer = PrioritizedReplayBufferBuilder::new()
            .capacity(1000)
            .uniform()
            .build()
            .unwrap();
        
        assert_eq!(buffer.capacity(), 1000);
        
        // Proportional
        let buffer = PrioritizedReplayBufferBuilder::new()
            .capacity(500)
            .proportional(0.6)
            .epsilon(1e-5)
            .build()
            .unwrap();
        
        assert_eq!(buffer.capacity(), 500);
        
        // Rank-based
        let buffer = PrioritizedReplayBufferBuilder::new()
            .capacity(2000)
            .rank_based(0.7)
            .build()
            .unwrap();
        
        assert_eq!(buffer.capacity(), 2000);
    }
    
    #[test]
    fn test_builder_errors() {
        // No capacity
        let result = ReplayBufferBuilder::new().build();
        assert!(result.is_err());
        
        // Zero capacity
        let result = ReplayBufferBuilder::new().capacity(0).build();
        assert!(result.is_err());
        
        // Invalid epsilon
        let result = PrioritizedReplayBufferBuilder::new()
            .capacity(100)
            .epsilon(0.0)
            .build();
        assert!(result.is_err());
    }
}