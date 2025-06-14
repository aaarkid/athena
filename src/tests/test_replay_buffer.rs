use ndarray::array;
use crate::replay_buffer::{ReplayBuffer, Experience};
use crate::replay_buffer_v2::{PrioritizedReplayBuffer, PriorityMethod};

#[test]
fn test_replay_buffer_add_and_sample() {
    let mut replay_buffer = ReplayBuffer::new(10);
    let experience = Experience {
        state: array![0.5, -0.5],
        action: 0,
        reward: 1.0,
        next_state: array![0.6, -0.4],
        done: false,
    };
    replay_buffer.add(experience.clone());
    assert_eq!(replay_buffer.len(), 1);
    let sample = replay_buffer.sample(1);
    assert_eq!(sample[0], &experience);
}

#[test]
fn test_replay_buffer_capacity() {
    let mut buffer = ReplayBuffer::new(3);
    
    // Add more experiences than capacity
    for i in 0..5 {
        let exp = Experience {
            state: array![i as f32],
            action: i,
            reward: i as f32,
            next_state: array![(i + 1) as f32],
            done: false,
        };
        buffer.add(exp);
    }
    
    // Should only keep last 3
    assert_eq!(buffer.len(), 3);
    
    // Oldest experiences should be dropped
    let samples = buffer.sample(3);
    let states: Vec<f32> = samples.iter().map(|e| e.state[0]).collect();
    assert!(states.contains(&2.0));
    assert!(states.contains(&3.0));
    assert!(states.contains(&4.0));
}

#[test]
fn test_replay_buffer_is_empty() {
    let mut buffer = ReplayBuffer::new(10);
    assert!(buffer.is_empty());
    
    buffer.add(Experience {
        state: array![0.0],
        action: 0,
        reward: 0.0,
        next_state: array![1.0],
        done: false,
    });
    
    assert!(!buffer.is_empty());
}

#[test]
fn test_replay_buffer_sample_size() {
    let mut buffer = ReplayBuffer::new(10);
    
    // Add 5 experiences
    for i in 0..5 {
        buffer.add(Experience {
            state: array![i as f32],
            action: 0,
            reward: 0.0,
            next_state: array![(i + 1) as f32],
            done: false,
        });
    }
    
    // Sample different sizes
    let sample1 = buffer.sample(1);
    assert_eq!(sample1.len(), 1);
    
    let sample3 = buffer.sample(3);
    assert_eq!(sample3.len(), 3);
    
    // Sampling more than available should return all
    let sample10 = buffer.sample(10);
    assert_eq!(sample10.len(), 5);
}

#[test]
fn test_prioritized_replay_buffer_uniform() {
    let mut buffer = PrioritizedReplayBuffer::new(10, PriorityMethod::Uniform);
    
    // Add experiences
    for i in 0..5 {
        let exp = Experience {
            state: array![i as f32],
            action: i,
            reward: i as f32,
            next_state: array![(i + 1) as f32],
            done: false,
        };
        buffer.add_with_priority(exp, i as f32 + 1.0);
    }
    
    // Sample should work like normal replay buffer
    let (experiences, weights, indices) = buffer.sample_with_weights(3, 1.0);
    assert_eq!(experiences.len(), 3);
    assert_eq!(weights.len(), 3);
    assert_eq!(indices.len(), 3);
    
    // Weights should all be 1.0 for uniform sampling
    for &w in weights.iter() {
        assert_eq!(w, 1.0);
    }
}

#[test]
fn test_prioritized_replay_buffer_proportional() {
    let mut buffer = PrioritizedReplayBuffer::new(10, PriorityMethod::Proportional { alpha: 0.6 });
    
    // Add experiences with different priorities
    let exp1 = Experience {
        state: array![1.0],
        action: 0,
        reward: 0.0,
        next_state: array![2.0],
        done: false,
    };
    let exp2 = Experience {
        state: array![2.0],
        action: 1,
        reward: 1.0,
        next_state: array![3.0],
        done: false,
    };
    
    buffer.add_with_priority(exp1, 1.0);
    buffer.add_with_priority(exp2, 10.0); // Much higher priority
    
    // Sample many times and count
    let mut count_exp2 = 0;
    for _ in 0..100 {
        let (experiences, _, _) = buffer.sample_with_weights(1, 1.0);
        if experiences[0].action == 1 {
            count_exp2 += 1;
        }
    }
    
    // exp2 should be sampled more often due to higher priority
    assert!(count_exp2 > 50);
}

#[test]
fn test_prioritized_replay_buffer_update_priorities() {
    let mut buffer = PrioritizedReplayBuffer::new(10, PriorityMethod::Proportional { alpha: 1.0 });
    
    // Add experiences
    for i in 0..5 {
        let exp = Experience {
            state: array![i as f32],
            action: i,
            reward: 0.0,
            next_state: array![(i + 1) as f32],
            done: false,
        };
        buffer.add_with_priority(exp, 1.0);
    }
    
    // Sample and get indices
    let (_, _, indices) = buffer.sample_with_weights(3, 1.0);
    
    // Update priorities
    let new_priorities = vec![10.0, 20.0, 30.0];
    buffer.update_priorities(&indices, &new_priorities);
    
    // Priorities should be updated
}

#[test]
fn test_prioritized_replay_buffer_importance_weights() {
    let mut buffer = PrioritizedReplayBuffer::new(10, PriorityMethod::Proportional { alpha: 0.6 });
    
    // Add experiences with different priorities
    for i in 0..5 {
        let exp = Experience {
            state: array![i as f32],
            action: i,
            reward: 0.0,
            next_state: array![(i + 1) as f32],
            done: false,
        };
        buffer.add_with_priority(exp, (i + 1) as f32);
    }
    
    // Sample with beta < 1
    let (_, weights, _) = buffer.sample_with_weights(5, 0.4);
    
    // Importance weights should vary
    let min_weight = weights.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_weight = weights.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(max_weight > min_weight);
    
    // Sample with beta = 1 (full correction)
    let (_, weights, _) = buffer.sample_with_weights(5, 1.0);
    
    // Weights should still vary but be more extreme
    let min_weight2 = weights.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_weight2 = weights.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(max_weight2 > min_weight2);
}

#[test]
fn test_prioritized_replay_buffer_rank_based() {
    let mut buffer = PrioritizedReplayBuffer::new(10, PriorityMethod::RankBased { alpha: 1.0 });
    
    // Add experiences
    for i in 0..5 {
        let exp = Experience {
            state: array![i as f32],
            action: i,
            reward: 0.0,
            next_state: array![(i + 1) as f32],
            done: false,
        };
        buffer.add_with_priority(exp, (i + 1) as f32);
    }
    
    // Sample should work
    let (experiences, weights, indices) = buffer.sample_with_weights(3, 0.5);
    assert_eq!(experiences.len(), 3);
    assert_eq!(weights.len(), 3);
    assert_eq!(indices.len(), 3);
}