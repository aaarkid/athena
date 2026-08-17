use ndarray::array;
use crate::replay_buffer::{ReplayBuffer, Experience, PrioritizedReplayBuffer, PriorityMethod};

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
    
    // sample_with_weights caps the batch at the buffer length, so one call draws at
    // most 5 and can legitimately draw the same index every time. Repeat and take the
    // widest spread seen, which converges on (p_max / p_min)^beta.
    let widest_spread = |beta: f32| -> f32 {
        let mut widest = 1.0f32;
        for _ in 0..50 {
            let (_, weights, _) = buffer.sample_with_weights(5, beta);
            assert!(weights.iter().all(|w| w.is_finite() && *w > 0.0));

            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            widest = widest.max(max / min);
        }
        widest
    };

    // The weight is (N * p)^-beta, so beta is the exponent on the spread between the
    // most and least likely entry. Raising it has to widen that spread.
    let partial = widest_spread(0.4);
    let full = widest_spread(1.0);

    assert!(partial > 1.0, "weights did not vary at beta 0.4");
    assert!(
        full > partial * 1.2,
    );
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


#[test]
fn a_high_priority_entry_is_drawn_far_more_often() {
    let mut buffer = PrioritizedReplayBuffer::new(10, PriorityMethod::Proportional { alpha: 1.0 });

    for i in 0..5 {
        buffer.add_with_priority(
            Experience {
                state: array![i as f32],
                action: i,
                reward: 0.0,
                next_state: array![(i + 1) as f32],
                done: false,
            },
            if i == 0 { 100.0 } else { 1.0 },
        );
    }

    let mut hits = 0;
    for _ in 0..200 {
        let (experiences, _, _) = buffer.sample_with_weights(1, 1.0);
        if experiences[0].action == 0 {
            hits += 1;
        }
    }

    // Slot 0 carries 100 of the 104 total priority, so anything near chance means the
    // priorities are not reaching the sampler
    assert!(hits > 100, "high priority entry drawn {} times in 200", hits);
}

#[test]
fn priorities_still_land_on_the_right_entry_after_an_eviction() {
    let mut buffer = PrioritizedReplayBuffer::new(4, PriorityMethod::Proportional { alpha: 1.0 });

    // Slots 0 to 3, carrying actions 0 to 3
    for i in 0..4 {
        buffer.add_with_priority(
            Experience {
                state: array![i as f32],
                action: i,
                reward: 0.0,
                next_state: array![(i + 1) as f32],
                done: false,
            },
            1.0,
        );
    }
    assert_eq!(buffer.first_slot(), 0);

    // This evicts slot 0 and shifts every position down by one. A positional index
    // captured before the add would now point at the wrong experience.
    buffer.add_with_priority(
        Experience {
            state: array![99.0],
            action: 99,
            reward: 0.0,
            next_state: array![100.0],
            done: false,
        },
        1.0,
    );
    assert_eq!(buffer.first_slot(), 1);

    // Slot 2 still means the experience carrying action 2, even though it has moved
    // from position 2 to position 1
    buffer.update_priorities(&[2], &[100.0]);
    // Slot 0 is gone, so this must be dropped rather than written to position 0
    buffer.update_priorities(&[0], &[100.0]);

    let mut counts = std::collections::HashMap::new();
    for _ in 0..400 {
        let (drawn, _, _) = buffer.sample_with_weights(1, 1.0);
        *counts.entry(drawn[0].action).or_insert(0) += 1;
    }

    let raised = counts.get(&2).copied().unwrap_or(0);
    assert!(
        raised > 300,
        "action 2 carries 100 of 103 total priority but was drawn {} times in 400",
        raised
    );

    // Action 1 sits at position 0, which is where a stale positional index would have
    // written, so it must still be at its original priority
    let action_one = counts.get(&1).copied().unwrap_or(0);
    assert!(
        action_one < 60,
        "action 1 was drawn {} times, so the evicted slot wrote to position 0",
        action_one
    );
}
