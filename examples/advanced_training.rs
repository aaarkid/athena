/***
# Advanced Training Example

* Demonstrates advanced features:
  - Prioritized replay buffer
  - Learning rate scheduling
  - Gradient clipping
  - Batch normalization and dropout
  - Different weight initialization strategies
***/

use athena::{
    replay_buffer::{PrioritizedReplayBuffer, PriorityMethod},
    optimizer::{LearningRateScheduler, GradientClipper},
    layers::{DenseLayer, WeightInit},
    activations::Activation,
};
use ndarray::array;

fn main() {
    println!("=== Advanced Training Features Demo ===\n");
    
    // 1. Prioritized Replay Buffer
    println!("1. Prioritized Replay Buffer");
    let mut prioritized_buffer = PrioritizedReplayBuffer::new(
        1000, 
        PriorityMethod::Proportional { alpha: 0.6 }
    );
    
    // Add some dummy experiences with priorities
    for i in 0..10 {
        let experience = athena::replay_buffer::Experience {
            state: array![i as f32, i as f32 * 2.0],
            action: i % 2,
            reward: i as f32,
            next_state: array![(i + 1) as f32, (i + 1) as f32 * 2.0],
            done: i == 9,
        };
        let priority = 1.0 + i as f32 * 0.1; // Higher priority for later experiences
        prioritized_buffer.add_with_priority(experience, priority);
    }
    
    // Sample with importance weights
    let (experiences, weights, indices) = prioritized_buffer.sample_with_weights(5, 0.4);
    println!("  Sampled {} experiences with importance weights", experiences.len());
    println!("  Weights: {:?}", weights);
    println!("  Indices: {:?}\n", indices);
    
    // 2. Learning Rate Scheduling
    println!("2. Learning Rate Scheduling");
    let schedulers = vec![
        ("Constant", LearningRateScheduler::constant(0.01)),
        ("Step Decay", LearningRateScheduler::step_decay(0.1, 0.5, 10)),
        ("Cosine Annealing", LearningRateScheduler::cosine_annealing(0.1, 0.001, 20)),
    ];
    
    for (name, scheduler) in schedulers {
        print!("  {} schedule: ", name);
        for step in [0, 5, 10, 15, 20] {
            print!("step {} = {:.4}, ", step, scheduler.get_lr(step));
        }
        println!();
    }
    println!();
    
    // 3. Gradient Clipping
    println!("3. Gradient Clipping");
    let mut gradients = array![[1.5, -2.0], [0.5, 3.0]];
    println!("  Original gradients: {:?}", gradients);
    
    let clipper = GradientClipper::ClipByValue { min: -1.0, max: 1.0 };
    clipper.clip_weights(&mut gradients);
    println!("  After value clipping [-1, 1]: {:?}", gradients);
    
    let mut gradients2 = array![[2.0, 3.0], [4.0, 5.0]];
    let clipper2 = GradientClipper::ClipByNorm { max_norm: 5.0 };
    clipper2.clip_weights(&mut gradients2);
    println!("  After norm clipping (max=5): {:?}\n", gradients2);
    
    // 4. Advanced Network Architecture
    println!("4. Advanced Network Architecture");
    
    // Create layers with different initialization strategies
    let _dense1 = DenseLayer::new_with_init(10, 64, Activation::Relu, WeightInit::HeNormal);
    let _dense2 = DenseLayer::new_with_init(64, 32, Activation::Relu, WeightInit::XavierUniform);
    
    println!("  Created dense layers with He and Xavier initialization");
    println!("  Layer 1: {} -> {} with ReLU", 10, 64);
    println!("  Layer 2: {} -> {} with ReLU", 64, 32);
    
    // Note: In a real implementation, you'd integrate these into a full network
    // with batch norm and dropout layers
    
    println!("\n=== Demo Complete ===");
}