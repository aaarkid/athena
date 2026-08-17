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
use athena::network::NeuralNetwork;
use athena::optimizer::{OptimizerWrapper, SGD};
use ndarray::{array, Array2};

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
    
    // Two identical architectures, differing only in how the weights start out
    let he = DenseLayer::new_with_init(10, 64, Activation::Relu, WeightInit::HeNormal);
    let xavier = DenseLayer::new_with_init(10, 64, Activation::Relu, WeightInit::XavierUniform);

    println!("  He normal spread:      {:.4}", spread(&he));
    println!("  Xavier uniform spread: {:.4}", spread(&xavier));

    // Train both on the same regression task and compare
    println!("\n  Training both for 200 steps on the same data:");
    let (inputs, targets) = make_regression_data(128);

    for (name, init) in [("He normal", WeightInit::HeNormal), ("Xavier uniform", WeightInit::XavierUniform)] {
        let layers = vec![
            DenseLayer::new_with_init(10, 64, Activation::Relu, init.clone()),
            DenseLayer::new_with_init(64, 32, Activation::Relu, init.clone()),
            DenseLayer::new_with_init(32, 1, Activation::Linear, init),
        ];
        let mut network = NeuralNetwork::new_empty()
            .with_layers(layers)
            .with_optimizer(OptimizerWrapper::SGD(SGD::new()));

        let mut loss = 0.0;
        for _ in 0..200 {
            network.train_minibatch(inputs.view(), targets.view(), 0.001);
            let outputs = network.forward_batch(inputs.view());
            loss = (&outputs - &targets).mapv(|x| x * x).mean().unwrap_or(0.0);
        }

        println!("    {name:<15} final loss {loss:.5}");
    }

    println!("\n=== Demo Complete ===");
}

/// Standard deviation of a layer's weights
fn spread(layer: &DenseLayer) -> f32 {
    let values = &layer.weights;
    let mean = values.mean().unwrap_or(0.0);
    (values.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0)).sqrt()
}

/// A deterministic regression target: the sum of the first three inputs
fn make_regression_data(rows: usize) -> (Array2<f32>, Array2<f32>) {
    let mut inputs = Array2::zeros((rows, 10));
    let mut targets = Array2::zeros((rows, 1));

    for i in 0..rows {
        for j in 0..10 {
            inputs[[i, j]] = ((i * 10 + j) as f32 * 0.017).sin();
        }
        targets[[i, 0]] = inputs[[i, 0]] + inputs[[i, 1]] + inputs[[i, 2]];
    }

    (inputs, targets)
}