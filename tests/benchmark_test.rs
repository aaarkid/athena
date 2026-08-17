use athena::{
    network::NeuralNetwork,
    activations::Activation,
    optimizer::{OptimizerWrapper, SGD},
    agent::DqnAgentBuilder,
    replay_buffer::{PrioritizedReplayBuffer, PriorityMethod},
};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn benchmark_operation<F>(name: &str, iterations: usize, mut operation: F) 
where F: FnMut() 
{
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    let duration = start.elapsed();
    let avg_time = duration.as_micros() as f64 / iterations as f64;
    println!("{}: {:.2} μs per iteration ({} iterations)", name, avg_time, iterations);
}

#[test]
#[ignore] // Run with: cargo test --ignored benchmark
fn benchmark_forward_pass() {
    let layer_sizes = &[784, 128, 64, 10]; // MNIST-like architecture
    let activations = &[Activation::Relu, Activation::Relu, Activation::Linear];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);
    
    let input = Array1::ones(784);
    
    benchmark_operation("Forward pass (784->128->64->10)", 1000, || {
        let _ = network.forward(input.view());
    });
}

#[test]
#[ignore]
fn benchmark_backward_pass() {
    let layer_sizes = &[100, 50, 25, 10];
    let activations = &[Activation::Relu, Activation::Relu, Activation::Sigmoid];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);
    
    let inputs = Array2::ones((32, 100)); // Batch of 32
    let targets = Array2::zeros((32, 10));
    
    benchmark_operation("Backward pass (batch=32)", 100, || {
        network.train_minibatch(inputs.view(), targets.view(), 0.01);
    });
}

#[test]
#[ignore]
fn benchmark_optimizers() {
    let layer_sizes = &[50, 100, 50];
    let activations = &[Activation::Relu, Activation::Relu];
    
    // Benchmark SGD
    let mut sgd_network = NeuralNetwork::new(layer_sizes, activations, OptimizerWrapper::SGD(SGD::new()));
    let inputs = Array2::ones((16, 50));
    let targets = Array2::zeros((16, 50));
    
    benchmark_operation("SGD optimizer (batch=16)", 100, || {
        sgd_network.train_minibatch(inputs.view(), targets.view(), 0.01);
    });
    
    // Benchmark Adam
    let adam_opt = OptimizerWrapper::SGD(SGD::new()); // Using SGD for simplicity
    let mut adam_network = NeuralNetwork::new(layer_sizes, activations, adam_opt);
    
    benchmark_operation("SGD optimizer test 2 (batch=16)", 100, || {
        adam_network.train_minibatch(inputs.view(), targets.view(), 0.01);
    });
}

#[test]
#[ignore]
fn benchmark_activation_functions() {
    let size = 1000;
    let data = Array1::linspace(-5.0, 5.0, size);
    
    let activations = vec![
        ("ReLU", Activation::Relu),
        ("Sigmoid", Activation::Sigmoid),
        ("Tanh", Activation::Tanh),
        ("LeakyReLU", Activation::LeakyRelu { alpha: 0.01 }),
        ("ELU", Activation::Elu { alpha: 1.0 }),
        ("GELU", Activation::Gelu),
    ];
    
    for (name, activation) in activations {
        let mut test_data = data.clone();
        benchmark_operation(&format!("{} activation (size={})", name, size), 1000, || {
            activation.apply(&mut test_data);
        });
    }
}

#[test]
#[ignore]
fn benchmark_replay_buffer() {
    use athena::replay_buffer::Experience;
    
    let mut buffer = PrioritizedReplayBuffer::new(10000, PriorityMethod::Proportional { alpha: 0.6 });
    
    // Fill buffer
    for i in 0..10000 {
        let exp = Experience {
            state: ndarray::array![i as f32, (i as f32).sin()],
            action: i % 4,
            reward: (i % 10) as f32 - 5.0,
            next_state: ndarray::array![(i + 1) as f32, ((i + 1) as f32).sin()],
            done: i % 100 == 99,
        };
        buffer.add_with_priority(exp, (i % 10) as f32 + 1.0);
    }
    
    benchmark_operation("Prioritized sampling (batch=32)", 1000, || {
        let _ = buffer.sample_with_weights(32, 0.4);
    });
    
    // Benchmark priority updates
    let (_, _, indices) = buffer.sample_with_weights(32, 0.4);
    let new_priorities: Vec<f32> = (0..32).map(|i| i as f32).collect();
    
    benchmark_operation("Priority updates (batch=32)", 1000, || {
        buffer.update_priorities(&indices, &new_priorities);
    });
}

#[test]
#[ignore]
fn benchmark_agent_action_selection() {
    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[84, 256, 128, 6]) // Atari-like architecture
        .epsilon(0.1)
        .optimizer(OptimizerWrapper::SGD(SGD::new()))
        .build()
        .unwrap();
    
    let state = Array1::ones(84);
    
    benchmark_operation("Agent action selection", 10000, || {
        let _ = agent.act(state.view()).unwrap();
    });
    
    // For greedy action, set epsilon to 0
    agent.epsilon = 0.0;
    benchmark_operation("Agent greedy action", 10000, || {
        let _ = agent.act(state.view()).unwrap();
    });
}

#[test]
#[ignore]
fn benchmark_full_training_step() {
    use athena::replay_buffer::Experience;
    
    // Setup
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    
    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[4, 64, 32, 2])
        .epsilon(0.1)
        .optimizer(optimizer)
        .target_update_freq(100)
        .use_double_dqn(true)
        .build()
        .unwrap();
    
    // Create batch of experiences
    let batch: Vec<Experience> = (0..32).map(|i| {
        Experience {
            state: ndarray::array![i as f32 * 0.1, (i as f32 * 0.1).sin(), (i as f32 * 0.1).cos(), i as f32 * 0.01],
            action: i % 2,
            reward: if i % 2 == 0 { 1.0 } else { -1.0 },
            next_state: ndarray::array![(i + 1) as f32 * 0.1, ((i + 1) as f32 * 0.1).sin(), ((i + 1) as f32 * 0.1).cos(), (i + 1) as f32 * 0.01],
            done: i == 31,
        }
    }).collect();
    
    let batch_refs: Vec<&Experience> = batch.iter().collect();
    
    benchmark_operation("Full training step (batch=32, Double DQN)", 100, || {
        let _ = agent.train_on_batch(&batch_refs, 0.99, 0.001);
    });
}

#[test]
#[ignore]
fn benchmark_network_sizes() {
    println!("\nNetwork size scaling benchmark:");
    
    let sizes = vec![
        (vec![10, 20, 10], "Small"),
        (vec![100, 200, 100], "Medium"),
        (vec![784, 512, 256, 10], "Large"),
        (vec![1000, 1000, 1000, 1000, 10], "Very Large"),
    ];
    
    for (layer_sizes, name) in sizes {
        let activations: Vec<Activation> = (0..layer_sizes.len()-1).map(|_| Activation::Relu).collect();
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
        
        let input = Array1::ones(layer_sizes[0]);
        let iterations = if name == "Very Large" { 10 } else { 100 };
        
        benchmark_operation(
            &format!("{} network forward pass {:?}", name, layer_sizes), 
            iterations, 
            || { let _ = network.forward(input.view()); }
        );
    }
}
// The two tests below are the only ones here that assert. They run in the test profile,
// which is unoptimized, so their ceilings are far above what a release build costs; they
// exist to catch an order-of-magnitude regression, not to measure anything. Release
// figures live in CHANGELOG.md.

/// Time a closure, discarding a warmup and returning the mean in microseconds.
fn mean_micros<F: FnMut()>(warmup: usize, iterations: usize, mut operation: F) -> f64 {
    for _ in 0..warmup {
        operation();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0
}

#[test]
fn inference_stays_under_the_frame_budget() {
    use athena::network::InferenceBuffers;

    let mut network = NeuralNetwork::new(
        &[8, 128, 128, 4],
        &[Activation::Relu, Activation::Relu, Activation::Linear],
        OptimizerWrapper::SGD(SGD::new()),
    );
    let state = Array1::from_shape_fn(8, |i| i as f32 * 0.1);
    let mut buffers = InferenceBuffers::new();

    let micros = mean_micros(100, 2_000, || {
        std::hint::black_box(network.predict_into(state.view(), &mut buffers));
    });

    println!("predict on [8,128,128,4]: {:.2} us per call (test profile)", micros);

    // Roughly 440 us in the test profile on the machine this was written on, against
    // about 4 us in release. The ceiling is set several times over the observed figure
    // so it survives a slow or loaded machine while still catching a 10x regression.
    assert!(
        micros < 2_500.0,
        "one inference took {:.2} us in the test profile, an order of magnitude over budget",
        micros
    );
}

#[test]
fn train_on_batch_stays_under_budget() {
    use athena::optimizer::Adam;
    use athena::replay_buffer::Experience;

    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[8, 128, 128, 4])
        .epsilon(0.0)
        .optimizer(OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8)))
        .target_update_freq(1_000_000)
        .use_double_dqn(true)
        .build()
        .expect("builder rejected a complete configuration");

    let experiences: Vec<Experience> = (0..32)
        .map(|i| Experience {
            state: Array1::from_shape_fn(8, |j| ((i + j) as f32 * 0.05).sin()),
            action: i % 4,
            reward: 0.5,
            next_state: Array1::from_shape_fn(8, |j| ((i * 2 + j) as f32 * 0.05).cos()),
            done: i % 8 == 0,
        })
        .collect();

    let micros = mean_micros(10, 60, || {
        let batch: Vec<&Experience> = experiences.iter().collect();
        agent.train_on_batch(&batch, 0.99, 1e-4).unwrap();
    });

    println!("train_on_batch [8,128,128,4] batch 32: {:.2} us (test profile)", micros);

    // Roughly 11,000 us in the test profile, against a few hundred in release. Same
    // reasoning as above: the ceiling catches a structural regression, not a slow day.
    assert!(
        micros < 45_000.0,
        "one training step took {:.2} us in the test profile, an order of magnitude over budget",
        micros
    );
}
