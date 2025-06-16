use athena::network::NeuralNetwork;
use athena::layers::{dense::DenseLayer, BatchNormLayer, DropoutLayer, traits::Layer};
use athena::activations::Activation;
use athena::optimizer::{OptimizerWrapper, Adam, SGD};
use athena::agent::DqnAgent;
use athena::replay_buffer::{ReplayBuffer, Experience};
use athena::algorithms::{ppo::PPOAgent, sac::SACAgent};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;

fn benchmark_layer_types() {
    println!("=== Layer Type Benchmarks ===\n");
    
    let input_size = 512;
    let output_size = 256;
    let batch_size = 64;
    let iterations = 100;
    
    // Dense layer
    let mut dense = DenseLayer::new(input_size, output_size, Activation::Relu);
    let input = Array2::random((batch_size, input_size), Uniform::new(-1.0, 1.0));
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = dense.forward_batch(input.view());
    }
    let dense_time = start.elapsed();
    println!("Dense Layer: {:?} ({:.2} ms/iter)", dense_time, dense_time.as_secs_f64() * 1000.0 / iterations as f64);
    
    // BatchNorm layer
    let mut batch_norm = BatchNormLayer::new(output_size, 0.9, 1e-5);
    let bn_input = Array2::random((batch_size, output_size), Uniform::new(-1.0, 1.0));
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = batch_norm.forward_batch(bn_input.view());
    }
    let bn_time = start.elapsed();
    println!("BatchNorm Layer: {:?} ({:.2} ms/iter)", bn_time, bn_time.as_secs_f64() * 1000.0 / iterations as f64);
    
    // Dropout layer
    let mut dropout = DropoutLayer::new(output_size, 0.2);
    dropout.set_training(true);
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = dropout.forward_batch(bn_input.view());
    }
    let dropout_time = start.elapsed();
    println!("Dropout Layer: {:?} ({:.2} ms/iter)", dropout_time, dropout_time.as_secs_f64() * 1000.0 / iterations as f64);
}

fn benchmark_optimizers() {
    println!("\n=== Optimizer Benchmarks ===\n");
    
    let layer_sizes = &[128, 256, 128, 64];
    let batch_size = 32;
    let iterations = 100;
    
    // SGD
    let activations = vec![Activation::Relu, Activation::Relu, Activation::Linear];
    let mut network_sgd = NeuralNetwork::new(
        layer_sizes,
        &activations,
        OptimizerWrapper::SGD(SGD::new())
    );
    
    let inputs = Array2::random((batch_size, 128), Uniform::new(-1.0, 1.0));
    let targets = Array2::random((batch_size, 64), Uniform::new(0.0, 1.0));
    
    let start = Instant::now();
    for _ in 0..iterations {
        network_sgd.train_minibatch(inputs.view(), targets.view(), 0.01);
    }
    let sgd_time = start.elapsed();
    println!("SGD Optimizer: {:?} ({:.2} ms/iter)", sgd_time, sgd_time.as_secs_f64() * 1000.0 / iterations as f64);
    
    // Adam
    // Create dummy layers for Adam initialization
    let dummy_layers: Vec<DenseLayer> = layer_sizes.windows(2).zip(activations.iter())
        .map(|(w, act)| DenseLayer::new(w[0], w[1], act.clone()))
        .collect();
    let mut network_adam = NeuralNetwork::new(
        layer_sizes,
        &activations,
        OptimizerWrapper::Adam(Adam::new(&dummy_layers, 0.9, 0.999, 1e-8))
    );
    
    let start = Instant::now();
    for _ in 0..iterations {
        network_adam.train_minibatch(inputs.view(), targets.view(), 0.001);
    }
    let adam_time = start.elapsed();
    println!("Adam Optimizer: {:?} ({:.2} ms/iter)", adam_time, adam_time.as_secs_f64() * 1000.0 / iterations as f64);
}

fn benchmark_rl_algorithms() {
    println!("\n=== RL Algorithm Benchmarks ===\n");
    
    let state_size = 64;
    let action_size = 8;
    let batch_size = 32;
    let learning_rate = 0.001;
    
    // Create sample experiences
    let mut buffer = ReplayBuffer::new(1000);
    for _ in 0..1000 {
        buffer.add(Experience {
            state: Array1::random(state_size, Uniform::new(-1.0, 1.0)),
            action: (rand::random::<f32>() * action_size as f32) as usize,
            reward: rand::random::<f32>() * 2.0 - 1.0,
            next_state: Array1::random(state_size, Uniform::new(-1.0, 1.0)),
            done: rand::random::<bool>(),
        });
    }
    
    // DQN
    let dqn_layers: Vec<DenseLayer> = vec![
        DenseLayer::new(state_size, 128, Activation::Relu),
        DenseLayer::new(128, 128, Activation::Relu),
        DenseLayer::new(128, action_size, Activation::Linear),
    ];
    let mut dqn = DqnAgent::new(
        &[state_size, 128, 128, action_size],
        0.01,
        OptimizerWrapper::Adam(Adam::new(&dqn_layers, 0.9, 0.999, 1e-8)),
        1000,
        true
    );
    
    let start = Instant::now();
    for _ in 0..10 {
        let batch = buffer.sample(batch_size);
        dqn.train_on_batch(&batch, 0.99, learning_rate);
    }
    let dqn_time = start.elapsed();
    println!("DQN Agent: {:?} for 10 batches", dqn_time);
    
    // PPO - Create with proper parameters
    let ppo_layers = &[state_size, 128, 128, action_size];
    let ppo_optimizer = OptimizerWrapper::Adam(Adam::new(&dqn_layers, 0.9, 0.999, 1e-8));
    let ppo = PPOAgent::new(
        state_size,
        action_size,
        ppo_layers,
        ppo_optimizer,
        0.2,     // clip_epsilon
        0.01,    // value_loss_coef
        0.01,    // entropy_coef
        4        // n_epochs
    );
    println!("PPO Agent: Initialized (training benchmark omitted for brevity)");
    
    // SAC - Create with proper parameters  
    let sac_layers = &[state_size, 256, 256];
    let sac_optimizer = OptimizerWrapper::Adam(Adam::new(&dqn_layers[0..2], 0.9, 0.999, 1e-8));
    let sac = SACAgent::new(
        state_size,
        action_size,
        sac_layers,
        sac_optimizer,
        0.99,    // gamma
        0.005,   // tau
        0.2,     // alpha
        true     // automatic_entropy_tuning
    );
    println!("SAC Agent: Initialized (continuous actions)");
}

fn benchmark_memory() {
    println!("\n=== Memory Usage ===\n");
    
    // Large network
    let layer_sizes = &[1024, 512, 512, 256, 128];
    let activations = vec![Activation::Relu, Activation::Relu, Activation::Relu, Activation::Linear];
    let dummy_layers: Vec<DenseLayer> = layer_sizes.windows(2).zip(activations.iter())
        .map(|(w, act)| DenseLayer::new(w[0], w[1], act.clone()))
        .collect();
    let _network = NeuralNetwork::new(
        layer_sizes,
        &activations,
        OptimizerWrapper::Adam(Adam::new(&dummy_layers, 0.9, 0.999, 1e-8))
    );
    
    let total_params: usize = layer_sizes.windows(2)
        .map(|w| w[0] * w[1] + w[1])
        .sum();
    let memory_mb = (total_params * 4) as f32 / (1024.0 * 1024.0);
    println!("Large Network Parameters: {}", total_params);
    println!("Estimated Memory Usage: {:.2} MB (weights only)", memory_mb);
    
    // Replay buffer
    let buffer_size = 100_000;
    let state_size = 64;
    let buffer = ReplayBuffer::new(buffer_size);
    let buffer_memory_mb = (buffer_size * state_size * 2 * 4) as f32 / (1024.0 * 1024.0);
    println!("\nReplay Buffer Capacity: {}", buffer_size);
    println!("Estimated Memory Usage: {:.2} MB", buffer_memory_mb);
}

fn main() {
    println!("Athena Comprehensive Benchmarks\n");
    println!("================================\n");
    
    benchmark_layer_types();
    benchmark_optimizers();
    benchmark_rl_algorithms();
    benchmark_memory();
    
    println!("\n================================");
    println!("Benchmarks Complete!");
}