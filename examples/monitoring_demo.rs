/***
# Monitoring and Debugging Demo

* Demonstrates monitoring and debugging features:
  - Metrics tracking during training
  - Network health inspection
  - Gradient flow analysis
  - Numerical stability checks
  - Visualization of training progress
***/

use athena::{
    agent::DqnAgentBuilder,
    optimizer::{OptimizerWrapper, Adam},
    layers::Layer,
    activations::Activation,
    replay_buffer::ReplayBuffer,
    metrics::MetricsTracker,
    debug::{NetworkInspector, check_weights},
    visualization::{plot_loss_history, plot_reward_history, training_progress, export_metrics_csv},
};
use ndarray::array;

fn main() {
    println!("=== Monitoring and Debugging Demo ===\n");
    
    // Create agent with monitoring
    let layers = vec![
        Layer::new(4, 128, Activation::Relu),
        Layer::new(128, 64, Activation::Relu),
        Layer::new(64, 2, Activation::Linear),
    ];
    
    let optimizer = OptimizerWrapper::Adam(Adam::new(&layers, 0.9, 0.999, 1e-8));
    
    let mut agent = DqnAgentBuilder::new()
        .layer_sizes(&[4, 128, 64, 2])
        .epsilon(1.0)
        .optimizer(optimizer)
        .build()
        .unwrap();
    
    // Initialize monitoring tools
    let mut metrics_tracker = MetricsTracker::new(3, 1000); // 3 layers, 1000 history
    let mut network_inspector = NetworkInspector::new();
    let mut replay_buffer = ReplayBuffer::new(1000);
    
    // Training parameters
    let episodes = 100;
    let steps_per_episode = 50;
    let mut total_steps = 0;
    
    println!("Starting training with monitoring...\n");
    
    for episode in 0..episodes {
        metrics_tracker.start_episode();
        let mut state = array![0.0, 0.0, 0.0, 0.0];
        
        for step in 0..steps_per_episode {
            // Take action
            let action = agent.act(state.view()).unwrap_or(0);
            
            // Simulate environment (simple dummy environment)
            let reward = if action == (step % 2) { 1.0 } else { -0.1 };
            let next_state = array![
                (step as f32).sin() * 0.1,
                (step as f32).cos() * 0.1,
                state[2] + 0.01,
                state[3] - 0.01,
            ];
            let done = step == steps_per_episode - 1;
            
            // Record step metrics
            metrics_tracker.step(reward);
            
            // Store experience
            let experience = athena::replay_buffer::Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            };
            replay_buffer.add(experience);
            
            // Train if enough experiences
            if replay_buffer.len() >= 32 {
                let batch = replay_buffer.sample(32);
                let loss = agent.train_on_batch(&batch, 0.99, 0.001).unwrap_or(0.0);
                
                // Record training metrics
                metrics_tracker.record_loss(loss);
                metrics_tracker.record_epsilon(agent.epsilon);
                
                // Record Q-values
                let q_values = agent.q_network.forward(state.view());
                let max_q = q_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                metrics_tracker.record_q_value(max_q);
                
                // Inspect network health periodically
                if total_steps % 100 == 0 {
                    let weight_stats = network_inspector.inspect_weights(&agent.q_network);
                    
                    // Check for numerical issues
                    let weights: Vec<_> = agent.q_network.layers.iter()
                        .map(|l| l.weights.clone())
                        .collect();
                    let weight_issues = check_weights(&weights);
                    
                    if !weight_issues.is_empty() {
                        println!("WARNING: Numerical issues detected: {:?}", weight_issues);
                    }
                    
                    // Record weight norms
                    for (i, stats) in weight_stats.iter().enumerate() {
                        let norm = (stats.mean * stats.mean + stats.std * stats.std).sqrt();
                        metrics_tracker.record_weight_norm(i, norm);
                    }
                }
            }
            
            state = next_state;
            total_steps += 1;
        }
        
        metrics_tracker.end_episode();
        
        // Update epsilon
        agent.update_epsilon(agent.epsilon * 0.95);
        
        // Print progress every 10 episodes
        if (episode + 1) % 10 == 0 {
            let avg_reward = metrics_tracker.avg_episode_reward(10).unwrap_or(0.0);
            let avg_loss = metrics_tracker.avg_loss(100).unwrap_or(0.0);
            
            println!("{}", training_progress(
                episode + 1,
                episodes,
                avg_reward,
                avg_loss,
                agent.epsilon
            ));
            
            // Show network health report
            let health_report = network_inspector.generate_report();
            println!("Network Health: {} inspections, {} numerical issues, {} dead neurons",
                     health_report.num_inspections,
                     health_report.numerical_issues_count,
                     health_report.num_dead_neurons);
        }
    }
    
    println!("\n=== Training Complete ===\n");
    
    // Display final metrics
    let metrics = metrics_tracker.metrics();
    
    // Plot loss history
    println!("Loss History:");
    println!("{}", plot_loss_history(metrics, 60, 10));
    
    // Plot reward history
    println!("\nReward History:");
    println!("{}", plot_reward_history(metrics, 60, 10));
    
    // Export metrics
    if let Err(e) = export_metrics_csv(metrics, "training_metrics.csv") {
        eprintln!("Failed to export metrics: {}", e);
    } else {
        println!("\nMetrics exported to training_metrics.csv");
    }
    
    // Save metrics in JSON format
    if let Err(e) = metrics_tracker.save("training_metrics.json") {
        eprintln!("Failed to save metrics: {}", e);
    } else {
        println!("Metrics saved to training_metrics.json");
    }
    
    // Final statistics
    println!("\nFinal Statistics:");
    println!("Total episodes: {}", metrics_tracker.episode_count());
    println!("Total steps: {}", metrics_tracker.total_steps());
    
    if let Some(avg_reward) = metrics_tracker.avg_episode_reward(episodes) {
        println!("Average episode reward: {:.2}", avg_reward);
    }
    
    if let Some(avg_loss) = metrics_tracker.avg_loss(1000) {
        println!("Average loss: {:.4}", avg_loss);
    }
    
    println!("\n=== Demo Complete ===");
}