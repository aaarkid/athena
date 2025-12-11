use crate::metrics::tracker::TrainingMetrics;
use crate::network::NeuralNetwork;
use std::io::Write;

/// Export metrics to CSV format
pub fn export_metrics_csv(metrics: &TrainingMetrics, path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    
    // Write header
    writeln!(file, "step,loss,episode_reward,episode_length,q_value,learning_rate,epsilon")?;
    
    // Find the maximum length among all metrics
    let max_len = [
        metrics.losses.len(),
        metrics.episode_rewards.len(),
        metrics.episode_lengths.len(),
        metrics.q_values.len(),
        metrics.learning_rates.len(),
        metrics.epsilons.len(),
    ].iter().max().copied().unwrap_or(0);
    
    // Write data rows
    for i in 0..max_len {
        let loss = metrics.losses.get(i).copied().unwrap_or(f32::NAN);
        let reward = metrics.episode_rewards.get(i).copied().unwrap_or(f32::NAN);
        let length = metrics.episode_lengths.get(i).copied().unwrap_or(0) as f32;
        let q_value = metrics.q_values.get(i).copied().unwrap_or(f32::NAN);
        let lr = metrics.learning_rates.get(i).copied().unwrap_or(f32::NAN);
        let epsilon = metrics.epsilons.get(i).copied().unwrap_or(f32::NAN);
        
        writeln!(file, "{},{},{},{},{},{},{}", 
                 i, loss, reward, length, q_value, lr, epsilon)?;
    }
    
    Ok(())
}

/// Export network structure to a text format
pub fn export_network_structure(network: &NeuralNetwork, path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    
    writeln!(file, "Neural Network Architecture")?;
    writeln!(file, "==========================")?;
    writeln!(file, "Number of layers: {}", network.layers.len())?;
    writeln!(file)?;
    
    for (i, layer) in network.layers.iter().enumerate() {
        writeln!(file, "Layer {}: {} -> {}", 
                 i + 1, 
                 layer.weights.shape()[0], 
                 layer.weights.shape()[1])?;
        writeln!(file, "  Activation: {:?}", layer.activation)?;
        writeln!(file, "  Weight shape: {:?}", layer.weights.shape())?;
        writeln!(file, "  Bias shape: {:?}", layer.biases.shape())?;
        
        // Weight statistics
        let weight_mean = layer.weights.mean().unwrap_or(0.0);
        let weight_std = layer.weights.std(0.0);
        let weight_min = layer.weights.iter().copied().fold(f32::INFINITY, f32::min);
        let weight_max = layer.weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        writeln!(file, "  Weight stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", 
                 weight_mean, weight_std, weight_min, weight_max)?;
        
        // Bias statistics
        let bias_mean = layer.biases.mean().unwrap_or(0.0);
        let bias_std = layer.biases.std(0.0);
        let bias_min = layer.biases.iter().copied().fold(f32::INFINITY, f32::min);
        let bias_max = layer.biases.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        writeln!(file, "  Bias stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", 
                 bias_mean, bias_std, bias_min, bias_max)?;
        writeln!(file)?;
    }
    
    writeln!(file, "Optimizer: {:?}", network.optimizer)?;
    
    Ok(())
}

/// Export weight matrices to numpy-compatible format
pub fn export_weights_npz(network: &NeuralNetwork, prefix: &str) -> std::io::Result<()> {
    for (i, layer) in network.layers.iter().enumerate() {
        // Export weights
        let weight_path = format!("{}_layer{}_weights.csv", prefix, i);
        let mut weight_file = std::fs::File::create(weight_path)?;
        
        for row in layer.weights.outer_iter() {
            let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
            writeln!(weight_file, "{}", row_str.join(","))?;
        }
        
        // Export biases
        let bias_path = format!("{}_layer{}_biases.csv", prefix, i);
        let mut bias_file = std::fs::File::create(bias_path)?;
        
        let bias_str: Vec<String> = layer.biases.iter().map(|x| x.to_string()).collect();
        writeln!(bias_file, "{}", bias_str.join(","))?;
    }
    
    Ok(())
}

/// Export training history in JSON format
pub fn export_metrics_json(metrics: &TrainingMetrics, path: &str) -> crate::error::Result<()> {
    let json = serde_json::to_string_pretty(metrics)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Create a markdown report of training results
pub fn export_training_report(
    metrics: &TrainingMetrics,
    network: &NeuralNetwork,
    path: &str,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    
    writeln!(file, "# Training Report")?;
    writeln!(file)?;
    writeln!(file, "## Network Architecture")?;
    writeln!(file)?;
    
    // Network details
    for (i, layer) in network.layers.iter().enumerate() {
        writeln!(file, "- Layer {}: {} → {} ({})", 
                 i + 1,
                 layer.weights.shape()[0],
                 layer.weights.shape()[1],
                 format!("{:?}", layer.activation).split("::").last().unwrap_or("Unknown"))?;
    }
    
    writeln!(file)?;
    writeln!(file, "## Training Results")?;
    writeln!(file)?;
    
    // Loss statistics
    if !metrics.losses.is_empty() {
        let final_loss = metrics.losses.back().copied().unwrap_or(0.0);
        let avg_loss = metrics.losses.iter().sum::<f32>() / metrics.losses.len() as f32;
        let min_loss = metrics.losses.iter().copied().fold(f32::INFINITY, f32::min);
        
        writeln!(file, "### Loss")?;
        writeln!(file, "- Final: {:.6}", final_loss)?;
        writeln!(file, "- Average: {:.6}", avg_loss)?;
        writeln!(file, "- Minimum: {:.6}", min_loss)?;
        writeln!(file)?;
    }
    
    // Reward statistics
    if !metrics.episode_rewards.is_empty() {
        let final_reward = metrics.episode_rewards.back().copied().unwrap_or(0.0);
        let avg_reward = metrics.episode_rewards.iter().sum::<f32>() / metrics.episode_rewards.len() as f32;
        let max_reward = metrics.episode_rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        writeln!(file, "### Episode Rewards")?;
        writeln!(file, "- Final: {:.2}", final_reward)?;
        writeln!(file, "- Average: {:.2}", avg_reward)?;
        writeln!(file, "- Maximum: {:.2}", max_reward)?;
        writeln!(file, "- Episodes: {}", metrics.episode_rewards.len())?;
        writeln!(file)?;
    }
    
    // Custom metrics
    if !metrics.custom_metrics.is_empty() {
        writeln!(file, "### Custom Metrics")?;
        for (name, values) in &metrics.custom_metrics {
            if !values.is_empty() {
                let final_val = values.back().copied().unwrap_or(0.0);
                let avg_val = values.iter().sum::<f32>() / values.len() as f32;
                writeln!(file, "- {}: Final={:.4}, Average={:.4}", name, final_val, avg_val)?;
            }
        }
    }
    
    Ok(())
}