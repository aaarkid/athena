use crate::metrics::tracker::TrainingMetrics;

/// Plot loss history as ASCII art
pub fn plot_loss_history(metrics: &TrainingMetrics, width: usize, height: usize) -> String {
    if metrics.losses.is_empty() {
        return "No loss data available".to_string();
    }
    
    let losses: Vec<f32> = metrics.losses.iter().copied().collect();
    plot_series(&losses, "Loss History", width, height)
}

/// Plot reward history as ASCII art
pub fn plot_reward_history(metrics: &TrainingMetrics, width: usize, height: usize) -> String {
    if metrics.episode_rewards.is_empty() {
        return "No reward data available".to_string();
    }
    
    let rewards: Vec<f32> = metrics.episode_rewards.iter().copied().collect();
    plot_series(&rewards, "Episode Rewards", width, height)
}

/// Plot arbitrary metrics as ASCII art
pub fn plot_metrics(values: &[f32], title: &str, width: usize, height: usize) -> String {
    plot_series(values, title, width, height)
}

/// Generic series plotter
fn plot_series(values: &[f32], title: &str, width: usize, height: usize) -> String {
    if values.is_empty() || width < 10 || height < 5 {
        return format!("{}: Invalid data or dimensions", title);
    }
    
    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    if (max_val - min_val).abs() < f32::EPSILON {
        return format!("{}: All values are {:.4}", title, min_val);
    }
    
    let mut plot = vec![vec![' '; width]; height];
    
    // Add axes
    for i in 0..height {
        plot[i][0] = '|';
    }
    for j in 0..width {
        plot[height - 1][j] = '-';
    }
    plot[height - 1][0] = '+';
    
    // Plot data points
    let x_scale = (values.len() - 1) as f32 / (width - 3) as f32;
    let y_scale = (height - 3) as f32 / (max_val - min_val);
    
    for (i, &value) in values.iter().enumerate() {
        let x = ((i as f32 / x_scale) as usize + 2).min(width - 1);
        let y = (height - 3 - ((value - min_val) * y_scale) as usize).max(0).min(height - 2);
        plot[y][x] = '*';
    }
    
    // Build output string
    let mut output = format!("{}\n", title);
    output.push_str(&format!("Max: {:.4}\n", max_val));
    
    for row in plot.iter() {
        output.push_str(&row.iter().collect::<String>());
        output.push('\n');
    }
    
    output.push_str(&format!("Min: {:.4}\n", min_val));
    output.push_str(&format!("Points: {}\n", values.len()));
    
    output
}

/// Create a simple histogram
pub fn histogram(values: &[f32], bins: usize) -> String {
    if values.is_empty() || bins == 0 {
        return "No data for histogram".to_string();
    }
    
    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    if (max_val - min_val).abs() < f32::EPSILON {
        return format!("All values are {:.4}", min_val);
    }
    
    let bin_width = (max_val - min_val) / bins as f32;
    let mut counts = vec![0; bins];
    
    for &value in values {
        let bin = ((value - min_val) / bin_width).floor() as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += 1;
    }
    
    let max_count = *counts.iter().max().unwrap_or(&1);
    let scale = 40.0 / max_count as f32;
    
    let mut output = String::new();
    output.push_str("Histogram:\n");
    
    for (i, &count) in counts.iter().enumerate() {
        let bin_start = min_val + i as f32 * bin_width;
        let bin_end = bin_start + bin_width;
        let bar_length = (count as f32 * scale) as usize;
        let bar = "#".repeat(bar_length);
        output.push_str(&format!("[{:>7.2}, {:>7.2}): {:>4} {}\n", 
                                bin_start, bin_end, count, bar));
    }
    
    output
}

/// Display training progress
pub fn training_progress(
    episode: usize,
    total_episodes: usize,
    avg_reward: f32,
    avg_loss: f32,
    epsilon: f32,
) -> String {
    let progress = episode as f32 / total_episodes as f32;
    let bar_length = 30;
    let filled = (progress * bar_length as f32) as usize;
    let bar = format!("[{}{}]", "=".repeat(filled), " ".repeat(bar_length - filled));
    
    format!(
        "Episode {}/{} {} {:.1}% | Avg Reward: {:.2} | Avg Loss: {:.4} | ε: {:.3}",
        episode, total_episodes, bar, progress * 100.0, avg_reward, avg_loss, epsilon
    )
}

/// Create a summary table of metrics
pub fn metrics_summary(metrics: &TrainingMetrics) -> String {
    let mut output = String::new();
    output.push_str("Training Metrics Summary\n");
    output.push_str("========================\n");
    
    if !metrics.losses.is_empty() {
        let recent_loss = metrics.losses.back().copied().unwrap_or(0.0);
        let avg_loss = metrics.losses.iter().sum::<f32>() / metrics.losses.len() as f32;
        output.push_str(&format!("Loss: Current={:.4}, Average={:.4}\n", recent_loss, avg_loss));
    }
    
    if !metrics.episode_rewards.is_empty() {
        let recent_reward = metrics.episode_rewards.back().copied().unwrap_or(0.0);
        let avg_reward = metrics.episode_rewards.iter().sum::<f32>() / metrics.episode_rewards.len() as f32;
        let max_reward = metrics.episode_rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        output.push_str(&format!("Rewards: Current={:.2}, Average={:.2}, Max={:.2}\n", 
                                recent_reward, avg_reward, max_reward));
    }
    
    if !metrics.episode_lengths.is_empty() {
        let recent_length = metrics.episode_lengths.back().copied().unwrap_or(0);
        let avg_length = metrics.episode_lengths.iter().sum::<usize>() / metrics.episode_lengths.len();
        output.push_str(&format!("Episode Length: Current={}, Average={}\n", recent_length, avg_length));
    }
    
    if !metrics.q_values.is_empty() {
        let recent_q = metrics.q_values.back().copied().unwrap_or(0.0);
        let avg_q = metrics.q_values.iter().sum::<f32>() / metrics.q_values.len() as f32;
        output.push_str(&format!("Q-Values: Current={:.4}, Average={:.4}\n", recent_q, avg_q));
    }
    
    output
}