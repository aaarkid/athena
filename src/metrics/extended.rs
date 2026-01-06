//! Extended metrics tracking functionality
//! 
//! This module provides additional metrics tracking capabilities
//! beyond the basic MetricsTracker, including visualization and
//! flexible metric storage.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Write, BufWriter};

/// Extended metrics tracker with visualization support
pub struct ExtendedMetricsTracker {
    /// Stores all metrics by name
    metrics: HashMap<String, Vec<f32>>,
    /// Optional file writer for logging
    log_file: Option<BufWriter<File>>,
}

impl ExtendedMetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            log_file: None,
        }
    }
    
    /// Create metrics tracker with file logging
    pub fn with_file_logging(log_path: &str) -> std::io::Result<Self> {
        let file = File::create(log_path)?;
        let writer = BufWriter::new(file);
        
        Ok(Self {
            metrics: HashMap::new(),
            log_file: Some(writer),
        })
    }
    
    /// Add a metric value
    pub fn add_metric(&mut self, name: &str, value: f32) {
        self.metrics.entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
        
        // Log to file if enabled
        if let Some(ref mut writer) = self.log_file {
            let _ = writeln!(writer, "{},{},{}", 
                self.metrics[name].len() - 1, name, value);
            let _ = writer.flush();
        }
    }
    
    /// Get the latest value for a metric
    pub fn get_latest(&self, name: &str) -> Option<f32> {
        self.metrics.get(name)?.last().copied()
    }
    
    /// Get all values for a metric
    pub fn get_history(&self, name: &str) -> Option<&Vec<f32>> {
        self.metrics.get(name)
    }
    
    /// Get average over last N values
    pub fn get_moving_average(&self, name: &str, window: usize) -> Option<f32> {
        let values = self.metrics.get(name)?;
        if values.len() < window {
            return None;
        }
        
        let sum: f32 = values.iter()
            .rev()
            .take(window)
            .sum();
        
        Some(sum / window as f32)
    }
    
    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }
    
    /// Get summary statistics for a metric
    pub fn get_stats(&self, name: &str) -> Option<MetricStats> {
        let values = self.metrics.get(name)?;
        if values.is_empty() {
            return None;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();
        
        Some(MetricStats {
            mean,
            std_dev,
            min,
            max,
            count: values.len(),
        })
    }
    
    /// Plot metrics to console (ASCII)
    pub fn plot_ascii(&self, name: &str, width: usize, height: usize) {
        if let Some(values) = self.metrics.get(name) {
            if values.is_empty() {
                return;
            }
            
            let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_val - min_val;
            
            if range == 0.0 {
                println!("{}: constant at {}", name, min_val);
                return;
            }
            
            println!("\n{} (min: {:.4}, max: {:.4})", name, min_val, max_val);
            println!("{}", "─".repeat(width + 10));
            
            // Create plot grid
            let mut grid = vec![vec![' '; width]; height];
            
            // Plot values
            for (i, &value) in values.iter().enumerate() {
                let x = (i * width) / values.len();
                let y = height - 1 - ((value - min_val) / range * (height - 1) as f32) as usize;
                
                if x < width && y < height {
                    grid[y][x] = '●';
                }
            }
            
            // Print grid
            for (i, row) in grid.iter().enumerate() {
                let y_val = max_val - (i as f32 / (height - 1) as f32) * range;
                print!("{:8.3} │", y_val);
                println!("{}", row.iter().collect::<String>());
            }
            
            println!("         └{}", "─".repeat(width));
            println!("          {:<width$}", "epochs →", width = width);
        }
    }
}

/// Statistics for a metric
#[derive(Debug, Clone)]
pub struct MetricStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub count: usize,
}

impl Default for ExtendedMetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}