//! Tensorboard integration for Athena
//! 
//! This module provides logging capabilities that can be visualized with Tensorboard.
//! It exports training metrics, model graphs, and histograms in a format compatible
//! with Tensorboard's log format.

use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::{Array1, Array2};

/// Tensorboard writer for logging metrics and model data
pub struct TensorboardWriter {
    log_dir: PathBuf,
    run_name: String,
    step: i64,
    start_time: u64,
    scalar_writer: Option<BufWriter<File>>,
    histogram_writer: Option<BufWriter<File>>,
}

impl TensorboardWriter {
    /// Create a new Tensorboard writer
    /// 
    /// # Arguments
    /// * `log_dir` - Directory to save logs
    /// * `run_name` - Name of this training run
    pub fn new(log_dir: &str, run_name: &str) -> std::io::Result<Self> {
        let log_path = Path::new(log_dir).join(run_name);
        create_dir_all(&log_path)?;
        
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Create CSV files for simple logging
        let scalar_file = File::create(log_path.join("scalars.csv"))?;
        let histogram_file = File::create(log_path.join("histograms.csv"))?;
        
        let mut scalar_writer = BufWriter::new(scalar_file);
        let mut histogram_writer = BufWriter::new(histogram_file);
        
        // Write headers
        writeln!(scalar_writer, "step,tag,value,wall_time")?;
        writeln!(histogram_writer, "step,tag,count,min,max,mean,std,wall_time")?;
        
        Ok(Self {
            log_dir: log_path,
            run_name: run_name.to_string(),
            step: 0,
            start_time,
            scalar_writer: Some(scalar_writer),
            histogram_writer: Some(histogram_writer),
        })
    }
    
    /// Log a scalar value
    /// 
    /// # Arguments
    /// * `tag` - Name of the metric
    /// * `value` - Value to log
    pub fn add_scalar(&mut self, tag: &str, value: f32) -> std::io::Result<()> {
        let wall_time = self.get_wall_time();
        if let Some(ref mut writer) = self.scalar_writer {
            writeln!(writer, "{},{},{},{}", self.step, tag, value, wall_time)?;
            writer.flush()?;
        }
        Ok(())
    }
    
    /// Log multiple scalars at once
    /// 
    /// # Arguments
    /// * `main_tag` - Main category name
    /// * `tag_scalar_dict` - Dictionary of tag-value pairs
    pub fn add_scalars(&mut self, main_tag: &str, tag_scalar_dict: &[(String, f32)]) -> std::io::Result<()> {
        for (tag, value) in tag_scalar_dict {
            self.add_scalar(&format!("{}/{}", main_tag, tag), *value)?;
        }
        Ok(())
    }
    
    /// Log a histogram of values
    /// 
    /// # Arguments
    /// * `tag` - Name of the histogram
    /// * `values` - Array of values
    pub fn add_histogram(&mut self, tag: &str, values: &Array1<f32>) -> std::io::Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        
        let count = values.len();
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean = values.mean().unwrap_or(0.0);
        
        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / count as f32;
        let std = variance.sqrt();
        
        let wall_time = self.get_wall_time();
        
        if let Some(ref mut writer) = self.histogram_writer {
            writeln!(writer, "{},{},{},{},{},{},{},{}", 
                     self.step, tag, count, min, max, mean, std, wall_time)?;
            writer.flush()?;
        }
        Ok(())
    }
    
    /// Log model weights and biases
    /// 
    /// # Arguments
    /// * `layer_name` - Name of the layer
    /// * `weights` - Weight matrix
    /// * `biases` - Bias vector
    pub fn add_layer_stats(&mut self, layer_name: &str, weights: &Array2<f32>, biases: &Array1<f32>) -> std::io::Result<()> {
        // Log weight statistics
        let weight_values = weights.iter().cloned().collect::<Vec<_>>();
        let weight_array = Array1::from_vec(weight_values);
        self.add_histogram(&format!("{}/weights", layer_name), &weight_array)?;
        
        // Log bias statistics
        self.add_histogram(&format!("{}/biases", layer_name), biases)?;
        
        // Log weight norms
        let weight_norm = weights.mapv(|x| x * x).sum().sqrt();
        self.add_scalar(&format!("{}/weight_norm", layer_name), weight_norm)?;
        
        Ok(())
    }
    
    /// Create a graph visualization file (simplified)
    pub fn add_graph(&self, layer_info: &[(String, usize, usize)]) -> std::io::Result<()> {
        let graph_file = self.log_dir.join("graph.txt");
        let mut file = File::create(graph_file)?;
        
        writeln!(file, "Model Architecture:")?;
        writeln!(file, "===================")?;
        
        for (i, (name, input_size, output_size)) in layer_info.iter().enumerate() {
            writeln!(file, "Layer {}: {} ({} -> {})", i, name, input_size, output_size)?;
        }
        
        Ok(())
    }
    
    /// Log learning rate
    pub fn add_lr(&mut self, lr: f32) -> std::io::Result<()> {
        self.add_scalar("learning_rate", lr)
    }
    
    /// Log train and validation metrics together
    pub fn add_train_val_metrics(&mut self, train_loss: f32, val_loss: f32, train_acc: f32, val_acc: f32) -> std::io::Result<()> {
        self.add_scalars("loss", &[
            ("train".to_string(), train_loss),
            ("validation".to_string(), val_loss),
        ])?;
        
        self.add_scalars("accuracy", &[
            ("train".to_string(), train_acc),
            ("validation".to_string(), val_acc),
        ])?;
        
        Ok(())
    }
    
    /// Increment the global step
    pub fn set_step(&mut self, step: i64) {
        self.step = step;
    }
    
    /// Get current wall time in seconds since start
    fn get_wall_time(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now - self.start_time
    }
    
    /// Flush all writers
    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(ref mut writer) = self.scalar_writer {
            writer.flush()?;
        }
        if let Some(ref mut writer) = self.histogram_writer {
            writer.flush()?;
        }
        Ok(())
    }
    
    /// Create a simple HTML dashboard
    pub fn create_dashboard(&self) -> std::io::Result<()> {
        let dashboard_path = self.log_dir.join("dashboard.html");
        let mut file = File::create(dashboard_path)?;
        
        writeln!(file, r#"<!DOCTYPE html>
<html>
<head>
    <title>Athena Training Dashboard - {}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
        .metrics {{ margin-top: 20px; }}
        pre {{ background: #f5f5f5; padding: 10px; }}
    </style>
</head>
<body>
    <h1>Athena Training Dashboard</h1>
    <div class="info">
        <strong>Run Name:</strong> {}<br>
        <strong>Log Directory:</strong> {}<br>
    </div>
    
    <div class="metrics">
        <h2>Metrics</h2>
        <p>To view the metrics:</p>
        <ol>
            <li>Open <code>scalars.csv</code> to see scalar metrics over time</li>
            <li>Open <code>histograms.csv</code> to see weight distributions</li>
            <li>Open <code>graph.txt</code> to see model architecture</li>
        </ol>
        
        <p>For full Tensorboard visualization, convert the CSV files to TensorBoard format or use a plotting library.</p>
    </div>
    
    <div class="instructions">
        <h2>Visualization Options</h2>
        <pre>
# Option 1: Use Python to plot the CSV files
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('scalars.csv')
for tag in df['tag'].unique():
    data = df[df['tag'] == tag]
    plt.figure()
    plt.plot(data['step'], data['value'])
    plt.title(tag)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.show()

# Option 2: Import into spreadsheet software
# The CSV files can be opened in Excel, Google Sheets, etc.
        </pre>
    </div>
</body>
</html>"#, self.run_name, self.run_name, self.log_dir.display())?;
        
        Ok(())
    }
}

impl Drop for TensorboardWriter {
    fn drop(&mut self) {
        let _ = self.flush();
        let _ = self.create_dashboard();
    }
}