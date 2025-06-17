use athena::network::NeuralNetwork;
use athena::activations::Activation;
use athena::optimizer::{OptimizerWrapper, Adam};
use athena::layers::{Layer, LayerTrait};
use athena::tensorboard::TensorboardWriter;
use ndarray::{array, Array1};
use std::f32::consts::PI;

fn main() -> std::io::Result<()> {
    println!("=== Tensorboard Logging Example ===\n");
    
    // Create Tensorboard writer
    let mut tb_writer = TensorboardWriter::new("logs", "example_run")?;
    
    // Create a simple network
    let layers = vec![Layer::new(2, 16, Activation::Relu)];
    let optimizer = OptimizerWrapper::Adam(Adam::new(&layers, 0.9, 0.999, 1e-8));
    let mut network = NeuralNetwork::new(
        &[2, 16, 8, 1],
        &[Activation::Relu, Activation::Relu, Activation::Linear],
        optimizer
    );
    
    // Log model architecture
    let layer_info = vec![
        ("Dense".to_string(), 2, 16),
        ("Dense".to_string(), 16, 8),
        ("Dense".to_string(), 8, 1),
    ];
    tb_writer.add_graph(&layer_info)?;
    
    // Training loop with logging
    println!("Training and logging metrics...");
    let _rng = rand::thread_rng();
    
    for epoch in 0..100 {
        let mut epoch_loss = 0.0;
        let mut epoch_accuracy = 0.0;
        
        // Generate synthetic data (sin function approximation)
        for _step in 0..50 {
            let x1 = rand::random::<f32>() * 2.0 * PI;
            let x2 = rand::random::<f32>() * 2.0 * PI;
            let target = (x1.sin() + x2.cos()) / 2.0;
            
            let input = array![x1, x2];
            let prediction = network.forward(input.view());
            
            let loss = (prediction[0] - target).powi(2);
            epoch_loss += loss;
            
            // Simple accuracy metric (within threshold)
            if (prediction[0] - target).abs() < 0.1 {
                epoch_accuracy += 1.0;
            }
        }
        
        // Calculate epoch metrics
        epoch_loss /= 50.0;
        epoch_accuracy /= 50.0;
        
        // Set step for logging
        tb_writer.set_step(epoch as i64);
        
        // Log scalar metrics
        tb_writer.add_scalar("loss/train", epoch_loss)?;
        tb_writer.add_scalar("accuracy/train", epoch_accuracy)?;
        
        // Log learning rate (simulated decay)
        let lr = 0.001 * 0.95_f32.powi(epoch / 10);
        tb_writer.add_lr(lr)?;
        
        // Log layer statistics every 10 epochs
        if epoch % 10 == 0 {
            for (i, layer) in network.layers.iter().enumerate() {
                tb_writer.add_layer_stats(
                    &format!("layer_{}", i),
                    layer.weights(),
                    layer.biases()
                )?;
            }
            
            // Simulate validation metrics
            let val_loss = epoch_loss * (1.0 + rand::random::<f32>() * 0.2 - 0.1);
            let val_accuracy = epoch_accuracy * (1.0 + rand::random::<f32>() * 0.1 - 0.05);
            
            tb_writer.add_train_val_metrics(
                epoch_loss,
                val_loss,
                epoch_accuracy,
                val_accuracy
            )?;
            
            println!("Epoch {}: Train Loss={:.4}, Val Loss={:.4}, Train Acc={:.2}%, Val Acc={:.2}%",
                     epoch, epoch_loss, val_loss, epoch_accuracy * 100.0, val_accuracy * 100.0);
        }
        
        // Log custom histograms
        if epoch % 20 == 0 {
            // Generate some distribution data
            let normal_dist: Vec<f32> = (0..100)
                .map(|_| {
                    let u1: f32 = rand::random();
                    let u2: f32 = rand::random();
                    // Box-Muller transform
                    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
                })
                .collect();
            
            let dist_array = Array1::from_vec(normal_dist);
            tb_writer.add_histogram("distributions/normal", &dist_array)?;
        }
    }
    
    // Flush all data
    tb_writer.flush()?;
    
    println!("\n=== Logging Complete ===");
    println!("Logs saved to: logs/example_run/");
    println!("Files created:");
    println!("  - scalars.csv: Scalar metrics over time");
    println!("  - histograms.csv: Weight and bias distributions");
    println!("  - graph.txt: Model architecture");
    println!("  - dashboard.html: Simple visualization dashboard");
    
    println!("\nTo visualize the data:");
    println!("1. Open dashboard.html in a web browser");
    println!("2. Use a spreadsheet program to plot the CSV files");
    println!("3. Use the Python script shown in dashboard.html");
    
    Ok(())
}