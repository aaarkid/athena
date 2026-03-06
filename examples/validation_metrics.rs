use athena::network::NeuralNetwork;
use athena::activations::Activation;
use athena::optimizer::{OptimizerWrapper, Adam};
use athena::layers::Layer;
use athena::metrics::{ClassificationMetrics, RegressionMetrics, ExtendedMetricsTracker};
use ndarray::array;

fn main() {
    println!("=== Validation Metrics Example ===\n");
    
    // Example 1: Binary Classification Metrics
    println!("1. Binary Classification Metrics:");
    
    let predictions = array![0.9, 0.1, 0.8, 0.3, 0.7, 0.2, 0.95, 0.05];
    let targets = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    
    let accuracy = ClassificationMetrics::accuracy(predictions.view(), targets.view());
    println!("   Accuracy: {:.2}%", accuracy * 100.0);
    
    let (precision, recall, f1) = ClassificationMetrics::precision_recall_f1(
        predictions.view(), targets.view(), 0.5
    );
    println!("   Precision: {:.2}%", precision * 100.0);
    println!("   Recall: {:.2}%", recall * 100.0);
    println!("   F1 Score: {:.2}", f1);
    
    let confusion = ClassificationMetrics::confusion_matrix(
        predictions.view(), targets.view(), 0.5
    );
    println!("   Confusion Matrix:");
    println!("                 Predicted");
    println!("                 Neg  Pos");
    println!("   Actual Neg  [{:3}, {:3}]", confusion[0][0], confusion[0][1]);
    println!("   Actual Pos  [{:3}, {:3}]", confusion[1][0], confusion[1][1]);
    
    // Example 2: Multi-class Classification
    println!("\n2. Multi-class Classification Metrics:");
    
    let multiclass_preds = array![
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ];
    let multiclass_targets = array![0, 1, 2, 0, 1, 2];
    
    let mc_accuracy = ClassificationMetrics::multiclass_accuracy(
        multiclass_preds.view(), multiclass_targets.view()
    );
    println!("   Multi-class Accuracy: {:.2}%", mc_accuracy * 100.0);
    
    let per_class = ClassificationMetrics::per_class_metrics(
        multiclass_preds.view(), multiclass_targets.view()
    );
    for (i, (prec, rec)) in per_class.iter().enumerate() {
        println!("   Class {}: Precision={:.2}%, Recall={:.2}%", 
                 i, prec * 100.0, rec * 100.0);
    }
    
    // Example 3: Regression Metrics
    println!("\n3. Regression Metrics:");
    
    let reg_predictions = array![2.5, 0.0, 2.1, 7.8];
    let reg_targets = array![3.0, -0.5, 2.0, 8.0];
    
    let mse = RegressionMetrics::mse(reg_predictions.view(), reg_targets.view());
    let rmse = RegressionMetrics::rmse(reg_predictions.view(), reg_targets.view());
    let mae = RegressionMetrics::mae(reg_predictions.view(), reg_targets.view());
    let r2 = RegressionMetrics::r_squared(reg_predictions.view(), reg_targets.view());
    
    println!("   MSE: {:.4}", mse);
    println!("   RMSE: {:.4}", rmse);
    println!("   MAE: {:.4}", mae);
    println!("   R²: {:.4}", r2);
    
    // Example 4: Training with Validation Set
    println!("\n4. Training with Validation Set:");
    
    // Create a simple network for binary classification
    let layers = vec![Layer::new(2, 4, Activation::Relu)];
    let optimizer = OptimizerWrapper::Adam(Adam::new(&layers, 0.9, 0.999, 1e-8));
    let mut network = NeuralNetwork::new(
        &[2, 4, 1],
        &[Activation::Relu, Activation::Sigmoid],
        optimizer
    );
    
    // Create training data (XOR problem)
    let train_inputs = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let train_targets = array![[0.0], [1.0], [1.0], [0.0]];
    
    // Create validation set
    let val_inputs = train_inputs.clone();
    let val_targets = train_targets.clone();
    
    // Create metrics tracker
    let mut tracker = ExtendedMetricsTracker::new();
    
    // Training loop with validation
    println!("   Training XOR problem...");
    for epoch in 0..100 {
        let mut epoch_loss = 0.0;
        
        // Training
        for i in 0..train_inputs.shape()[0] {
            let input = train_inputs.row(i);
            let target = train_targets.row(i);
            
            let prediction = network.forward(input);
            let loss = ((&prediction - &target) * (&prediction - &target)).sum();
            epoch_loss += loss;
        }
        
        // Track training loss
        tracker.add_metric("train_loss", epoch_loss / train_inputs.shape()[0] as f32);
        
        // Validation every 20 epochs
        if epoch % 20 == 0 {
            // Calculate validation accuracy manually
            let mut correct = 0;
            for i in 0..val_inputs.shape()[0] {
                let input = val_inputs.row(i);
                let target = val_targets.row(i);
                let prediction = network.forward(input);
                
                if ((prediction[0] > 0.5) as i32) == (target[0] as i32) {
                    correct += 1;
                }
            }
            let val_accuracy = correct as f32 / val_inputs.shape()[0] as f32;
            
            println!("   Epoch {}: Train Loss={:.4}, Val Accuracy={:.2}%", 
                     epoch, epoch_loss / 4.0, val_accuracy * 100.0);
        }
    }
    
    // Example 5: Metrics Visualization
    println!("\n5. Metrics Visualization (ASCII Plot):");
    
    // Create some dummy metrics for visualization
    let mut viz_tracker = ExtendedMetricsTracker::new();
    for i in 0..50 {
        let loss = 1.0 / (1.0 + i as f32 * 0.1);
        let accuracy = 1.0 - loss + (i as f32 * 0.01).sin() * 0.1;
        
        viz_tracker.add_metric("loss", loss);
        viz_tracker.add_metric("accuracy", accuracy);
    }
    
    // Plot the metrics
    viz_tracker.plot_ascii("loss", 60, 10);
    viz_tracker.plot_ascii("accuracy", 60, 10);
    
    // Show statistics
    if let Some(stats) = viz_tracker.get_stats("loss") {
        println!("\nLoss Statistics:");
        println!("  Mean: {:.4}", stats.mean);
        println!("  Std Dev: {:.4}", stats.std_dev);
        println!("  Min: {:.4}", stats.min);
        println!("  Max: {:.4}", stats.max);
    }
}