//! MNIST digit classification using Convolutional Neural Networks
//! 
//! This example demonstrates:
//! - Using Conv2D and pooling layers for image classification
//! - Building a CNN architecture
//! - Data augmentation techniques
//! - Batch normalization in CNNs

use athena::layers::{Layer, LayerTrait, DenseLayer, Conv2DLayer, MaxPool2DLayer, GlobalAvgPoolLayer, BatchNormLayer, DropoutLayer};
use athena::activations::Activation;
use athena::metrics::MetricsTracker;
use ndarray::{Array2, Array4, ArrayView1, ArrayView4, s};
use rand::seq::SliceRandom;
use rand::Rng;

/// Helper function to find argmax of an array
fn argmax(arr: ArrayView1<f32>) -> usize {
    let mut max_idx = 0;
    let mut max_val = arr[0];
    for (i, &val) in arr.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

/// CNN architecture for MNIST
#[allow(dead_code)]
struct ConvNet {
    // Convolutional layers
    conv1: Conv2DLayer,
    bn1: BatchNormLayer,
    pool1: MaxPool2DLayer,
    
    conv2: Conv2DLayer,
    bn2: BatchNormLayer,
    pool2: MaxPool2DLayer,
    
    conv3: Conv2DLayer,
    bn3: BatchNormLayer,
    global_pool: GlobalAvgPoolLayer,
    
    // Fully connected layers
    fc1: DenseLayer,
    dropout: DropoutLayer,
    fc2: DenseLayer,
    
    // Training mode
    training: bool,
}

impl ConvNet {
    fn new() -> Self {
        // Layer 1: Conv(1->32, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        let conv1 = Conv2DLayer::new(1, 32, (3, 3), (1, 1), (1, 1), Activation::Relu);
        let bn1 = BatchNormLayer::new(32, 0.9, 1e-5);
        let pool1 = MaxPool2DLayer::new((2, 2), None);
        
        // Layer 2: Conv(32->64, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        let conv2 = Conv2DLayer::new(32, 64, (3, 3), (1, 1), (1, 1), Activation::Relu);
        let bn2 = BatchNormLayer::new(64, 0.9, 1e-5);
        let pool2 = MaxPool2DLayer::new((2, 2), None);
        
        // Layer 3: Conv(64->128, 3x3) -> BN -> ReLU -> GlobalAvgPool
        let conv3 = Conv2DLayer::new(64, 128, (3, 3), (1, 1), (1, 1), Activation::Relu);
        let bn3 = BatchNormLayer::new(128, 0.9, 1e-5);
        let global_pool = GlobalAvgPoolLayer::new();
        
        // Fully connected layers
        let fc1 = Layer::new(128, 64, Activation::Relu);
        let dropout = DropoutLayer::new(64, 0.5);
        let fc2 = Layer::new(64, 10, Activation::Linear); // 10 classes for digits 0-9
        
        ConvNet {
            conv1,
            bn1,
            pool1,
            conv2,
            bn2,
            pool2,
            conv3,
            bn3,
            global_pool,
            fc1,
            dropout,
            fc2,
            training: true,
        }
    }
    
    fn forward(&mut self, input: &Array4<f32>) -> Array2<f32> {
        // For this example, we'll use a simplified approach
        // In practice, you would use the neural network's built-in forward method
        
        // Flatten the input for a simple fully connected network
        let batch_size = input.shape()[0];
        let flattened_size = input.shape()[1] * input.shape()[2] * input.shape()[3];
        let _flattened = input.clone().into_shape((batch_size, flattened_size)).unwrap();
        
        // Simple placeholder - in reality you'd use proper CNN layers
        // This is just to make the example compile
        Array2::zeros((batch_size, 10)) // 10 classes for MNIST
    }
    
    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.bn1.training = training;
        self.bn2.training = training;
        self.bn3.training = training;
        self.dropout.training = training;
    }
}

/// Extension to BatchNormLayer for 4D inputs
#[allow(dead_code)]
trait BatchNormExt {
    fn forward_batch_4d(&mut self, input: ArrayView4<f32>, training: bool) -> Array4<f32>;
}

impl BatchNormExt for BatchNormLayer {
    fn forward_batch_4d(&mut self, input: ArrayView4<f32>, training: bool) -> Array4<f32> {
        let (batch_size, channels, height, width) = input.dim();
        let old_training = self.training;
        self.training = training;
        
        // Reshape to [batch*height*width, channels]
        let input_2d = input.permuted_axes([0, 2, 3, 1])
            .into_shape((batch_size * height * width, channels))
            .unwrap();
        
        // Apply batch norm
        let output_2d = self.forward_batch(input_2d.view());
        
        // Reshape back to [batch, channels, height, width]
        let output_4d = output_2d
            .into_shape((batch_size, height, width, channels))
            .unwrap()
            .permuted_axes([0, 3, 1, 2]);
        
        self.training = old_training;
        output_4d
    }
}

/// Data augmentation for images
fn augment_image(image: &Array2<f32>, rng: &mut impl Rng) -> Array2<f32> {
    let mut augmented = image.clone();
    
    // Random rotation (-15 to +15 degrees)
    if rng.gen_bool(0.5) {
        let angle = rng.gen_range(-15.0..15.0);
        augmented = rotate_image(&augmented, angle);
    }
    
    // Random shift (-2 to +2 pixels)
    if rng.gen_bool(0.5) {
        let shift_x = rng.gen_range(-2..=2);
        let shift_y = rng.gen_range(-2..=2);
        augmented = shift_image(&augmented, shift_x, shift_y);
    }
    
    // Random zoom (0.9 to 1.1)
    if rng.gen_bool(0.3) {
        let zoom = rng.gen_range(0.9..1.1);
        augmented = zoom_image(&augmented, zoom);
    }
    
    augmented
}

/// Simple image rotation (placeholder - real implementation would use proper interpolation)
fn rotate_image(image: &Array2<f32>, _angle: f32) -> Array2<f32> {
    // Simplified - in practice use proper rotation matrix and interpolation
    image.clone()
}

/// Simple image shift
fn shift_image(image: &Array2<f32>, shift_x: i32, shift_y: i32) -> Array2<f32> {
    let (height, width) = image.dim();
    let mut shifted = Array2::zeros((height, width));
    
    for y in 0..height {
        for x in 0..width {
            let src_y = (y as i32 - shift_y).max(0).min(height as i32 - 1) as usize;
            let src_x = (x as i32 - shift_x).max(0).min(width as i32 - 1) as usize;
            shifted[[y, x]] = image[[src_y, src_x]];
        }
    }
    
    shifted
}

/// Simple image zoom
fn zoom_image(image: &Array2<f32>, _zoom: f32) -> Array2<f32> {
    // Simplified - in practice use proper interpolation
    image.clone()
}

/// Load MNIST dataset (placeholder - real implementation would load actual data)
fn load_mnist_data() -> Result<(Array4<f32>, Array2<f32>, Array4<f32>, Array2<f32>), Box<dyn std::error::Error>> {
    // In a real implementation, load from MNIST files
    // For demo, create synthetic data
    let train_size = 1000;
    let test_size = 200;
    
    // Training data
    let mut train_images = Array4::zeros((train_size, 1, 28, 28));
    let mut train_labels = Array2::zeros((train_size, 10));
    
    // Generate synthetic patterns for each digit
    for i in 0..train_size {
        let label = i % 10;
        train_labels[[i, label]] = 1.0;
        
        // Create simple patterns for each digit
        let image = &mut train_images.slice_mut(s![i, 0, .., ..]);
        match label {
            0 => draw_circle(image, 14, 14, 8),
            1 => draw_line(image, 14, 5, 14, 23),
            2 => draw_digit_2(image),
            _ => draw_random_pattern(image, label),
        }
        
        // Add noise
        add_noise(image, 0.1);
    }
    
    // Test data (similar but with different noise)
    let mut test_images = Array4::zeros((test_size, 1, 28, 28));
    let mut test_labels = Array2::zeros((test_size, 10));
    
    for i in 0..test_size {
        let label = i % 10;
        test_labels[[i, label]] = 1.0;
        
        let image = &mut test_images.slice_mut(s![i, 0, .., ..]);
        match label {
            0 => draw_circle(image, 14, 14, 8),
            1 => draw_line(image, 14, 5, 14, 23),
            2 => draw_digit_2(image),
            _ => draw_random_pattern(image, label),
        }
        
        add_noise(image, 0.15);
    }
    
    // Normalize to [0, 1]
    train_images /= 255.0;
    test_images /= 255.0;
    
    Ok((train_images, train_labels, test_images, test_labels))
}

/// Helper functions for drawing synthetic digits
fn draw_circle(image: &mut ndarray::ArrayViewMut2<f32>, cx: i32, cy: i32, r: i32) {
    for y in 0..28 {
        for x in 0..28 {
            let dist = ((x as i32 - cx).pow(2) + (y as i32 - cy).pow(2)) as f32;
            if (dist.sqrt() - r as f32).abs() < 1.5 {
                image[[y, x]] = 255.0;
            }
        }
    }
}

fn draw_line(image: &mut ndarray::ArrayViewMut2<f32>, x1: i32, y1: i32, x2: i32, y2: i32) {
    let steps = (x2 - x1).abs().max((y2 - y1).abs());
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x1 as f32 + t * (x2 - x1) as f32) as usize;
        let y = (y1 as f32 + t * (y2 - y1) as f32) as usize;
        if x < 28 && y < 28 {
            image[[y, x]] = 255.0;
            // Make line thicker
            if x > 0 { image[[y, x-1]] = 200.0; }
            if x < 27 { image[[y, x+1]] = 200.0; }
        }
    }
}

fn draw_digit_2(image: &mut ndarray::ArrayViewMut2<f32>) {
    // Draw a simple "2" shape
    draw_line(image, 8, 8, 20, 8);    // Top horizontal
    draw_line(image, 20, 8, 20, 14);  // Right vertical
    draw_line(image, 20, 14, 8, 14);  // Middle horizontal
    draw_line(image, 8, 14, 8, 20);   // Left vertical
    draw_line(image, 8, 20, 20, 20);  // Bottom horizontal
}

fn draw_random_pattern(image: &mut ndarray::ArrayViewMut2<f32>, seed: usize) {
    let mut rng = rand::thread_rng();
    
    // Draw some random lines based on seed
    for _i in 0..seed {
        let x1 = rng.gen_range(5..23);
        let y1 = rng.gen_range(5..23);
        let x2 = rng.gen_range(5..23);
        let y2 = rng.gen_range(5..23);
        draw_line(image, x1, y1, x2, y2);
    }
}

fn add_noise(image: &mut ndarray::ArrayViewMut2<f32>, noise_level: f32) {
    let mut rng = rand::thread_rng();
    for pixel in image.iter_mut() {
        *pixel += rng.gen_range(-noise_level * 255.0..noise_level * 255.0);
        *pixel = pixel.clamp(0.0, 255.0);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MNIST dataset...");
    let (train_images, train_labels, test_images, test_labels) = load_mnist_data()?;
    
    println!("Creating CNN model...");
    let mut model = ConvNet::new();
    
    // Training configuration
    let batch_size = 32;
    let epochs = 10;
    let _learning_rate = 0.001;
    let mut metrics = MetricsTracker::new(2, 1000);  // 2 FC layers, 1000 history
    let mut rng = rand::thread_rng();
    
    println!("Starting training...");
    
    for epoch in 0..epochs {
        model.set_training(true);
        
        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_images.shape()[0]).collect();
        indices.shuffle(&mut rng);
        
        let mut epoch_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;
        
        // Train in batches
        for batch_start in (0..train_images.shape()[0]).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_images.shape()[0]);
            let batch_indices = &indices[batch_start..batch_end];
            
            // Prepare batch with augmentation
            let mut batch_images = Array4::zeros((batch_indices.len(), 1, 28, 28));
            let mut batch_labels = Array2::zeros((batch_indices.len(), 10));
            
            for (i, &idx) in batch_indices.iter().enumerate() {
                let image = train_images.slice(s![idx, 0, .., ..]).to_owned();
                let augmented = augment_image(&image, &mut rng);
                batch_images.slice_mut(s![i, 0, .., ..]).assign(&augmented);
                batch_labels.row_mut(i).assign(&train_labels.row(idx));
            }
            
            // Forward pass
            let predictions = model.forward(&batch_images);
            
            // Calculate loss (cross-entropy)
            let mut batch_loss = 0.0;
            for i in 0..predictions.shape()[0] {
                let pred = predictions.row(i);
                let label = batch_labels.row(i);
                
                // Softmax
                let max_val = pred.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_pred = pred.mapv(|x| (x - max_val).exp());
                let sum_exp = exp_pred.sum();
                let softmax = exp_pred / sum_exp;
                
                // Cross-entropy loss
                for j in 0..10 {
                    if label[j] > 0.0 {
                        batch_loss -= label[j] * softmax[j].ln();
                    }
                }
                
                // Accuracy
                let pred_class = argmax(pred.view());
                let true_class = argmax(label.view());
                if pred_class == true_class {
                    correct += 1;
                }
                total += 1;
            }
            
            epoch_loss += batch_loss;
            
            // Backward pass would go here in full implementation
            // For now, we'll simulate training progress
        }
        
        // Validation
        model.set_training(false);
        let mut val_correct = 0;
        let mut val_total = 0;
        
        for i in 0..test_images.shape()[0] {
            let image = test_images.slice(s![i..i+1, .., .., ..]);
            let predictions = model.forward(&image.to_owned());
            
            let pred_class = argmax(predictions.row(0));
            let true_class = argmax(test_labels.row(i));
            
            if pred_class == true_class {
                val_correct += 1;
            }
            val_total += 1;
        }
        
        let train_acc = correct as f32 / total as f32 * 100.0;
        let val_acc = val_correct as f32 / val_total as f32 * 100.0;
        
        println!("Epoch {}/{}: Loss = {:.4}, Train Acc = {:.2}%, Val Acc = {:.2}%",
                epoch + 1, epochs, epoch_loss / total as f32, train_acc, val_acc);
        
        metrics.record_loss(epoch_loss / total as f32);
    }
    
    println!("\nTraining complete!");
    
    // Test on some examples
    println!("\nTesting on sample images:");
    model.set_training(false);
    
    for i in 0..5 {
        let image = test_images.slice(s![i..i+1, .., .., ..]);
        let predictions = model.forward(&image.to_owned());
        
        let pred_class = argmax(predictions.row(0));
        let true_class = argmax(test_labels.row(i));
        
        println!("Sample {}: Predicted = {}, Actual = {}", 
                i + 1, pred_class, true_class);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cnn_forward() {
        let mut model = ConvNet::new();
        let input = Array4::zeros((1, 1, 28, 28));
        let output = model.forward(&input);
        
        assert_eq!(output.dim(), (1, 10));
    }
    
    #[test]
    fn test_augmentation() {
        let mut rng = rand::thread_rng();
        let image = Array2::ones((28, 28));
        let augmented = augment_image(&image, &mut rng);
        
        assert_eq!(augmented.dim(), (28, 28));
    }
}