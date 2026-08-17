//! Shape classification with a small convolutional network.
//!
//! The images are generated, not loaded: four shapes (horizontal bar, vertical
//! bar, cross, filled square) drawn at random positions on a 12x12 grid with
//! noise. That keeps the example self-contained while still exercising the real
//! conv, pooling and dense layers, including their backward passes.
//!
//! Run with: cargo run --release --example conv_shapes

use athena::activations::Activation;
use athena::layers::{Conv2DLayer, DenseLayer, LayerTrait, MaxPool2DLayer};
use ndarray::{Array2, Array4, Axis};
use rand::seq::SliceRandom;
use rand::Rng;

const IMAGE_SIZE: usize = 12;
const NUM_CLASSES: usize = 4;
const TRAIN_SAMPLES: usize = 800;
const TEST_SAMPLES: usize = 200;
const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 15;
const LEARNING_RATE: f32 = 0.05;

/// Draw one shape of the given class into a fresh image
fn draw_shape(class: usize, rng: &mut impl Rng) -> Array2<f32> {
    let mut image = Array2::zeros((IMAGE_SIZE, IMAGE_SIZE));

    match class {
        // Horizontal bar
        0 => {
            let row = rng.gen_range(2..IMAGE_SIZE - 2);
            let start = rng.gen_range(1..4);
            let end = rng.gen_range(IMAGE_SIZE - 4..IMAGE_SIZE - 1);
            for col in start..end {
                image[[row, col]] = 1.0;
                image[[row + 1, col]] = 1.0;
            }
        }
        // Vertical bar
        1 => {
            let col = rng.gen_range(2..IMAGE_SIZE - 2);
            let start = rng.gen_range(1..4);
            let end = rng.gen_range(IMAGE_SIZE - 4..IMAGE_SIZE - 1);
            for row in start..end {
                image[[row, col]] = 1.0;
                image[[row, col + 1]] = 1.0;
            }
        }
        // Cross
        2 => {
            let centre = rng.gen_range(4..IMAGE_SIZE - 4);
            for i in 1..IMAGE_SIZE - 1 {
                image[[centre, i]] = 1.0;
                image[[i, centre]] = 1.0;
            }
        }
        // Filled square
        _ => {
            let size = rng.gen_range(3..6);
            let top = rng.gen_range(1..IMAGE_SIZE - size - 1);
            let left = rng.gen_range(1..IMAGE_SIZE - size - 1);
            for row in top..top + size {
                for col in left..left + size {
                    image[[row, col]] = 1.0;
                }
            }
        }
    }

    // A little noise so the classes are not trivially separable
    for value in image.iter_mut() {
        *value += rng.gen_range(-0.1..0.1);
    }

    image
}

/// Build a dataset of `count` images with one-hot labels
fn make_dataset(count: usize, rng: &mut impl Rng) -> (Array4<f32>, Array2<f32>) {
    let mut images = Array4::zeros((count, 1, IMAGE_SIZE, IMAGE_SIZE));
    let mut labels = Array2::zeros((count, NUM_CLASSES));

    for i in 0..count {
        let class = i % NUM_CLASSES;
        let image = draw_shape(class, rng);
        images.slice_mut(ndarray::s![i, 0, .., ..]).assign(&image);
        labels[[i, class]] = 1.0;
    }

    (images, labels)
}

/// Row-wise softmax
fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut result = logits.clone();
    for mut row in result.axis_iter_mut(Axis(0)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|x| (x - max).exp());
        let sum: f32 = row.sum();
        row.mapv_inplace(|x| x / sum);
    }
    result
}

fn argmax(row: ndarray::ArrayView1<f32>) -> usize {
    let mut best = 0;
    for (i, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = i;
        }
    }
    best
}

/// Conv -> pool -> conv -> pool -> flatten -> dense, wired by hand
struct ShapeNet {
    conv1: Conv2DLayer,
    pool1: MaxPool2DLayer,
    conv2: Conv2DLayer,
    pool2: MaxPool2DLayer,
    dense: DenseLayer,
    // Shape of the tensor that gets flattened, needed on the way back
    flat_shape: (usize, usize, usize, usize),
}

impl ShapeNet {
    fn new() -> Self {
        // 12x12 -> conv -> pool 6x6 -> conv -> pool 3x3, 16 channels
        ShapeNet {
            conv1: Conv2DLayer::new(1, 8, (3, 3), (1, 1), (1, 1), Activation::Relu),
            pool1: MaxPool2DLayer::new((2, 2), None),
            conv2: Conv2DLayer::new(8, 16, (3, 3), (1, 1), (1, 1), Activation::Relu),
            pool2: MaxPool2DLayer::new((2, 2), None),
            dense: DenseLayer::new(16 * 3 * 3, NUM_CLASSES, Activation::Linear),
            flat_shape: (0, 0, 0, 0),
        }
    }

    fn forward(&mut self, images: &Array4<f32>) -> Array2<f32> {
        let c1 = self.conv1.forward_batch(images.view());
        let p1 = self.pool1.forward_batch(c1.view());
        let c2 = self.conv2.forward_batch(p1.view());
        let p2 = self.pool2.forward_batch(c2.view());

        self.flat_shape = p2.dim();
        let (batch, channels, height, width) = self.flat_shape;
        let flat = p2.into_shape((batch, channels * height * width)).expect("flatten");

        self.dense.forward_batch(flat.view())
    }

    /// Backpropagate the loss gradient and apply plain SGD
    fn backward(&mut self, logit_grad: &Array2<f32>, learning_rate: f32) {
        let (dense_error, dense_wgrad, dense_bgrad) = self.dense.backward_batch(logit_grad.view());
        let flat_grad = dense_error.dot(&self.dense.weights.t());

        let conv2_grad = self.pool2.backward_batch(
            flat_grad.into_shape(self.flat_shape).expect("unflatten").view(),
        );
        let (pool_grad, conv2_wgrad, conv2_bgrad) = self.conv2.backward_batch(conv2_grad.view());
        let conv1_grad = self.pool1.backward_batch(pool_grad.view());
        let (_, conv1_wgrad, conv1_bgrad) = self.conv1.backward_batch(conv1_grad.view());

        self.dense.weights.scaled_add(-learning_rate, &dense_wgrad);
        self.dense.biases.scaled_add(-learning_rate, &dense_bgrad);
        self.conv2.kernels.scaled_add(-learning_rate, &conv2_wgrad);
        self.conv2.biases.scaled_add(-learning_rate, &conv2_bgrad);
        self.conv1.kernels.scaled_add(-learning_rate, &conv1_wgrad);
        self.conv1.biases.scaled_add(-learning_rate, &conv1_bgrad);
    }

    fn accuracy(&mut self, images: &Array4<f32>, labels: &Array2<f32>) -> f32 {
        let count = images.shape()[0];
        let mut correct = 0;

        for start in (0..count).step_by(BATCH_SIZE) {
            let end = (start + BATCH_SIZE).min(count);
            let batch = images.slice(ndarray::s![start..end, .., .., ..]).to_owned();
            let logits = self.forward(&batch);

            for (i, row) in logits.axis_iter(Axis(0)).enumerate() {
                if argmax(row) == argmax(labels.row(start + i)) {
                    correct += 1;
                }
            }
        }

        correct as f32 / count as f32
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    println!("Generating {TRAIN_SAMPLES} training and {TEST_SAMPLES} test images");
    let (train_images, train_labels) = make_dataset(TRAIN_SAMPLES, &mut rng);
    let (test_images, test_labels) = make_dataset(TEST_SAMPLES, &mut rng);

    let mut model = ShapeNet::new();
    let mut order: Vec<usize> = (0..TRAIN_SAMPLES).collect();

    println!("\nepoch   loss   train acc   test acc");
    for epoch in 1..=EPOCHS {
        order.shuffle(&mut rng);
        let mut epoch_loss = 0.0;
        let mut batches = 0;

        for chunk in order.chunks(BATCH_SIZE) {
            let mut images = Array4::zeros((chunk.len(), 1, IMAGE_SIZE, IMAGE_SIZE));
            let mut labels = Array2::zeros((chunk.len(), NUM_CLASSES));
            for (slot, &index) in chunk.iter().enumerate() {
                images
                    .slice_mut(ndarray::s![slot, .., .., ..])
                    .assign(&train_images.slice(ndarray::s![index, .., .., ..]));
                labels.row_mut(slot).assign(&train_labels.row(index));
            }

            let logits = model.forward(&images);
            let probabilities = softmax(&logits);

            // Cross entropy, and its gradient with respect to the logits
            let mut batch_loss = 0.0;
            for (probs, target) in probabilities.axis_iter(Axis(0)).zip(labels.axis_iter(Axis(0))) {
                for (&p, &t) in probs.iter().zip(target.iter()) {
                    if t > 0.0 {
                        batch_loss -= (p.max(1e-7)).ln();
                    }
                }
            }
            epoch_loss += batch_loss / chunk.len() as f32;
            batches += 1;

            let grad = (&probabilities - &labels) / chunk.len() as f32;
            model.backward(&grad, LEARNING_RATE);
        }

        let train_accuracy = model.accuracy(&train_images, &train_labels);
        let test_accuracy = model.accuracy(&test_images, &test_labels);
        println!(
            "{epoch:>5}   {:.3}   {:>9.1}%   {:>7.1}%",
            epoch_loss / batches as f32,
            train_accuracy * 100.0,
            test_accuracy * 100.0
        );
    }

    // Show what the trained network makes of one fresh image per class
    println!("\nPredictions on new images:");
    let names = ["horizontal", "vertical", "cross", "square"];
    for class in 0..NUM_CLASSES {
        let image = draw_shape(class, &mut rng);
        let mut batch = Array4::zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE));
        batch.slice_mut(ndarray::s![0, 0, .., ..]).assign(&image);

        let probabilities = softmax(&model.forward(&batch));
        let predicted = argmax(probabilities.row(0));
        let confidence = probabilities[[0, predicted]];

        println!(
            "  drew {:<11} -> {:<11} ({:.0}% confident){}",
            names[class],
            names[predicted],
            confidence * 100.0,
            if predicted == class { "" } else { "   <- wrong" }
        );
    }
}
