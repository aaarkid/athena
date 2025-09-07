use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Trait defining the interface for loss functions
pub trait Loss: Send + Sync {
    /// Compute the loss for a single prediction and target
    fn compute(&self, prediction: ArrayView1<f32>, target: ArrayView1<f32>) -> f32;
    
    /// Compute the loss for a batch of predictions and targets
    fn compute_batch(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32;
    
    /// Compute the gradient of the loss with respect to predictions
    fn gradient(&self, prediction: ArrayView1<f32>, target: ArrayView1<f32>) -> Array1<f32>;
    
    /// Compute the gradient of the loss for a batch
    fn gradient_batch(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32>;
}

/// Mean Squared Error loss
pub struct MSE;

impl Loss for MSE {
    fn compute(&self, prediction: ArrayView1<f32>, target: ArrayView1<f32>) -> f32 {
        let diff = &prediction - &target;
        (&diff * &diff).sum() / (2.0 * prediction.len() as f32)
    }
    
    fn compute_batch(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        let diff = &predictions - &targets;
        (&diff * &diff).sum() / (2.0 * predictions.shape()[0] as f32 * predictions.shape()[1] as f32)
    }
    
    fn gradient(&self, prediction: ArrayView1<f32>, target: ArrayView1<f32>) -> Array1<f32> {
        (&prediction - &target) / prediction.len() as f32
    }
    
    fn gradient_batch(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32> {
        (&predictions - &targets) / predictions.shape()[0] as f32
    }
}

/// Huber loss (smooth L1)
pub struct HuberLoss {
    pub delta: f32,
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        HuberLoss { delta }
    }
}

impl Loss for HuberLoss {
    fn compute(&self, prediction: ArrayView1<f32>, target: ArrayView1<f32>) -> f32 {
        let diff = &prediction - &target;
        diff.mapv(|x| {
            let abs_x = x.abs();
            if abs_x <= self.delta {
                0.5 * x * x
            } else {
                self.delta * abs_x - 0.5 * self.delta * self.delta
            }
        }).sum() / prediction.len() as f32
    }
    
    fn compute_batch(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        let diff = &predictions - &targets;
        let batch_size = predictions.shape()[0] as f32;
        diff.mapv(|x| {
            let abs_x = x.abs();
            if abs_x <= self.delta {
                0.5 * x * x
            } else {
                self.delta * abs_x - 0.5 * self.delta * self.delta
            }
        }).sum() / batch_size
    }
    
    fn gradient(&self, prediction: ArrayView1<f32>, target: ArrayView1<f32>) -> Array1<f32> {
        let diff = &prediction - &target;
        diff.mapv(|x| {
            if x.abs() <= self.delta {
                x
            } else {
                self.delta * x.signum()
            }
        }) / prediction.len() as f32
    }
    
    fn gradient_batch(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32> {
        let diff = &predictions - &targets;
        let batch_size = predictions.shape()[0] as f32;
        diff.mapv(|x| {
            if x.abs() <= self.delta {
                x
            } else {
                self.delta * x.signum()
            }
        }) / batch_size
    }
}

/// Cross-entropy loss for classification
pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn compute(&self, prediction: ArrayView1<f32>, target: ArrayView1<f32>) -> f32 {
        let epsilon = 1e-7;
        -target.iter()
            .zip(prediction.iter())
            .map(|(&t, &p)| t * (p + epsilon).ln())
            .sum::<f32>() / prediction.len() as f32
    }
    
    fn compute_batch(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        let epsilon = 1e-7;
        let batch_size = predictions.shape()[0] as f32;
        
        let mut total_loss = 0.0;
        for (pred_row, target_row) in predictions.axis_iter(ndarray::Axis(0))
            .zip(targets.axis_iter(ndarray::Axis(0))) {
            total_loss += -target_row.iter()
                .zip(pred_row.iter())
                .map(|(&t, &p)| t * (p + epsilon).ln())
                .sum::<f32>();
        }
        total_loss / batch_size
    }
    
    fn gradient(&self, prediction: ArrayView1<f32>, target: ArrayView1<f32>) -> Array1<f32> {
        let epsilon = 1e-7;
        Array1::from_shape_fn(prediction.len(), |i| {
            -(target[i] / (prediction[i] + epsilon)) / prediction.len() as f32
        })
    }
    
    fn gradient_batch(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> Array2<f32> {
        let epsilon = 1e-7;
        let batch_size = predictions.shape()[0] as f32;
        
        Array2::from_shape_fn(predictions.dim(), |(i, j)| {
            -(targets[[i, j]] / (predictions[[i, j]] + epsilon)) / batch_size
        })
    }
}