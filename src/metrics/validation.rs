use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Metrics for model validation and evaluation
pub struct ValidationMetrics {
    /// Running average of metrics
    metrics: HashMap<String, Vec<f32>>,
}

impl ValidationMetrics {
    /// Create new validation metrics tracker
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
    
    /// Add a metric value
    pub fn add(&mut self, name: &str, value: f32) {
        self.metrics.entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }
    
    /// Get average of a metric
    pub fn get_average(&self, name: &str) -> Option<f32> {
        self.metrics.get(name).map(|values| {
            values.iter().sum::<f32>() / values.len() as f32
        })
    }
    
    /// Get all values for a metric
    pub fn get_values(&self, name: &str) -> Option<&Vec<f32>> {
        self.metrics.get(name)
    }
    
    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }
    
    /// Get summary of all metrics
    pub fn summary(&self) -> HashMap<String, f32> {
        self.metrics.iter()
            .map(|(name, values)| {
                let avg = values.iter().sum::<f32>() / values.len() as f32;
                (name.clone(), avg)
            })
            .collect()
    }
}

/// Classification metrics
pub struct ClassificationMetrics;

impl ClassificationMetrics {
    /// Calculate accuracy
    pub fn accuracy(predictions: ArrayView1<f32>, targets: ArrayView1<f32>) -> f32 {
        let correct = predictions.iter()
            .zip(targets.iter())
            .filter(|(p, t)| ((**p > 0.5) as i32) == (**t as i32))
            .count();
        
        correct as f32 / predictions.len() as f32
    }
    
    /// Calculate binary accuracy with threshold
    pub fn binary_accuracy(predictions: ArrayView1<f32>, targets: ArrayView1<f32>, threshold: f32) -> f32 {
        let correct = predictions.iter()
            .zip(targets.iter())
            .filter(|(p, t)| ((**p > threshold) as i32) == (**t as i32))
            .count();
        
        correct as f32 / predictions.len() as f32
    }
    
    /// Calculate precision, recall, and F1 score
    pub fn precision_recall_f1(predictions: ArrayView1<f32>, targets: ArrayView1<f32>, threshold: f32) -> (f32, f32, f32) {
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_positive = *pred > threshold;
            let target_positive = *target > 0.5;
            
            if pred_positive && target_positive {
                true_positives += 1;
            } else if pred_positive && !target_positive {
                false_positives += 1;
            } else if !pred_positive && target_positive {
                false_negatives += 1;
            }
        }
        
        let precision = if true_positives + false_positives > 0 {
            true_positives as f32 / (true_positives + false_positives) as f32
        } else {
            0.0
        };
        
        let recall = if true_positives + false_negatives > 0 {
            true_positives as f32 / (true_positives + false_negatives) as f32
        } else {
            0.0
        };
        
        let f1 = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };
        
        (precision, recall, f1)
    }
    
    /// Calculate confusion matrix for binary classification
    pub fn confusion_matrix(predictions: ArrayView1<f32>, targets: ArrayView1<f32>, threshold: f32) -> [[u32; 2]; 2] {
        let mut matrix = [[0u32; 2]; 2];
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_class = (*pred > threshold) as usize;
            let target_class = (*target > 0.5) as usize;
            matrix[target_class][pred_class] += 1;
        }
        
        matrix
    }
    
    /// Calculate multi-class accuracy
    pub fn multiclass_accuracy(predictions: ArrayView2<f32>, targets: ArrayView1<usize>) -> f32 {
        let pred_classes: Vec<usize> = predictions
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();
        
        let correct = pred_classes.iter()
            .zip(targets.iter())
            .filter(|(p, t)| *p == *t)
            .count();
        
        correct as f32 / predictions.shape()[0] as f32
    }
    
    /// Calculate per-class precision and recall
    pub fn per_class_metrics(predictions: ArrayView2<f32>, targets: ArrayView1<usize>) -> Vec<(f32, f32)> {
        let num_classes = predictions.shape()[1];
        let pred_classes: Vec<usize> = predictions
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();
        
        let mut metrics = vec![(0.0, 0.0); num_classes];
        
        for class_idx in 0..num_classes {
            let mut true_positives = 0;
            let mut false_positives = 0;
            let mut false_negatives = 0;
            
            for (pred, target) in pred_classes.iter().zip(targets.iter()) {
                if *pred == class_idx && *target == class_idx {
                    true_positives += 1;
                } else if *pred == class_idx && *target != class_idx {
                    false_positives += 1;
                } else if *pred != class_idx && *target == class_idx {
                    false_negatives += 1;
                }
            }
            
            let precision = if true_positives + false_positives > 0 {
                true_positives as f32 / (true_positives + false_positives) as f32
            } else {
                0.0
            };
            
            let recall = if true_positives + false_negatives > 0 {
                true_positives as f32 / (true_positives + false_negatives) as f32
            } else {
                0.0
            };
            
            metrics[class_idx] = (precision, recall);
        }
        
        metrics
    }
}

/// Regression metrics
pub struct RegressionMetrics;

impl RegressionMetrics {
    /// Mean Squared Error
    pub fn mse(predictions: ArrayView1<f32>, targets: ArrayView1<f32>) -> f32 {
        let diff = &predictions.to_owned() - &targets;
        (&diff * &diff).mean().unwrap()
    }
    
    /// Root Mean Squared Error
    pub fn rmse(predictions: ArrayView1<f32>, targets: ArrayView1<f32>) -> f32 {
        Self::mse(predictions, targets).sqrt()
    }
    
    /// Mean Absolute Error
    pub fn mae(predictions: ArrayView1<f32>, targets: ArrayView1<f32>) -> f32 {
        (&predictions.to_owned() - &targets).mapv(f32::abs).mean().unwrap()
    }
    
    /// R-squared (coefficient of determination)
    pub fn r_squared(predictions: ArrayView1<f32>, targets: ArrayView1<f32>) -> f32 {
        let mean_target = targets.mean().unwrap();
        let ss_tot = targets.mapv(|t| (t - mean_target).powi(2)).sum();
        let ss_res = (&predictions.to_owned() - &targets).mapv(|e| e.powi(2)).sum();
        
        1.0 - (ss_res / ss_tot)
    }
    
    /// Mean Absolute Percentage Error
    pub fn mape(predictions: ArrayView1<f32>, targets: ArrayView1<f32>) -> f32 {
        predictions.iter()
            .zip(targets.iter())
            .filter(|(_, t)| **t != 0.0)
            .map(|(p, t)| ((p - t) / t).abs())
            .sum::<f32>() / targets.len() as f32 * 100.0
    }
}

/// Validation set handler
pub struct ValidationSet {
    pub inputs: Array2<f32>,
    pub targets: Array2<f32>,
    pub metrics: ValidationMetrics,
}

impl ValidationSet {
    /// Create new validation set
    pub fn new(inputs: Array2<f32>, targets: Array2<f32>) -> Self {
        Self {
            inputs,
            targets,
            metrics: ValidationMetrics::new(),
        }
    }
    
    /// Evaluate model on validation set
    pub fn evaluate<F>(&mut self, predict_fn: F, task_type: &str) -> HashMap<String, f32>
    where
        F: Fn(ArrayView1<f32>) -> Array1<f32>,
    {
        self.metrics.clear();
        
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();
        
        // Get predictions for all samples
        for i in 0..self.inputs.shape()[0] {
            let input = self.inputs.row(i);
            let target = self.targets.row(i);
            let prediction = predict_fn(input);
            
            all_predictions.extend(prediction.iter().cloned());
            all_targets.extend(target.iter().cloned());
        }
        
        let predictions = Array1::from_vec(all_predictions);
        let targets = Array1::from_vec(all_targets);
        
        // Calculate metrics based on task type
        match task_type {
            "binary_classification" => {
                let accuracy = ClassificationMetrics::binary_accuracy(predictions.view(), targets.view(), 0.5);
                let (precision, recall, f1) = ClassificationMetrics::precision_recall_f1(predictions.view(), targets.view(), 0.5);
                
                self.metrics.add("accuracy", accuracy);
                self.metrics.add("precision", precision);
                self.metrics.add("recall", recall);
                self.metrics.add("f1_score", f1);
            }
            "regression" => {
                let mse = RegressionMetrics::mse(predictions.view(), targets.view());
                let rmse = RegressionMetrics::rmse(predictions.view(), targets.view());
                let mae = RegressionMetrics::mae(predictions.view(), targets.view());
                let r2 = RegressionMetrics::r_squared(predictions.view(), targets.view());
                
                self.metrics.add("mse", mse);
                self.metrics.add("rmse", rmse);
                self.metrics.add("mae", mae);
                self.metrics.add("r_squared", r2);
            }
            _ => {
                // Default to MSE
                let mse = RegressionMetrics::mse(predictions.view(), targets.view());
                self.metrics.add("mse", mse);
            }
        }
        
        self.metrics.summary()
    }
}