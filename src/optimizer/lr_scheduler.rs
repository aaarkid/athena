use serde::{Serialize, Deserialize};

/// Learning rate scheduling strategies
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LearningRateScheduler {
    /// Constant learning rate
    Constant { lr: f32 },
    
    /// Step decay: lr = initial_lr * decay_rate^(step / step_size)
    StepDecay {
        initial_lr: f32,
        decay_rate: f32,
        step_size: usize,
    },
    
    /// Exponential decay: lr = initial_lr * decay_rate^step
    ExponentialDecay {
        initial_lr: f32,
        decay_rate: f32,
    },
    
    /// Linear decay: lr = initial_lr * (1 - step / max_steps)
    LinearDecay {
        initial_lr: f32,
        final_lr: f32,
        max_steps: usize,
    },
    
    /// Cosine annealing: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / max_steps))
    CosineAnnealing {
        max_lr: f32,
        min_lr: f32,
        period: usize,
    },
    
    /// Warm restart with cosine annealing
    CosineAnnealingWarmRestarts {
        max_lr: f32,
        min_lr: f32,
        period: usize,
        mult: f32,
    },
    
    /// Linear warmup then constant
    WarmupConstant {
        initial_lr: f32,
        warmup_steps: usize,
        target_lr: f32,
    },
    
    /// Polynomial decay
    PolynomialDecay {
        initial_lr: f32,
        final_lr: f32,
        max_steps: usize,
        power: f32,
    },
}

impl LearningRateScheduler {
    /// Get the learning rate for a given step
    pub fn get_lr(&self, step: usize) -> f32 {
        match self {
            LearningRateScheduler::Constant { lr } => *lr,
            
            LearningRateScheduler::StepDecay { initial_lr, decay_rate, step_size } => {
                let num_decays = (step / step_size) as f32;
                initial_lr * decay_rate.powf(num_decays)
            }
            
            LearningRateScheduler::ExponentialDecay { initial_lr, decay_rate } => {
                initial_lr * decay_rate.powf(step as f32)
            }
            
            LearningRateScheduler::LinearDecay { initial_lr, final_lr, max_steps } => {
                if step >= *max_steps {
                    *final_lr
                } else {
                    let progress = step as f32 / *max_steps as f32;
                    initial_lr * (1.0 - progress) + final_lr * progress
                }
            }
            
            LearningRateScheduler::CosineAnnealing { max_lr, min_lr, period } => {
                let progress = (step % period) as f32 / *period as f32;
                min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
            }
            
            LearningRateScheduler::CosineAnnealingWarmRestarts { max_lr, min_lr, period, mult } => {
                // Find which restart period we're in
                let mut current_period = *period;
                let mut step_in_period = step;
                let mut period_start = 0;
                
                while step_in_period >= current_period {
                    period_start += current_period;
                    step_in_period = step - period_start;
                    current_period = (current_period as f32 * mult) as usize;
                }
                
                let progress = step_in_period as f32 / current_period as f32;
                min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
            }
            
            LearningRateScheduler::WarmupConstant { initial_lr, warmup_steps, target_lr } => {
                if step < *warmup_steps {
                    let progress = step as f32 / *warmup_steps as f32;
                    initial_lr + (target_lr - initial_lr) * progress
                } else {
                    *target_lr
                }
            }
            
            LearningRateScheduler::PolynomialDecay { initial_lr, final_lr, max_steps, power } => {
                if step >= *max_steps {
                    *final_lr
                } else {
                    let progress = 1.0 - (step as f32 / *max_steps as f32);
                    final_lr + (initial_lr - final_lr) * progress.powf(*power)
                }
            }
        }
    }
    
    /// Create a constant learning rate scheduler
    pub fn constant(lr: f32) -> Self {
        LearningRateScheduler::Constant { lr }
    }
    
    /// Create a step decay scheduler
    pub fn step_decay(initial_lr: f32, decay_rate: f32, step_size: usize) -> Self {
        LearningRateScheduler::StepDecay {
            initial_lr,
            decay_rate,
            step_size,
        }
    }
    
    /// Create a cosine annealing scheduler
    pub fn cosine_annealing(max_lr: f32, min_lr: f32, period: usize) -> Self {
        LearningRateScheduler::CosineAnnealing {
            max_lr,
            min_lr,
            period,
        }
    }
}