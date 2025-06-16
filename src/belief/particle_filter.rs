use ndarray::Array1;
use rand::prelude::*;
use std::fmt::Debug;

/// Particle filter for belief state representation
pub struct ParticleFilter<S: Clone> {
    particles: Vec<S>,
    weights: Array1<f32>,
    transition_fn: Box<dyn Fn(&S, usize) -> S + Send + Sync>,
    observation_fn: Box<dyn Fn(&S) -> Array1<f32> + Send + Sync>,
    resampling_threshold: f32,
}

impl<S: Clone + Send + Sync> ParticleFilter<S> {
    pub fn new(
        num_particles: usize,
        initial_state_fn: impl Fn() -> S,
        transition_fn: impl Fn(&S, usize) -> S + 'static + Send + Sync,
        observation_fn: impl Fn(&S) -> Array1<f32> + 'static + Send + Sync,
    ) -> Self {
        let particles = (0..num_particles)
            .map(|_| initial_state_fn())
            .collect();
        let weights = Array1::ones(num_particles) / num_particles as f32;
        
        Self {
            particles,
            weights,
            transition_fn: Box::new(transition_fn),
            observation_fn: Box::new(observation_fn),
            resampling_threshold: 0.5,
        }
    }
    
    /// Update particles based on action and observation
    pub fn update_particles(&mut self, action: usize, observation: &Array1<f32>) {
        // Transition particles
        for particle in &mut self.particles {
            *particle = (self.transition_fn)(particle, action);
        }
        
        // Update weights based on observation likelihood
        for (i, particle) in self.particles.iter().enumerate() {
            let predicted_obs = (self.observation_fn)(particle);
            self.weights[i] *= observation_likelihood(&predicted_obs, observation);
        }
        
        // Normalize weights
        let sum = self.weights.sum();
        if sum > 0.0 {
            self.weights /= sum;
        } else {
            // Reset to uniform if all weights are zero
            self.weights.fill(1.0 / self.particles.len() as f32);
        }
        
        // Resample if effective sample size is low
        if self.effective_sample_size() < self.resampling_threshold * self.particles.len() as f32 {
            self.resample();
        }
    }
    
    /// Calculate effective sample size
    pub fn effective_sample_size(&self) -> f32 {
        let sum_squared = self.weights.mapv(|w| w * w).sum();
        if sum_squared > 0.0 {
            1.0 / sum_squared
        } else {
            0.0
        }
    }
    
    /// Systematic resampling
    fn resample(&mut self) {
        let n = self.particles.len();
        let mut new_particles = Vec::with_capacity(n);
        let cumsum = cumulative_sum(&self.weights);
        
        let step = 1.0 / n as f32;
        let mut rng = thread_rng();
        let mut u = rng.gen::<f32>() * step;
        
        for _ in 0..n {
            let idx = cumsum.iter().position(|&w| w > u).unwrap_or(n - 1);
            new_particles.push(self.particles[idx].clone());
            u += step;
            if u > 1.0 {
                u -= 1.0;
            }
        }
        
        self.particles = new_particles;
        self.weights.fill(1.0 / n as f32);
    }
    
    /// Get weighted mean of particle states (if S can be averaged)
    pub fn mean_state<F>(&self, state_to_vector: F) -> Array1<f32>
    where
        F: Fn(&S) -> Array1<f32>,
    {
        let mut mean = Array1::zeros(state_to_vector(&self.particles[0]).len());
        
        for (particle, &weight) in self.particles.iter().zip(self.weights.iter()) {
            let state_vec = state_to_vector(particle);
            mean = mean + state_vec * weight;
        }
        
        mean
    }
    
    /// Sample a particle according to weights
    pub fn sample_particle(&self) -> &S {
        let mut rng = thread_rng();
        let u = rng.gen::<f32>();
        let cumsum = cumulative_sum(&self.weights);
        
        let idx = cumsum.iter().position(|&w| w > u).unwrap_or(self.particles.len() - 1);
        &self.particles[idx]
    }
    
    /// Get entropy of weight distribution
    pub fn entropy(&self) -> f32 {
        -self.weights.iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| w * w.ln())
            .sum::<f32>()
    }
}

/// Calculate observation likelihood
fn observation_likelihood(predicted: &Array1<f32>, actual: &Array1<f32>) -> f32 {
    // Gaussian likelihood
    let diff = predicted - actual;
    let squared_error = diff.mapv(|x| x * x).sum();
    (-0.5 * squared_error).exp()
}

/// Calculate cumulative sum
fn cumulative_sum(weights: &Array1<f32>) -> Vec<f32> {
    let mut cumsum = Vec::with_capacity(weights.len());
    let mut sum = 0.0;
    
    for &w in weights.iter() {
        sum += w;
        cumsum.push(sum);
    }
    
    cumsum
}

impl<S: Clone + Debug> ParticleFilter<S> {
    /// Debug print particle states and weights
    pub fn debug_particles(&self) {
        for (i, (particle, &weight)) in self.particles.iter().zip(self.weights.iter()).enumerate() {
            println!("Particle {}: {:?}, weight: {:.4}", i, particle, weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[derive(Clone, Debug)]
    struct SimpleState {
        position: f32,
        velocity: f32,
    }
    
    #[test]
    fn test_particle_filter() {
        let initial_state = || SimpleState { position: 0.0, velocity: 0.0 };
        
        let transition = |state: &SimpleState, action: usize| {
            let accel = if action == 0 { -1.0 } else { 1.0 };
            SimpleState {
                position: state.position + state.velocity * 0.1,
                velocity: state.velocity + accel * 0.1,
            }
        };
        
        let observe = |state: &SimpleState| array![state.position, state.velocity];
        
        let mut pf = ParticleFilter::new(100, initial_state, transition, observe);
        
        // Update with observation
        let obs = array![0.1, 0.1];
        pf.update_particles(1, &obs);
        
        // Check that weights sum to 1
        assert!((pf.weights.sum() - 1.0).abs() < 1e-6);
        
        // Check effective sample size
        let ess = pf.effective_sample_size();
        assert!(ess > 0.0 && ess <= 100.0);
    }
    
    #[test]
    fn test_resampling() {
        let initial_state = || 0.0f32;
        let transition = |&state: &f32, _: usize| state + 1.0;
        let observe = |&state: &f32| array![state];
        
        let mut pf = ParticleFilter::new(10, initial_state, transition, observe);
        
        // Set non-uniform weights
        pf.weights[0] = 0.9;
        pf.weights[1] = 0.1;
        for i in 2..10 {
            pf.weights[i] = 0.0;
        }
        
        pf.resample();
        
        // After resampling, weights should be uniform
        for &w in pf.weights.iter() {
            assert!((w - 0.1).abs() < 1e-6);
        }
    }
}