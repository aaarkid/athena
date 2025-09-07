use ndarray::{Array1, Array2, ArrayView2};

/// GELU (Gaussian Error Linear Unit) activation function
pub struct Gelu;

impl Gelu {
    /// Approximation of GELU using tanh
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn apply(input: &mut Array1<f32>) {
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        input.mapv_inplace(|x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + inner.tanh())
        });
    }
    
    pub fn apply_batch(inputs: &mut Array2<f32>) {
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        inputs.mapv_inplace(|x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + inner.tanh())
        });
    }
    
    pub fn derivative(input: &Array1<f32>) -> Array1<f32> {
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        input.mapv(|x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            let tanh_inner = inner.tanh();
            let sech2_inner = 1.0 - tanh_inner.powi(2);
            0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * sqrt_2_over_pi * (1.0 + 0.134145 * x.powi(2))
        })
    }
    
    pub fn derivative_batch(inputs: ArrayView2<f32>) -> Array2<f32> {
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        inputs.mapv(|x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            let tanh_inner = inner.tanh();
            let sech2_inner = 1.0 - tanh_inner.powi(2);
            0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * sqrt_2_over_pi * (1.0 + 0.134145 * x.powi(2))
        })
    }
}