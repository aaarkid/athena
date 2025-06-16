use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use super::traits::Layer as LayerTrait;
use super::traits::Layer;

/// GRU (Gated Recurrent Unit) layer for sequence processing
/// 
/// The GRU layer is a simplified version of LSTM with fewer parameters,
/// combining the forget and input gates into a single update gate.
#[derive(Clone)]
pub struct GRULayer {
    /// Input size
    pub input_size: usize,
    /// Hidden size (number of GRU units)
    pub hidden_size: usize,
    /// Whether to return sequences (all time steps) or just the last output
    pub return_sequences: bool,
    
    // Weight matrices for reset gate
    pub w_ir: Array2<f32>, // Input to reset gate
    pub w_hr: Array2<f32>, // Hidden to reset gate
    pub b_r: Array1<f32>,  // Reset gate bias
    
    // Weight matrices for update gate
    pub w_iz: Array2<f32>, // Input to update gate
    pub w_hz: Array2<f32>, // Hidden to update gate
    pub b_z: Array1<f32>,  // Update gate bias
    
    // Weight matrices for new gate (candidate hidden state)
    pub w_in: Array2<f32>, // Input to new gate
    pub w_hn: Array2<f32>, // Hidden to new gate
    pub b_n: Array1<f32>,  // New gate bias
    
    // Hidden state
    hidden_state: Option<Array2<f32>>,
    
    // Cache for backward pass
    cache: Option<GRUCache>,
}

#[derive(Clone)]
struct GRUCache {
    inputs: Array3<f32>,
    hidden_states: Vec<Array2<f32>>,
    reset_gates: Vec<Array2<f32>>,
    update_gates: Vec<Array2<f32>>,
    new_gates: Vec<Array2<f32>>,
}

impl GRULayer {
    /// Create a new GRU layer
    pub fn new(input_size: usize, hidden_size: usize, return_sequences: bool) -> Self {
        let scale = (1.0 / (input_size + hidden_size) as f32).sqrt();
        
        Self {
            input_size,
            hidden_size,
            return_sequences,
            
            // Initialize weights with Xavier/Glorot initialization
            w_ir: Array2::random((input_size, hidden_size), Uniform::new(-scale, scale)),
            w_hr: Array2::random((hidden_size, hidden_size), Uniform::new(-scale, scale)),
            b_r: Array1::zeros(hidden_size),
            
            w_iz: Array2::random((input_size, hidden_size), Uniform::new(-scale, scale)),
            w_hz: Array2::random((hidden_size, hidden_size), Uniform::new(-scale, scale)),
            b_z: Array1::zeros(hidden_size),
            
            w_in: Array2::random((input_size, hidden_size), Uniform::new(-scale, scale)),
            w_hn: Array2::random((hidden_size, hidden_size), Uniform::new(-scale, scale)),
            b_n: Array1::zeros(hidden_size),
            
            hidden_state: None,
            cache: None,
        }
    }
    
    /// Reset the hidden state
    pub fn reset_state(&mut self) {
        self.hidden_state = None;
    }
    
    /// Set the initial hidden state
    pub fn set_initial_hidden_state(&mut self, state: Array2<f32>) {
        assert_eq!(state.shape()[1], self.hidden_size, "Hidden state size mismatch");
        self.hidden_state = Some(state);
    }
    
    /// Forward pass for a sequence
    /// Input shape: (batch_size, sequence_length, input_size)
    /// Output shape: 
    ///   - If return_sequences: (batch_size, sequence_length, hidden_size)
    ///   - Else: (batch_size, hidden_size)
    pub fn forward_sequence(&mut self, input: ArrayView3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = input.dim();
        
        // Initialize state if not set
        let mut h_t = self.hidden_state.clone()
            .unwrap_or_else(|| Array2::zeros((batch_size, self.hidden_size)));
        
        let mut outputs = Vec::with_capacity(seq_len);
        let mut hidden_states = Vec::with_capacity(seq_len + 1);
        let mut reset_gates = Vec::with_capacity(seq_len);
        let mut update_gates = Vec::with_capacity(seq_len);
        let mut new_gates = Vec::with_capacity(seq_len);
        
        hidden_states.push(h_t.clone());
        
        // Process each time step
        for t in 0..seq_len {
            let x_t = input.slice(s![.., t, ..]);
            
            // Reset gate: r_t = sigmoid(W_ir @ x_t + W_hr @ h_{t-1} + b_r)
            let r_t = Self::sigmoid(&(x_t.dot(&self.w_ir) + h_t.dot(&self.w_hr) + &self.b_r));
            
            // Update gate: z_t = sigmoid(W_iz @ x_t + W_hz @ h_{t-1} + b_z)
            let z_t = Self::sigmoid(&(x_t.dot(&self.w_iz) + h_t.dot(&self.w_hz) + &self.b_z));
            
            // New gate: n_t = tanh(W_in @ x_t + W_hn @ (r_t * h_{t-1}) + b_n)
            let n_t = Self::tanh(&(x_t.dot(&self.w_in) + (&r_t * &h_t).dot(&self.w_hn) + &self.b_n));
            
            // Update hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
            h_t = &(1.0 - &z_t) * &n_t + &z_t * &h_t;
            
            outputs.push(h_t.clone());
            hidden_states.push(h_t.clone());
            reset_gates.push(r_t);
            update_gates.push(z_t);
            new_gates.push(n_t);
        }
        
        // Store final state
        self.hidden_state = Some(h_t);
        
        // Store cache for backward pass
        self.cache = Some(GRUCache {
            inputs: input.to_owned(),
            hidden_states,
            reset_gates,
            update_gates,
            new_gates,
        });
        
        // Stack outputs
        let output_array = Array3::from_shape_fn((batch_size, seq_len, self.hidden_size), |(b, t, h)| {
            outputs[t][[b, h]]
        });
        
        if self.return_sequences {
            output_array
        } else {
            // Return only the last output
            let last_output = output_array.slice(s![.., -1, ..]).to_owned();
            last_output.insert_axis(Axis(1))
        }
    }
    
    /// Backward pass for GRU
    pub fn backward_sequence(&self, output_grad: ArrayView3<f32>) -> GRUGradients {
        let cache = self.cache.as_ref().expect("Forward pass must be called before backward");
        let (batch_size, seq_len, _) = cache.inputs.dim();
        
        // Initialize gradients
        let mut dw_ir = Array2::zeros((self.input_size, self.hidden_size));
        let mut dw_hr = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_r = Array1::zeros(self.hidden_size);
        
        let mut dw_iz = Array2::zeros((self.input_size, self.hidden_size));
        let mut dw_hz = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_z = Array1::zeros(self.hidden_size);
        
        let mut dw_in = Array2::zeros((self.input_size, self.hidden_size));
        let mut dw_hn = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_n = Array1::zeros(self.hidden_size);
        
        let mut dx = Array3::zeros((batch_size, seq_len, self.input_size));
        let mut dh_next = Array2::zeros((batch_size, self.hidden_size));
        
        // Process in reverse order
        for t in (0..seq_len).rev() {
            let grad_t = if self.return_sequences {
                output_grad.slice(s![.., t, ..])
            } else if t == seq_len - 1 {
                output_grad.slice(s![.., 0, ..])
            } else {
                continue;
            };
            
            let dh = &grad_t.to_owned() + &dh_next;
            let x_t = cache.inputs.slice(s![.., t, ..]);
            let h_prev = &cache.hidden_states[t];
            
            let r_t = &cache.reset_gates[t];
            let z_t = &cache.update_gates[t];
            let n_t = &cache.new_gates[t];
            
            // Gradients of hidden state update
            let dn_t = &dh * &(1.0 - z_t);
            let dz_t = &dh * &(h_prev - n_t);
            let dh_prev = &dh * z_t;
            
            // Gate derivatives
            let dn_gate = dn_t * &Self::tanh_derivative(n_t);
            let dz_gate = dz_t * &Self::sigmoid_derivative(z_t);
            let dr_gate_pre = dn_gate.dot(&self.w_hn.t()) * h_prev;
            let dr_gate = dr_gate_pre * &Self::sigmoid_derivative(r_t);
            
            // Weight gradients
            dw_ir = &dw_ir + &x_t.t().dot(&dr_gate);
            dw_hr = &dw_hr + &h_prev.t().dot(&dr_gate);
            db_r = &db_r + &dr_gate.sum_axis(Axis(0));
            
            dw_iz = &dw_iz + &x_t.t().dot(&dz_gate);
            dw_hz = &dw_hz + &h_prev.t().dot(&dz_gate);
            db_z = &db_z + &dz_gate.sum_axis(Axis(0));
            
            dw_in = &dw_in + &x_t.t().dot(&dn_gate);
            dw_hn = &dw_hn + &(r_t * h_prev).t().dot(&dn_gate);
            db_n = &db_n + &dn_gate.sum_axis(Axis(0));
            
            // Input gradient
            let dx_t = dr_gate.dot(&self.w_ir.t()) + 
                       dz_gate.dot(&self.w_iz.t()) + 
                       dn_gate.dot(&self.w_in.t());
            
            dx.slice_mut(s![.., t, ..]).assign(&dx_t);
            
            // Hidden state gradient for next iteration
            dh_next = dh_prev + 
                      dr_gate.dot(&self.w_hr.t()) + 
                      dz_gate.dot(&self.w_hz.t()) + 
                      (dn_gate.dot(&self.w_hn.t()) * r_t);
        }
        
        GRUGradients {
            dw_ir, dw_hr, db_r,
            dw_iz, dw_hz, db_z,
            dw_in, dw_hn, db_n,
            dx,
        }
    }
    
    // Activation functions
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }
    
    fn sigmoid_derivative(s: &Array2<f32>) -> Array2<f32> {
        s * &(1.0 - s)
    }
    
    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
    }
    
    fn tanh_derivative(t: &Array2<f32>) -> Array2<f32> {
        1.0 - t * t
    }
}

/// Gradients for GRU layer
pub struct GRUGradients {
    pub dw_ir: Array2<f32>, pub dw_hr: Array2<f32>, pub db_r: Array1<f32>,
    pub dw_iz: Array2<f32>, pub dw_hz: Array2<f32>, pub db_z: Array1<f32>,
    pub dw_in: Array2<f32>, pub dw_hn: Array2<f32>, pub db_n: Array1<f32>,
    pub dx: Array3<f32>,
}

// Implement Layer trait for compatibility
impl LayerTrait for GRULayer {
    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.w_ir // Return one of the weight matrices
    }
    
    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.b_r // Return one of the bias vectors
    }
    
    fn weights(&self) -> &Array2<f32> {
        &self.w_ir // Return one of the weight matrices
    }
    
    fn biases(&self) -> &Array1<f32> {
        &self.b_r // Return one of the bias vectors
    }
    
    fn output_size(&self) -> usize {
        self.hidden_size
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        // Convert 1D input to 3D (batch_size=1, seq_len=1, input_size)
        let input_3d = input.to_owned()
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));
        
        let output = self.forward_sequence(input_3d.view());
        
        // Extract the output
        if self.return_sequences {
            output.slice(s![0, 0, ..]).to_owned()
        } else {
            output.slice(s![0, 0, ..]).to_owned()
        }
    }
    
    fn backward(&self, _output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        // For compatibility, return dummy gradients
        // Real GRU backward pass should use backward_sequence
        let dummy_weights = Array2::zeros((self.input_size, self.hidden_size));
        let dummy_bias = Array1::zeros(self.hidden_size);
        (dummy_weights, dummy_bias)
    }
    
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        // Convert 2D input to 3D (batch_size, seq_len=1, input_size)
        let batch_size = inputs.shape()[0];
        let input_3d = inputs.to_owned()
            .into_shape((batch_size, 1, self.input_size))
            .expect("Failed to reshape");
        
        let output = self.forward_sequence(input_3d.view());
        
        // Extract the output
        output.slice(s![.., 0, ..]).to_owned()
    }
    
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        // For compatibility, return dummy gradients
        let batch_size = output_errors.shape()[0];
        let dummy_output = Array2::zeros((batch_size, self.input_size));
        let dummy_weights = Array2::zeros((self.input_size, self.hidden_size));
        let dummy_bias = Array1::zeros(self.hidden_size);
        (dummy_output, dummy_weights, dummy_bias)
    }
}

// Re-export for cleaner imports
use ndarray::s;