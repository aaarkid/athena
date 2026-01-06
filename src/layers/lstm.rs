use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use super::traits::Layer as LayerTrait;
use super::traits::Layer;

/// LSTM (Long Short-Term Memory) layer for sequence processing
/// 
/// The LSTM layer maintains a cell state and hidden state across time steps,
/// allowing it to learn long-term dependencies in sequential data.
#[derive(Clone)]
pub struct LSTMLayer {
    /// Input size
    pub input_size: usize,
    /// Hidden size (number of LSTM units)
    pub hidden_size: usize,
    /// Whether to return sequences (all time steps) or just the last output
    pub return_sequences: bool,
    
    // Weight matrices for input gate
    pub w_ii: Array2<f32>, // Input to input gate
    pub w_hi: Array2<f32>, // Hidden to input gate
    pub b_i: Array1<f32>,  // Input gate bias
    
    // Weight matrices for forget gate
    pub w_if: Array2<f32>, // Input to forget gate
    pub w_hf: Array2<f32>, // Hidden to forget gate
    pub b_f: Array1<f32>,  // Forget gate bias
    
    // Weight matrices for cell gate (candidate values)
    pub w_ig: Array2<f32>, // Input to cell gate
    pub w_hg: Array2<f32>, // Hidden to cell gate
    pub b_g: Array1<f32>,  // Cell gate bias
    
    // Weight matrices for output gate
    pub w_io: Array2<f32>, // Input to output gate
    pub w_ho: Array2<f32>, // Hidden to output gate
    pub b_o: Array1<f32>,  // Output gate bias
    
    // Hidden and cell states
    hidden_state: Option<Array2<f32>>,
    cell_state: Option<Array2<f32>>,
    
    // Cache for backward pass
    cache: Option<LSTMCache>,
}

#[derive(Clone)]
struct LSTMCache {
    inputs: Array3<f32>,
    hidden_states: Vec<Array2<f32>>,
    cell_states: Vec<Array2<f32>>,
    input_gates: Vec<Array2<f32>>,
    forget_gates: Vec<Array2<f32>>,
    cell_gates: Vec<Array2<f32>>,
    output_gates: Vec<Array2<f32>>,
}

impl LSTMLayer {
    /// Create a new LSTM layer
    pub fn new(input_size: usize, hidden_size: usize, return_sequences: bool) -> Self {
        let scale = (1.0 / (input_size + hidden_size) as f32).sqrt();
        
        Self {
            input_size,
            hidden_size,
            return_sequences,
            
            // Initialize weights with Xavier/Glorot initialization
            w_ii: Array2::random((input_size, hidden_size), Uniform::new(-scale, scale)),
            w_hi: Array2::random((hidden_size, hidden_size), Uniform::new(-scale, scale)),
            b_i: Array1::zeros(hidden_size),
            
            w_if: Array2::random((input_size, hidden_size), Uniform::new(-scale, scale)),
            w_hf: Array2::random((hidden_size, hidden_size), Uniform::new(-scale, scale)),
            b_f: Array1::ones(hidden_size), // Initialize forget gate bias to 1 for better gradient flow
            
            w_ig: Array2::random((input_size, hidden_size), Uniform::new(-scale, scale)),
            w_hg: Array2::random((hidden_size, hidden_size), Uniform::new(-scale, scale)),
            b_g: Array1::zeros(hidden_size),
            
            w_io: Array2::random((input_size, hidden_size), Uniform::new(-scale, scale)),
            w_ho: Array2::random((hidden_size, hidden_size), Uniform::new(-scale, scale)),
            b_o: Array1::zeros(hidden_size),
            
            hidden_state: None,
            cell_state: None,
            cache: None,
        }
    }
    
    /// Reset the hidden and cell states
    pub fn reset_states(&mut self) {
        self.hidden_state = None;
        self.cell_state = None;
    }
    
    /// Set the initial hidden state
    pub fn set_initial_hidden_state(&mut self, state: Array2<f32>) {
        assert_eq!(state.shape()[1], self.hidden_size, "Hidden state size mismatch");
        self.hidden_state = Some(state);
    }
    
    /// Set the initial cell state
    pub fn set_initial_cell_state(&mut self, state: Array2<f32>) {
        assert_eq!(state.shape()[1], self.hidden_size, "Cell state size mismatch");
        self.cell_state = Some(state);
    }
    
    /// Forward pass for a sequence
    /// Input shape: (batch_size, sequence_length, input_size)
    /// Output shape: 
    ///   - If return_sequences: (batch_size, sequence_length, hidden_size)
    ///   - Else: (batch_size, hidden_size)
    pub fn forward_sequence(&mut self, input: ArrayView3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = input.dim();
        
        // Initialize states if not set
        let mut h_t = self.hidden_state.clone()
            .unwrap_or_else(|| Array2::zeros((batch_size, self.hidden_size)));
        let mut c_t = self.cell_state.clone()
            .unwrap_or_else(|| Array2::zeros((batch_size, self.hidden_size)));
        
        let mut outputs = Vec::with_capacity(seq_len);
        let mut hidden_states = Vec::with_capacity(seq_len + 1);
        let mut cell_states = Vec::with_capacity(seq_len + 1);
        let mut input_gates = Vec::with_capacity(seq_len);
        let mut forget_gates = Vec::with_capacity(seq_len);
        let mut cell_gates = Vec::with_capacity(seq_len);
        let mut output_gates = Vec::with_capacity(seq_len);
        
        hidden_states.push(h_t.clone());
        cell_states.push(c_t.clone());
        
        // Process each time step
        for t in 0..seq_len {
            let x_t = input.slice(s![.., t, ..]);
            
            // Input gate: i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
            let i_t = Self::sigmoid(&(x_t.dot(&self.w_ii) + h_t.dot(&self.w_hi) + &self.b_i));
            
            // Forget gate: f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)
            let f_t = Self::sigmoid(&(x_t.dot(&self.w_if) + h_t.dot(&self.w_hf) + &self.b_f));
            
            // Cell gate: g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
            let g_t = Self::tanh(&(x_t.dot(&self.w_ig) + h_t.dot(&self.w_hg) + &self.b_g));
            
            // Output gate: o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)
            let o_t = Self::sigmoid(&(x_t.dot(&self.w_io) + h_t.dot(&self.w_ho) + &self.b_o));
            
            // Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
            c_t = &f_t * &c_t + &i_t * &g_t;
            
            // Update hidden state: h_t = o_t * tanh(c_t)
            h_t = &o_t * &Self::tanh(&c_t);
            
            outputs.push(h_t.clone());
            hidden_states.push(h_t.clone());
            cell_states.push(c_t.clone());
            input_gates.push(i_t);
            forget_gates.push(f_t);
            cell_gates.push(g_t);
            output_gates.push(o_t);
        }
        
        // Store final states
        self.hidden_state = Some(h_t);
        self.cell_state = Some(c_t);
        
        // Store cache for backward pass
        self.cache = Some(LSTMCache {
            inputs: input.to_owned(),
            hidden_states,
            cell_states,
            input_gates,
            forget_gates,
            cell_gates,
            output_gates,
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
    
    /// Backward pass for LSTM
    pub fn backward_sequence(&self, output_grad: ArrayView3<f32>) -> LSTMGradients {
        let cache = self.cache.as_ref().expect("Forward pass must be called before backward");
        let (batch_size, seq_len, _) = cache.inputs.dim();
        
        // Initialize gradients
        let mut dw_ii = Array2::zeros((self.input_size, self.hidden_size));
        let mut dw_hi = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_i = Array1::zeros(self.hidden_size);
        
        let mut dw_if = Array2::zeros((self.input_size, self.hidden_size));
        let mut dw_hf = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_f = Array1::zeros(self.hidden_size);
        
        let mut dw_ig = Array2::zeros((self.input_size, self.hidden_size));
        let mut dw_hg = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_g = Array1::zeros(self.hidden_size);
        
        let mut dw_io = Array2::zeros((self.input_size, self.hidden_size));
        let mut dw_ho = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_o = Array1::zeros(self.hidden_size);
        
        let mut dx = Array3::zeros((batch_size, seq_len, self.input_size));
        let mut dh_next = Array2::zeros((batch_size, self.hidden_size));
        let mut dc_next = Array2::zeros((batch_size, self.hidden_size));
        
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
            let c_t = &cache.cell_states[t + 1];
            let c_prev = &cache.cell_states[t];
            
            let i_t = &cache.input_gates[t];
            let f_t = &cache.forget_gates[t];
            let g_t = &cache.cell_gates[t];
            let o_t = &cache.output_gates[t];
            
            // Gradients of hidden state
            let tanh_c_t = Self::tanh(c_t);
            let do_t = &dh * &tanh_c_t;
            let dc = &dh * o_t * &Self::tanh_derivative(&tanh_c_t) + &dc_next;
            
            // Gradients of gates
            let di_t = &dc * g_t;
            let df_t = &dc * c_prev;
            let dg_t = &dc * i_t;
            dc_next = &dc * f_t;
            
            // Gate derivatives
            let di_gate = di_t * &Self::sigmoid_derivative(i_t);
            let df_gate = df_t * &Self::sigmoid_derivative(f_t);
            let dg_gate = dg_t * &Self::tanh_derivative(g_t);
            let do_gate = do_t * &Self::sigmoid_derivative(o_t);
            
            // Weight gradients
            dw_ii = &dw_ii + &x_t.t().dot(&di_gate);
            dw_hi = &dw_hi + &h_prev.t().dot(&di_gate);
            db_i = &db_i + &di_gate.sum_axis(Axis(0));
            
            dw_if = &dw_if + &x_t.t().dot(&df_gate);
            dw_hf = &dw_hf + &h_prev.t().dot(&df_gate);
            db_f = &db_f + &df_gate.sum_axis(Axis(0));
            
            dw_ig = &dw_ig + &x_t.t().dot(&dg_gate);
            dw_hg = &dw_hg + &h_prev.t().dot(&dg_gate);
            db_g = &db_g + &dg_gate.sum_axis(Axis(0));
            
            dw_io = &dw_io + &x_t.t().dot(&do_gate);
            dw_ho = &dw_ho + &h_prev.t().dot(&do_gate);
            db_o = &db_o + &do_gate.sum_axis(Axis(0));
            
            // Input gradient
            let dx_t = di_gate.dot(&self.w_ii.t()) + 
                       df_gate.dot(&self.w_if.t()) + 
                       dg_gate.dot(&self.w_ig.t()) + 
                       do_gate.dot(&self.w_io.t());
            
            dx.slice_mut(s![.., t, ..]).assign(&dx_t);
            
            // Hidden state gradient for next iteration
            dh_next = di_gate.dot(&self.w_hi.t()) + 
                      df_gate.dot(&self.w_hf.t()) + 
                      dg_gate.dot(&self.w_hg.t()) + 
                      do_gate.dot(&self.w_ho.t());
        }
        
        LSTMGradients {
            dw_ii, dw_hi, db_i,
            dw_if, dw_hf, db_f,
            dw_ig, dw_hg, db_g,
            dw_io, dw_ho, db_o,
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

/// Gradients for LSTM layer
pub struct LSTMGradients {
    pub dw_ii: Array2<f32>, pub dw_hi: Array2<f32>, pub db_i: Array1<f32>,
    pub dw_if: Array2<f32>, pub dw_hf: Array2<f32>, pub db_f: Array1<f32>,
    pub dw_ig: Array2<f32>, pub dw_hg: Array2<f32>, pub db_g: Array1<f32>,
    pub dw_io: Array2<f32>, pub dw_ho: Array2<f32>, pub db_o: Array1<f32>,
    pub dx: Array3<f32>,
}

// Implement Layer trait for compatibility
impl LayerTrait for LSTMLayer {
    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.w_ii // Return one of the weight matrices
    }
    
    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.b_i // Return one of the bias vectors
    }
    
    fn weights(&self) -> &Array2<f32> {
        &self.w_ii // Return one of the weight matrices
    }
    
    fn biases(&self) -> &Array1<f32> {
        &self.b_i // Return one of the bias vectors
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
        // Real LSTM backward pass should use backward_sequence
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