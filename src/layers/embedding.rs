use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use super::traits::Layer as LayerTrait;
use super::traits::Layer;

/// Embedding layer for converting discrete tokens to continuous vectors
/// 
/// This layer is commonly used in NLP tasks to convert word indices or token IDs
/// into dense vector representations that can be learned during training.
#[derive(Clone)]
pub struct EmbeddingLayer {
    /// Number of unique tokens/words in vocabulary
    pub vocab_size: usize,
    /// Dimension of embedding vectors
    pub embedding_dim: usize,
    /// The embedding matrix (vocab_size x embedding_dim)
    pub embeddings: Array2<f32>,
    /// Cache for backward pass
    last_indices: Option<Vec<usize>>,
}

impl EmbeddingLayer {
    /// Create a new embedding layer
    /// 
    /// # Arguments
    /// * `vocab_size` - Number of unique tokens in vocabulary
    /// * `embedding_dim` - Dimension of embedding vectors
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Initialize embeddings with small random values
        let scale = (1.0 / embedding_dim as f32).sqrt();
        let embeddings = Array2::random((vocab_size, embedding_dim), Uniform::new(-scale, scale));
        
        Self {
            vocab_size,
            embedding_dim,
            embeddings,
            last_indices: None,
        }
    }
    
    /// Create embedding layer with pre-trained embeddings
    /// 
    /// # Arguments
    /// * `embeddings` - Pre-trained embedding matrix
    pub fn from_pretrained(embeddings: Array2<f32>) -> Self {
        let (vocab_size, embedding_dim) = embeddings.dim();
        
        Self {
            vocab_size,
            embedding_dim,
            embeddings,
            last_indices: None,
        }
    }
    
    /// Forward pass for single token
    /// 
    /// # Arguments
    /// * `index` - Token index
    /// 
    /// # Returns
    /// Embedding vector for the token
    pub fn forward_single(&mut self, index: usize) -> Array1<f32> {
        assert!(index < self.vocab_size, "Token index {} out of vocabulary range", index);
        self.last_indices = Some(vec![index]);
        self.embeddings.row(index).to_owned()
    }
    
    /// Forward pass for sequence of tokens
    /// 
    /// # Arguments
    /// * `indices` - Array of token indices
    /// 
    /// # Returns
    /// Matrix where each row is an embedding vector
    pub fn forward_sequence(&mut self, indices: &[usize]) -> Array2<f32> {
        let batch_size = indices.len();
        let mut output = Array2::zeros((batch_size, self.embedding_dim));
        
        for (i, &index) in indices.iter().enumerate() {
            assert!(index < self.vocab_size, "Token index {} out of vocabulary range", index);
            output.row_mut(i).assign(&self.embeddings.row(index));
        }
        
        self.last_indices = Some(indices.to_vec());
        output
    }
    
    /// Backward pass
    /// 
    /// # Arguments
    /// * `grad_output` - Gradient with respect to output
    /// 
    /// # Returns
    /// Gradient with respect to embeddings
    pub fn backward_embeddings(&self, grad_output: ArrayView2<f32>) -> Array2<f32> {
        let indices = self.last_indices.as_ref()
            .expect("Forward pass must be called before backward");
        
        let mut grad_embeddings = Array2::zeros((self.vocab_size, self.embedding_dim));
        
        for (i, &index) in indices.iter().enumerate() {
            grad_embeddings.row_mut(index).scaled_add(1.0, &grad_output.row(i));
        }
        
        grad_embeddings
    }
    
    /// Update embeddings with gradients
    /// 
    /// # Arguments
    /// * `grad_embeddings` - Gradients for embedding matrix
    /// * `learning_rate` - Learning rate for update
    pub fn update(&mut self, grad_embeddings: &Array2<f32>, learning_rate: f32) {
        self.embeddings.scaled_add(-learning_rate, grad_embeddings);
    }
    
    /// Get embedding for a specific token
    pub fn get_embedding(&self, index: usize) -> ArrayView1<f32> {
        assert!(index < self.vocab_size, "Token index {} out of vocabulary range", index);
        self.embeddings.row(index)
    }
    
    /// Set embedding for a specific token
    pub fn set_embedding(&mut self, index: usize, embedding: ArrayView1<f32>) {
        assert!(index < self.vocab_size, "Token index {} out of vocabulary range", index);
        assert_eq!(embedding.len(), self.embedding_dim, "Embedding dimension mismatch");
        self.embeddings.row_mut(index).assign(&embedding);
    }
    
    /// Find nearest neighbors to a given embedding
    /// 
    /// # Arguments
    /// * `embedding` - Query embedding vector
    /// * `k` - Number of nearest neighbors to return
    /// 
    /// # Returns
    /// Vector of (index, similarity) pairs sorted by similarity
    pub fn nearest_neighbors(&self, embedding: ArrayView1<f32>, k: usize) -> Vec<(usize, f32)> {
        assert_eq!(embedding.len(), self.embedding_dim, "Embedding dimension mismatch");
        
        let mut similarities: Vec<(usize, f32)> = (0..self.vocab_size)
            .map(|i| {
                let similarity = self.cosine_similarity(embedding, self.embeddings.row(i));
                (i, similarity)
            })
            .collect();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }
    
    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
        let dot_product = a.dot(&b);
        let norm_a = a.dot(&a).sqrt();
        let norm_b = b.dot(&b).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

// Implement Layer trait for compatibility
impl LayerTrait for EmbeddingLayer {
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        // Interpret input as token index
        let index = input[0] as usize;
        self.forward_single(index)
    }
    
    fn backward(&self, _output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        // For compatibility, return dummy gradients
        let dummy_weights = Array2::zeros((1, self.embedding_dim));
        let dummy_bias = Array1::zeros(self.embedding_dim);
        (dummy_weights, dummy_bias)
    }
    
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        // Interpret each row as containing a token index
        let batch_size = inputs.shape()[0];
        let indices: Vec<usize> = (0..batch_size)
            .map(|i| inputs[[i, 0]] as usize)
            .collect();
        
        self.forward_sequence(&indices)
    }
    
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let grad_embeddings = self.backward_embeddings(output_errors);
        
        // For compatibility, return dummy outputs
        let batch_size = output_errors.shape()[0];
        let dummy_output = Array2::zeros((batch_size, 1));
        let dummy_bias = Array1::zeros(self.embedding_dim);
        (dummy_output, grad_embeddings, dummy_bias)
    }
    
    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.embeddings
    }
    
    fn biases_mut(&mut self) -> &mut Array1<f32> {
        // Embeddings don't have biases, return a dummy mutable reference
        static mut DUMMY_BIAS: Option<Array1<f32>> = None;
        unsafe {
            if DUMMY_BIAS.is_none() {
                DUMMY_BIAS = Some(Array1::zeros(self.embedding_dim));
            }
            DUMMY_BIAS.as_mut().unwrap()
        }
    }
    
    fn weights(&self) -> &Array2<f32> {
        &self.embeddings
    }
    
    fn biases(&self) -> &Array1<f32> {
        // Embeddings don't have biases, return a dummy reference
        static mut DUMMY_BIAS: Option<Array1<f32>> = None;
        unsafe {
            if DUMMY_BIAS.is_none() {
                DUMMY_BIAS = Some(Array1::zeros(self.embedding_dim));
            }
            DUMMY_BIAS.as_ref().unwrap()
        }
    }
    
    fn output_size(&self) -> usize {
        self.embedding_dim
    }
    
    fn input_size(&self) -> usize {
        1 // Single token index
    }
    
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}