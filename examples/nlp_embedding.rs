use athena::layers::{EmbeddingLayer, Layer};
use athena::network::NeuralNetwork;
use athena::activations::Activation;
use athena::optimizer::{OptimizerWrapper, Adam};
use ndarray::Array1;
use ndarray_rand::RandomExt;

fn main() {
    // Simple example: Sentiment classification with embeddings
    
    // Vocabulary setup
    let vocab_size = 1000;  // Small vocabulary for example
    let embedding_dim = 50;
    let _max_sequence_length = 10;
    
    // Create embedding layer
    let mut embedding_layer = EmbeddingLayer::new(vocab_size, embedding_dim);
    
    // Example: Single word embedding
    let word_index = 42;
    let word_embedding = embedding_layer.forward_single(word_index);
    println!("Embedding for word {}: shape {:?}", word_index, word_embedding.shape());
    
    // Example: Sequence embedding
    let sentence_indices = vec![1, 5, 10, 20, 42, 100];
    let sentence_embeddings = embedding_layer.forward_sequence(&sentence_indices);
    println!("Embeddings for sentence: shape {:?}", sentence_embeddings.shape());
    
    // Example: Finding similar words
    let query_embedding = embedding_layer.get_embedding(word_index);
    let neighbors = embedding_layer.nearest_neighbors(query_embedding, 5);
    println!("\nNearest neighbors to word {}:", word_index);
    for (idx, similarity) in neighbors {
        println!("  Word {}: similarity {:.3}", idx, similarity);
    }
    
    // Example: Building a simple sentiment classifier
    println!("\n=== Building Sentiment Classifier ===");
    
    // Network architecture:
    // Embedding (1000 -> 50) -> Dense (50 -> 25) -> Dense (25 -> 1)
    let layers = vec![Layer::new(embedding_dim, 25, Activation::Relu)];
    let optimizer = OptimizerWrapper::Adam(Adam::new(&layers, 0.9, 0.999, 1e-8));
    let mut network = NeuralNetwork::new(
        &[embedding_dim, 25, 1],
        &[Activation::Relu, Activation::Sigmoid],
        optimizer
    );
    
    // Example training data (simplified)
    let positive_sentences = vec![
        vec![10, 20, 30],  // "I love this"
        vec![40, 50, 60],  // "Great product"
    ];
    
    let negative_sentences = vec![
        vec![70, 80, 90],   // "Not good"
        vec![100, 110, 120], // "Terrible quality"
    ];
    
    // Training loop (simplified)
    println!("\nTraining sentiment classifier...");
    for epoch in 0..5 {
        let mut total_loss = 0.0;
        
        // Train on positive examples
        for sentence in &positive_sentences {
            // Get embeddings for sentence
            let embeddings = embedding_layer.forward_sequence(sentence);
            // Average pooling over sequence
            let avg_embedding = embeddings.mean_axis(ndarray::Axis(0)).unwrap();
            
            // Forward through classifier
            let prediction = network.forward(avg_embedding.view());
            let target = Array1::from_elem(1, 1.0); // Positive = 1
            let loss = ((&prediction - &target) * (&prediction - &target)).sum();
            total_loss += loss;
        }
        
        // Train on negative examples
        for sentence in &negative_sentences {
            // Get embeddings for sentence
            let embeddings = embedding_layer.forward_sequence(sentence);
            // Average pooling over sequence
            let avg_embedding = embeddings.mean_axis(ndarray::Axis(0)).unwrap();
            
            // Forward through classifier
            let prediction = network.forward(avg_embedding.view());
            let target = Array1::from_elem(1, 0.0); // Negative = 0
            let loss = ((&prediction - &target) * (&prediction - &target)).sum();
            total_loss += loss;
        }
        
        println!("Epoch {}: Loss = {:.4}", epoch + 1, total_loss);
    }
    
    // Test the classifier
    println!("\n=== Testing Classifier ===");
    let test_sentence = vec![10, 50, 30]; // Mix of positive words
    let embeddings = embedding_layer.forward_sequence(&test_sentence);
    let avg_embedding = embeddings.mean_axis(ndarray::Axis(0)).unwrap();
    let prediction = network.forward(avg_embedding.view());
    
    println!("Test sentence indices: {:?}", test_sentence);
    println!("Prediction: {:.3} (>0.5 = positive, <0.5 = negative)", prediction[0]);
    
    // Example: Using pre-trained embeddings
    println!("\n=== Using Pre-trained Embeddings ===");
    let pretrained = ndarray::Array2::random((100, 25), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
    let pretrained_layer = EmbeddingLayer::from_pretrained(pretrained);
    println!("Loaded pre-trained embeddings: {} words, {} dimensions", 
             pretrained_layer.vocab_size, pretrained_layer.embedding_dim);
}