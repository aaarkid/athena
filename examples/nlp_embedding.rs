use athena::layers::{EmbeddingLayer, Layer};
use athena::network::NeuralNetwork;
use athena::activations::Activation;
use athena::optimizer::{OptimizerWrapper, Adam};
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
    
    // Training loop. The embedding table is not part of the NeuralNetwork, so its
    // gradient has to be carried across by hand: the classifier reports the gradient
    // with respect to its own input, average pooling splits that evenly over the
    // tokens, and backward_embeddings scatters it into the table.
    let samples: Vec<(&Vec<usize>, f32)> = positive_sentences
        .iter()
        .map(|s| (s, 1.0))
        .chain(negative_sentences.iter().map(|s| (s, 0.0)))
        .collect();

    println!("\nTraining sentiment classifier...");
    let learning_rate = 0.05;
    for epoch in 0..40 {
        let mut total_loss = 0.0;

        for (sentence, label) in &samples {
            let embeddings = embedding_layer.forward_sequence(sentence);
            let pooled = embeddings.mean_axis(ndarray::Axis(0)).unwrap();
            let input = pooled.insert_axis(ndarray::Axis(0));
            let target = ndarray::Array2::from_elem((1, 1), *label);

            let prediction = network.forward_batch(input.view());
            total_loss += (&prediction - &target).mapv(|e| e * e).sum();

            // Read the input gradient before training: it uses the pre-activations
            // cached by the forward pass above
            let output_errors = &prediction - &target;
            let input_grad = network.input_gradient_batch(output_errors.view());
            network.train_minibatch(input.view(), target.view(), learning_rate);

            let scale = 1.0 / sentence.len() as f32;
            let token_grads = ndarray::Array2::from_shape_fn(
                (sentence.len(), embedding_dim),
                |(_, d)| input_grad[[0, d]] * scale,
            );
            let table_grad = embedding_layer.backward_embeddings(token_grads.view());
            embedding_layer.update(&table_grad, learning_rate);
        }

        if epoch % 10 == 0 || epoch == 39 {
            println!("Epoch {}: Loss = {:.4}", epoch + 1, total_loss);
        }
    }

    // Test the classifier
    println!("\n=== Testing Classifier ===");
    for (sentence, label) in &samples {
        let embeddings = embedding_layer.forward_sequence(sentence);
        let pooled = embeddings.mean_axis(ndarray::Axis(0)).unwrap();
        let prediction = network.forward(pooled.view());

        println!(
            "{:?} labelled {:.0}: predicted {:.3}",
            sentence, label, prediction[0]
        );
    }

    // Example: Using pre-trained embeddings
    println!("\n=== Using Pre-trained Embeddings ===");
    let pretrained = ndarray::Array2::random((100, 25), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
    let pretrained_layer = EmbeddingLayer::from_pretrained(pretrained);
    println!("Loaded pre-trained embeddings: {} words, {} dimensions", 
             pretrained_layer.vocab_size, pretrained_layer.embedding_dim);
}