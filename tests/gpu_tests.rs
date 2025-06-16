#[cfg(feature = "gpu")]
mod gpu_tests {
    use athena::layers::{GpuDenseLayer, DenseLayer, LayerTrait};
    use athena::activations::Activation;
    use ndarray::{Array1, Array2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_gpu_dense_layer_creation() {
        let layer = GpuDenseLayer::new(128, 64, Activation::Relu);
        assert!(layer.is_ok(), "Failed to create GPU layer");
        
        let layer = layer.unwrap();
        assert_eq!(layer.weights.shape(), &[128, 64]);
        assert_eq!(layer.biases.shape(), &[64]);
    }

    #[test] 
    fn test_gpu_forward_pass() {
        let mut gpu_layer = GpuDenseLayer::new(32, 16, Activation::Relu).unwrap();
        let mut cpu_layer = DenseLayer::new(32, 16, Activation::Relu);
        
        // Copy weights to ensure same initialization
        cpu_layer.weights = gpu_layer.weights.clone();
        cpu_layer.biases = gpu_layer.biases.clone();
        
        let input = Array1::random(32, Uniform::new(-1.0, 1.0));
        
        let gpu_output = gpu_layer.forward(input.view());
        let cpu_output = cpu_layer.forward(input.view());
        
        assert_eq!(gpu_output.shape(), cpu_output.shape());
        
        // Check outputs are reasonably close (allowing for floating point differences)
        let diff: f32 = (&gpu_output - &cpu_output).mapv(f32::abs).sum();
        assert!(diff < 1e-4, "GPU and CPU outputs differ too much: {}", diff);
    }

    #[test]
    fn test_gpu_batch_forward() {
        let mut gpu_layer = GpuDenseLayer::new(16, 8, Activation::Relu).unwrap();
        let batch_input = Array2::random((32, 16), Uniform::new(-1.0, 1.0));
        
        let output = gpu_layer.forward_batch(batch_input.view());
        assert_eq!(output.shape(), &[32, 8]);
        
        // Test that all outputs are non-negative (ReLU)
        assert!(output.iter().all(|&x| x >= 0.0), "ReLU should produce non-negative outputs");
    }

    #[test]
    fn test_gpu_device_info() {
        let layer = GpuDenseLayer::new(10, 10, Activation::Relu).unwrap();
        let info = layer.device_info();
        assert!(info.is_ok(), "Should be able to get device info");
        
        let info_str = info.unwrap();
        // Should contain device information (either real or mock)
        assert!(info_str.contains("Device:"));
        assert!(info_str.contains("Vendor:"));
    }

    #[test]
    fn test_gpu_fallback_on_error() {
        // Even if GPU initialization fails internally, the layer should work
        let mut layer = GpuDenseLayer::new(64, 32, Activation::Relu).unwrap();
        let input = Array1::random(64, Uniform::new(-1.0, 1.0));
        
        // This should work even if GPU is not available (falls back to CPU)
        let output = layer.forward(input.view());
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_gpu_layer_clone() {
        let layer = GpuDenseLayer::new(20, 10, Activation::Relu).unwrap();
        let cloned = layer.clone_box();
        
        // Verify clone has same dimensions
        assert_eq!(cloned.input_size(), 20);
        assert_eq!(cloned.output_size(), 10);
    }
}