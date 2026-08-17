//! The mock GPU backend has to produce the same numbers as the CPU path.
//!
//! It runs on the CPU either way, so any difference here is a bug in the mock rather
//! than a device precision question. Compiled only under `gpu-mock`, which is the
//! feature that exists so this path has coverage on a machine with no OpenCL.

use ndarray::Array2;

use crate::activations::Activation;
use crate::gpu::{ComputeBackend, MockGpuBackend};
use crate::layers::{Layer, LayerTrait};

#[test]
fn the_mock_matmul_matches_a_dense_forward_pass() {
    let backend = MockGpuBackend::new();

    let mut layer = Layer::new(6, 4, Activation::Linear);
    let inputs = Array2::from_shape_fn((3, 6), |(i, j)| ((i * 6 + j) as f32 * 0.19).sin());

    let from_layer = layer.forward_batch(inputs.view());

    let mut through_backend = backend
        .matmul(inputs.view(), layer.weights.view())
        .expect("mock matmul rejected a valid pair of shapes");
    through_backend += &layer.biases;

    assert_eq!(from_layer.dim(), through_backend.dim());
    for (a, b) in from_layer.iter().zip(through_backend.iter()) {
        assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
    }
}

#[test]
fn the_mock_elementwise_operations_match_ndarray() {
    let backend = MockGpuBackend::new();

    let a = Array2::from_shape_fn((4, 5), |(i, j)| (i as f32) - (j as f32) * 0.4);
    let b = Array2::from_shape_fn((4, 5), |(i, j)| (j as f32) * 0.7 - 1.0);

    let sum = backend.add(a.view(), b.view()).unwrap();
    let product = backend.multiply(a.view(), b.view()).unwrap();
    let rectified = backend.relu(a.view()).unwrap();

    for (x, y) in sum.iter().zip((&a + &b).iter()) {
        assert!((x - y).abs() < 1e-5);
    }
    for (x, y) in product.iter().zip((&a * &b).iter()) {
        assert!((x - y).abs() < 1e-5);
    }
    for (x, y) in rectified.iter().zip(a.mapv(|v| v.max(0.0)).iter()) {
        assert!((x - y).abs() < 1e-5);
    }
}

#[test]
fn the_mock_reports_a_shape_mismatch_instead_of_panicking() {
    let backend = MockGpuBackend::new();

    let a = Array2::<f32>::zeros((3, 4));
    let b = Array2::<f32>::zeros((5, 2));

    assert!(backend.matmul(a.view(), b.view()).is_err());
    assert!(backend.add(a.view(), b.view()).is_err());
    assert!(backend.multiply(a.view(), b.view()).is_err());
}

#[test]
fn the_mock_inserts_no_artificial_delay_by_default() {
    // An artificial sleep makes every measurement taken against this backend describe
    // nothing, so it has to be off unless something asks for it
    let backend = MockGpuBackend::new();
    let a = Array2::<f32>::zeros((64, 64));

    let start = std::time::Instant::now();
    for _ in 0..20 {
        let _ = backend.relu(a.view()).unwrap();
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed < std::time::Duration::from_millis(50),
        "20 relu calls on a 64 by 64 array took {:?}, so a delay is being inserted",
        elapsed
    );
}
