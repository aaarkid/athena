//! The cache-free inference path has to agree with the training forward pass and has to
//! work from behind a shared reference.

use std::sync::Arc;
use std::thread;

use ndarray::{array, Array1, Array2};

use crate::activations::Activation;
use crate::layers::Layer;
use crate::network::{InferenceBuffers, NeuralNetwork};
use crate::optimizer::{OptimizerWrapper, SGD};

fn network() -> NeuralNetwork {
    NeuralNetwork::new(
        &[4, 8, 6, 3],
        &[Activation::Relu, Activation::Tanh, Activation::Linear],
        OptimizerWrapper::SGD(SGD::new()),
    )
}

fn close(a: &Array1<f32>, b: &Array1<f32>) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-6, "{:?} vs {:?}", a, b);
    }
}

#[test]
fn predict_gives_the_same_numbers_as_forward() {
    let mut net = network();
    let input = array![0.4, -1.2, 0.05, 2.0];

    let from_forward = net.forward(input.view());
    let from_predict = net.predict(input.view());

    close(&from_forward, &from_predict);
}

#[test]
fn predict_batch_gives_the_same_numbers_as_forward_batch() {
    let mut net = network();
    let inputs = Array2::from_shape_fn((5, 4), |(i, j)| ((i * 4 + j) as f32 * 0.31).sin());

    let from_forward = net.forward_batch(inputs.view());
    let from_predict = net.predict_batch(inputs.view());

    assert_eq!(from_forward.dim(), from_predict.dim());
    for (x, y) in from_forward.iter().zip(from_predict.iter()) {
        assert!((x - y).abs() < 1e-6, "{} vs {}", x, y);
    }
}

#[test]
fn predict_leaves_no_caches_behind() {
    // backward_batch reads what forward_batch stored. predict must not store anything,
    // so a layer that has only ever seen predict still has nothing to hand back.
    use crate::layers::LayerTrait;

    let net = network();
    let input = array![1.0, 0.5, -0.5, 0.25];
    let _ = net.predict(input.view());

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let errors = Array2::zeros((1, 3));
        net.layers[2].backward_batch(errors.view())
    }));
    std::panic::set_hook(previous);

    assert!(caught.is_err(), "predict wrote the backward caches");
}

#[test]
fn reused_buffers_give_the_same_answer_every_call() {
    let net = network();
    let mut buffers = InferenceBuffers::new();

    let first = array![0.4, -1.2, 0.05, 2.0];
    let second = array![-1.0, 0.0, 3.0, 0.5];

    let expected_first = net.predict(first.view());
    let expected_second = net.predict(second.view());

    // Three passes over the same buffers: a stale value from an earlier call would show
    // up as a difference here
    for _ in 0..3 {
        close(&net.predict_into(first.view(), &mut buffers).to_owned(), &expected_first);
        close(&net.predict_into(second.view(), &mut buffers).to_owned(), &expected_second);
    }
}

#[test]
fn buffers_resize_between_batch_sizes() {
    let net = network();
    let mut buffers = InferenceBuffers::new();

    for rows in [1usize, 8, 3, 32, 2] {
        let inputs = Array2::from_shape_fn((rows, 4), |(i, j)| (i as f32) - (j as f32) * 0.5);
        let reused = net.predict_batch_into(inputs.view(), &mut buffers).to_owned();
        let fresh = net.predict_batch(inputs.view());
        assert_eq!(reused.dim(), (rows, 3));
        for (x, y) in reused.iter().zip(fresh.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}

#[test]
fn one_network_behind_an_arc_serves_several_threads() {
    let net = Arc::new(network());
    let input = array![0.4, -1.2, 0.05, 2.0];
    let expected = net.predict(input.view());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let net = Arc::clone(&net);
            let input = input.clone();
            thread::spawn(move || {
                let mut buffers = InferenceBuffers::new();
                let mut last = Array1::zeros(3);
                for _ in 0..50 {
                    last = net.predict_into(input.view(), &mut buffers).to_owned();
                }
                last
            })
        })
        .collect();

    for handle in handles {
        close(&handle.join().expect("worker panicked"), &expected);
    }
}

#[test]
fn a_single_layer_predicts_without_the_ping_pong() {
    // One layer never swaps buffers, so the result has to come out of the first one
    let mut net = NeuralNetwork::new(&[3, 2], &[Activation::Relu], OptimizerWrapper::SGD(SGD::new()));
    let input = array![1.0, -2.0, 0.5];
    close(&net.forward(input.view()), &net.predict(input.view()));
}

#[test]
fn try_predict_reports_a_wrong_input_width() {
    let net = network();
    assert!(net.try_predict(array![1.0, 2.0].view()).is_err());
    assert!(net.try_predict(array![1.0, 2.0, 3.0, 4.0].view()).is_ok());
}

#[test]
fn the_layer_default_forward_into_matches_its_forward() {
    // Layers that do not override forward_batch_into fall back to cloning themselves.
    // Checked on a dense layer through the trait default by calling it on a copy.
    use crate::layers::LayerTrait;

    let mut layer = Layer::new(3, 4, Activation::Relu);
    let inputs = Array2::from_shape_fn((2, 3), |(i, j)| (i as f32) * 0.5 - (j as f32));

    let expected = layer.forward_batch(inputs.view());
    let mut out = Array2::zeros((0, 0));
    layer.forward_batch_into(inputs.view(), &mut out);

    for (x, y) in expected.iter().zip(out.iter()) {
        assert!((x - y).abs() < 1e-6);
    }
}
