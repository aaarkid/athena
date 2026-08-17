//! The JSON export has to be readable back into a network that behaves the same.
//!
//! The old tests asserted only that a file appeared, so an exporter writing no weights
//! would have passed.

use std::path::Path;

use ndarray::Array2;

use crate::activations::Activation;
use crate::builders::NetworkBuilder;
use crate::export::{NetworkExporter, NetworkImporter};
use crate::network::NeuralNetwork;
use crate::optimizer::{OptimizerWrapper, SGD};

fn scratch(name: &str) -> String {
    format!("{}/{}", std::env::temp_dir().display(), name)
}

#[test]
fn a_network_survives_a_json_round_trip() {
    let mut net = NeuralNetwork::new(
        &[2, 3, 1],
        &[Activation::Relu, Activation::Linear],
        OptimizerWrapper::SGD(SGD::new()),
    );

    // Train first, so the weights are not the ones a fresh network would have anyway
    let inputs = Array2::from_shape_fn((8, 2), |(i, j)| (i as f32) * 0.2 - (j as f32));
    let targets = Array2::from_shape_fn((8, 1), |(i, _)| (i as f32).sin());
    for _ in 0..50 {
        net.train_minibatch(inputs.view(), targets.view(), 0.01);
    }

    let path = scratch("athena_round_trip.json");
    NetworkExporter::export_json(&net, Path::new(&path)).expect("export failed");

    let restored =
        NetworkImporter::import_network_json(Path::new(&path), OptimizerWrapper::SGD(SGD::new()))
            .expect("import failed");
    std::fs::remove_file(&path).ok();

    let probe = Array2::from_shape_fn((4, 2), |(i, j)| (i as f32) * 0.37 - (j as f32) * 0.11);
    let before = net.predict_batch(probe.view());
    let after = restored.predict_batch(probe.view());

    assert_eq!(before.dim(), after.dim());
    for (a, b) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1e-6, "{} vs {}", a, b);
    }
}

#[test]
fn the_exported_json_carries_the_weights() {
    let net = NeuralNetwork::new(
        &[2, 2],
        &[Activation::Linear],
        OptimizerWrapper::SGD(SGD::new()),
    );
    let known = net.layers[0].weights[[1, 0]];

    let path = scratch("athena_weights_present.json");
    NetworkExporter::export_json(&net, Path::new(&path)).expect("export failed");

    let text = std::fs::read_to_string(&path).expect("no file written");
    std::fs::remove_file(&path).ok();

    let parsed: serde_json::Value = serde_json::from_str(&text).expect("not valid JSON");
    assert_eq!(parsed["format"], "athena_network");

    let written = parsed["model"]["layers"][0]["weights"][1][0]
        .as_f64()
        .expect("no weight at [1][0]") as f32;
    assert!((written - known).abs() < 1e-6, "{} vs {}", written, known);
}

#[test]
fn an_import_reports_a_layer_that_does_not_chain() {
    let path = scratch("athena_broken_chain.json");
    // Layer 0 emits 3, layer 1 expects 5
    std::fs::write(
        &path,
        r#"{
            "format": "athena_network",
            "version": "1.0",
            "model": { "name": "n", "layers": [
                {"input_size": 2, "output_size": 3, "activation": "Relu",
                 "weights": [[0,0,0],[0,0,0]], "biases": [0,0,0]},
                {"input_size": 5, "output_size": 1, "activation": "Identity",
                 "weights": [[0],[0],[0],[0],[0]], "biases": [0]}
            ]}
        }"#,
    )
    .unwrap();

    let result =
        NetworkImporter::import_network_json(Path::new(&path), OptimizerWrapper::SGD(SGD::new()));
    std::fs::remove_file(&path).ok();
    assert!(result.is_err(), "a network whose layers do not chain was imported");
}

#[test]
fn an_import_reports_a_weight_matrix_of_the_wrong_size() {
    let path = scratch("athena_wrong_weights.json");
    // Declares 2 inputs but supplies one row
    std::fs::write(
        &path,
        r#"{
            "format": "athena_network",
            "version": "1.0",
            "model": { "name": "n", "layers": [
                {"input_size": 2, "output_size": 2, "activation": "Identity",
                 "weights": [[0,0]], "biases": [0,0]}
            ]}
        }"#,
    )
    .unwrap();

    let result =
        NetworkImporter::import_network_json(Path::new(&path), OptimizerWrapper::SGD(SGD::new()));
    std::fs::remove_file(&path).ok();
    assert!(result.is_err(), "a short weight matrix was imported");
}

#[test]
fn the_old_format_tag_still_reads() {
    let path = scratch("athena_old_tag.json");
    std::fs::write(
        &path,
        r#"{
            "format": "athena_onnx_export",
            "version": "1.0",
            "model": { "name": "n", "layers": [
                {"input_size": 2, "output_size": 1, "activation": "Identity",
                 "weights": [[0.5],[0.25]], "biases": [1.0]}
            ]}
        }"#,
    )
    .unwrap();

    let restored =
        NetworkImporter::import_network_json(Path::new(&path), OptimizerWrapper::SGD(SGD::new()))
            .expect("a file written before the tag changed no longer reads");
    std::fs::remove_file(&path).ok();

    let output = restored.predict(ndarray::array![2.0, 4.0].view());
    assert!((output[0] - (0.5 * 2.0 + 0.25 * 4.0 + 1.0)).abs() < 1e-6);
}

#[test]
fn try_new_reports_a_bad_shape_instead_of_overflowing() {
    // layer_sizes.len() - 1 underflows on an empty slice, so this used to abort
    assert!(NeuralNetwork::try_new(&[], &[], OptimizerWrapper::SGD(SGD::new())).is_err());
    assert!(NeuralNetwork::try_new(&[4], &[], OptimizerWrapper::SGD(SGD::new())).is_err());
    assert!(NeuralNetwork::try_new(
        &[4, 2],
        &[Activation::Relu, Activation::Linear],
        OptimizerWrapper::SGD(SGD::new())
    )
    .is_err());
    assert!(NeuralNetwork::try_new(
        &[4, 0, 2],
        &[Activation::Relu, Activation::Linear],
        OptimizerWrapper::SGD(SGD::new())
    )
    .is_err());
    assert!(NeuralNetwork::try_new(
        &[4, 8, 2],
        &[Activation::Relu, Activation::Linear],
        OptimizerWrapper::SGD(SGD::new())
    )
    .is_ok());
}

#[test]
fn choosing_adam_before_adding_layers_still_builds() {
    // with_adam used to do nothing when no layer had been added yet, after which
    // build() reported "Optimizer not specified"
    let network = NetworkBuilder::new()
        .with_adam(0.9, 0.999, 1e-8)
        .add_dense(4, 8, Activation::Relu)
        .add_dense(8, 2, Activation::Linear)
        .build()
        .expect("with_adam before the layers left the builder without an optimizer");

    assert!(matches!(network.optimizer, OptimizerWrapper::Adam(_)));

    let after = NetworkBuilder::new()
        .add_dense(4, 8, Activation::Relu)
        .add_dense(8, 2, Activation::Linear)
        .with_rmsprop(0.9, 1e-8)
        .build()
        .expect("with_rmsprop after the layers failed");
    assert!(matches!(after.optimizer, OptimizerWrapper::RMSProp(_)));
}
