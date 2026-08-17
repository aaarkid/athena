//! metrics and debug are what an agent reads when training misbehaves, so they have to
//! be right about numbers a reader can check by hand.

use ndarray::{array, Array2};

use crate::activations::Activation;
use crate::debug::numerical_check::{check_value_range, sanitize_array};
use crate::layers::{DenseLayer, WeightInit};
use crate::metrics::{RunningStats, Statistics};
use crate::metrics::statistics::{
    check_numerical_issues, dead_neuron_check, frobenius_norm, l1_norm, linf_norm,
};

/// mean 3.5, population variance 35/12, min 1, max 6
const SIX: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

#[test]
fn statistics_match_hand_computed_values() {
    let stats = Statistics::from_slice(&SIX);

    assert_eq!(stats.count, 6);
    assert!((stats.mean - 3.5).abs() < 1e-6, "mean was {}", stats.mean);
    // Population variance of 1..6 is 35/12; std is its square root
    assert!(
        (stats.std - (35.0f32 / 12.0).sqrt()).abs() < 1e-5,
        "std was {}",
        stats.std
    );
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 6.0);
}

#[test]
fn running_statistics_converge_on_the_same_numbers() {
    let mut running = RunningStats::new();
    for &value in SIX.iter() {
        running.update(value);
    }

    let batch = Statistics::from_slice(&SIX);

    assert_eq!(running.count(), batch.count);
    assert!((running.mean() - batch.mean).abs() < 1e-5);
    assert!((running.std() - batch.std).abs() < 1e-5);
    assert_eq!(running.min(), batch.min);
    assert_eq!(running.max(), batch.max);

    // The unbiased estimator divides by count - 1, so it is the larger of the two
    assert!(running.sample_variance() > running.variance());
    assert!((running.sample_variance() - 35.0 / 10.0).abs() < 1e-5);

    running.reset();
    assert_eq!(running.count(), 0);
}

#[test]
fn an_empty_slice_reports_zeros_rather_than_nan() {
    let stats = Statistics::from_slice(&[]);
    assert_eq!(stats.count, 0);
    assert!(stats.mean.is_finite() && stats.std.is_finite());
}

#[test]
fn the_dead_neuron_check_flags_exactly_the_silent_columns() {
    // Four samples, three neurons. Column 1 never fires; column 2 fires once, just
    // above the threshold, so it is alive.
    let activations = array![
        [0.5, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 0.0, 0.02],
        [0.9, 0.0, 0.0],
    ];

    let dead = dead_neuron_check(activations.view(), 0.01);
    assert_eq!(dead, vec![false, true, false], "got {:?}", dead);

    // Raise the threshold past that single firing and column 2 goes quiet too
    let dead = dead_neuron_check(activations.view(), 0.05);
    assert_eq!(dead, vec![false, true, true], "got {:?}", dead);
}

#[test]
fn the_numerical_check_counts_nan_and_inf_separately() {
    let mut array = Array2::<f32>::zeros((2, 3));
    array[[0, 0]] = f32::NAN;
    array[[0, 1]] = f32::INFINITY;
    array[[1, 2]] = f32::NEG_INFINITY;

    let (has_issues, nans, infs) = check_numerical_issues(array.view());
    assert!(has_issues);
    assert_eq!(nans, 1);
    assert_eq!(infs, 2);

    let clean = Array2::<f32>::from_elem((2, 3), 0.5);
    let (has_issues, nans, infs) = check_numerical_issues(clean.view());
    assert!(!has_issues);
    assert_eq!((nans, infs), (0, 0));
}

#[test]
fn the_norms_match_hand_computed_values() {
    // 3-4-0-0: Frobenius 5, L1 7, Linf 4
    let array = array![[3.0f32, -4.0], [0.0, 0.0]];

    assert!((frobenius_norm(array.view()) - 5.0).abs() < 1e-6);
    assert!((l1_norm(array.view()) - 7.0).abs() < 1e-6);
    assert!((linf_norm(array.view()) - 4.0).abs() < 1e-6);
}

#[test]
fn sanitize_replaces_nan_and_infinity_and_leaves_the_rest() {
    let mut array = array![[f32::NAN, 1.5], [f32::INFINITY, -2.0]];
    sanitize_array(&mut array, 0.0, 10.0);

    assert_eq!(array[[0, 0]], 0.0);
    assert_eq!(array[[0, 1]], 1.5);
    assert_eq!(array[[1, 0]], 10.0);
    assert_eq!(array[[1, 1]], -2.0);

    assert!(check_value_range(array.view(), -10.0, 10.0));
    assert!(!check_value_range(array.view(), -1.0, 1.0));
}

/// Sample variance of a weight matrix.
fn variance(weights: &ndarray::Array2<f32>) -> f32 {
    let n = weights.len() as f32;
    let mean = weights.iter().sum::<f32>() / n;
    weights.iter().map(|&w| (w - mean) * (w - mean)).sum::<f32>() / n
}

#[test]
fn he_normal_has_the_variance_it_claims() {
    // fan_in 200, fan_out 50, 10,000 samples. He normal is 2/fan_in = 0.01; the
    // fan_out value would be 0.04 and Xavier's 2/(fan_in+fan_out) is 0.008, so a
    // window of 0.002 separates all three. fan_in and fan_out confusion is the
    // resident bug class in initialization code.
    let layer = DenseLayer::new_with_init(200, 50, Activation::Relu, WeightInit::HeNormal);
    let measured = variance(&layer.weights);
    let expected = 2.0 / 200.0;

    assert!(
        (measured - expected).abs() < 0.002,
        "He normal variance {} against {} (fan_out would give {})",
        measured,
        expected,
        2.0 / 50.0
    );
}

#[test]
fn xavier_normal_has_the_variance_it_claims() {
    let layer = DenseLayer::new_with_init(200, 50, Activation::Tanh, WeightInit::XavierNormal);
    let measured = variance(&layer.weights);
    let expected = 2.0 / 250.0;

    assert!(
        (measured - expected).abs() < 0.002,
        "Xavier normal variance {} against {}",
        measured,
        expected
    );
}

#[test]
fn zeros_initialization_produces_exactly_zero() {
    let layer = DenseLayer::new_with_init(20, 10, Activation::Relu, WeightInit::Zeros);
    assert!(layer.weights.iter().all(|&w| w == 0.0));
    assert!(layer.biases.iter().all(|&b| b == 0.0));
}
