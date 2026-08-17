//! Every loss function's gradient must be the derivative of its own value.
//!
//! Checked against finite differences rather than against a second hand-derived formula,
//! so an error in the derivation shows up here instead of being copied into the test.

use ndarray::{Array1, Array2};

use crate::loss::{CrossEntropyLoss, HuberLoss, Loss, MSE};

// The loss is an f32, so the difference of two values is quantized at about value * 1e-7;
// dividing by 2 * EPS sets the noise floor on the numerical derivative
const EPS: f32 = 1e-2;
const TOLERANCE: f32 = 5e-2;
const ABS_FLOOR: f32 = 1e-4;

fn agrees(analytic: f32, numerical: f32, what: &str) {
    let scale = (analytic.abs() + numerical.abs()).max(1e-3);
    let relative = (analytic - numerical).abs() / scale;
    assert!(
        (analytic - numerical).abs() < ABS_FLOOR || relative < TOLERANCE,
        "{}: analytic {} vs numerical {}, relative error {}",
        what,
        analytic,
        numerical,
        relative
    );
}

/// Compares `gradient` against finite differences of `compute`.
fn check_single<L: Loss>(loss: &L, prediction: Array1<f32>, target: Array1<f32>, name: &str) {
    let analytic = loss.gradient(prediction.view(), target.view());

    for i in 0..prediction.len() {
        let mut perturbed = prediction.clone();

        perturbed[i] = prediction[i] + EPS;
        let plus = loss.compute(perturbed.view(), target.view());
        perturbed[i] = prediction[i] - EPS;
        let minus = loss.compute(perturbed.view(), target.view());

        let numerical = (plus - minus) / (2.0 * EPS);
        agrees(analytic[i], numerical, &format!("{} gradient [{}]", name, i));
    }
}

/// Compares `gradient_batch` against finite differences of `compute_batch`.
fn check_batch<L: Loss>(loss: &L, predictions: Array2<f32>, targets: Array2<f32>, name: &str) {
    let analytic = loss.gradient_batch(predictions.view(), targets.view());

    for i in 0..predictions.shape()[0] {
        for j in 0..predictions.shape()[1] {
            let mut perturbed = predictions.clone();

            perturbed[[i, j]] = predictions[[i, j]] + EPS;
            let plus = loss.compute_batch(perturbed.view(), targets.view());
            perturbed[[i, j]] = predictions[[i, j]] - EPS;
            let minus = loss.compute_batch(perturbed.view(), targets.view());

            let numerical = (plus - minus) / (2.0 * EPS);
            agrees(
                analytic[[i, j]],
                numerical,
                &format!("{} gradient_batch [{}, {}]", name, i, j),
            );
        }
    }
}

fn predictions() -> Array2<f32> {
    Array2::from_shape_fn((3, 4), |(i, j)| ((i * 4 + j) as f32 * 0.37).sin())
}

fn targets() -> Array2<f32> {
    Array2::from_shape_fn((3, 4), |(i, j)| ((i + j * 2) as f32 * 0.21).cos() * 0.5)
}

#[test]
fn mse_gradient_matches_its_value() {
    let loss = MSE;
    check_single(
        &loss,
        Array1::from_vec(vec![0.5, -0.2, 0.8, 1.4]),
        Array1::from_vec(vec![1.0, -0.5, 0.3, 0.9]),
        "MSE",
    );
    check_batch(&loss, predictions(), targets(), "MSE");
}

#[test]
fn huber_gradient_matches_its_value_on_both_sides_of_delta() {
    let loss = HuberLoss::new(0.5);

    // Residuals deliberately straddling delta, so both branches are exercised. The kink
    // itself is skipped: the derivative there is not defined.
    check_single(
        &loss,
        Array1::from_vec(vec![0.1, 2.0, -1.5, 0.05]),
        Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
        "Huber",
    );
    check_batch(&loss, predictions() * 3.0, targets(), "Huber");
}

#[test]
fn cross_entropy_gradient_matches_its_value() {
    let loss = CrossEntropyLoss;

    // Predictions must be positive: the loss takes their log
    let prediction = Array1::from_vec(vec![0.7, 0.2, 0.05, 0.05]);
    let target = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    check_single(&loss, prediction, target, "CrossEntropy");

    let predictions = Array2::from_shape_fn((3, 4), |(i, j)| 0.1 + 0.2 * ((i + j) % 4) as f32);
    let targets = Array2::from_shape_fn((3, 4), |(i, j)| if j == i % 4 { 1.0 } else { 0.0 });
    check_batch(&loss, predictions, targets, "CrossEntropy");
}

#[test]
fn a_single_row_batch_matches_the_single_sample_form() {
    // Both report the mean over samples and features, so a one-row batch has to agree
    // with the single-sample call on the same numbers
    let prediction = Array1::from_vec(vec![0.5, -0.2, 0.8, 1.4]);
    let target = Array1::from_vec(vec![1.0, -0.5, 0.3, 0.9]);

    let as_batch = |a: &Array1<f32>| a.clone().insert_axis(ndarray::Axis(0));

    let mse = MSE;
    assert!(
        (mse.compute(prediction.view(), target.view())
            - mse.compute_batch(as_batch(&prediction).view(), as_batch(&target).view()))
        .abs()
            < 1e-6
    );

    let huber = HuberLoss::new(0.5);
    assert!(
        (huber.compute(prediction.view(), target.view())
            - huber.compute_batch(as_batch(&prediction).view(), as_batch(&target).view()))
        .abs()
            < 1e-6
    );

    let positive = Array1::from_vec(vec![0.7, 0.2, 0.05, 0.05]);
    let one_hot = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let ce = CrossEntropyLoss;
    assert!(
        (ce.compute(positive.view(), one_hot.view())
            - ce.compute_batch(as_batch(&positive).view(), as_batch(&one_hot).view()))
        .abs()
            < 1e-6
    );
}
