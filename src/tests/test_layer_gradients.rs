//! Finite-difference checks for the layers that carry their own backward pass.
//!
//! Conv2D is checked in `src/layers/conv.rs`, the recurrent layers in
//! `src/tests/test_recurrent.rs`, and dense backprop through
//! `src/debug/gradient_check.rs`. This file covers pooling and batch norm.

use ndarray::{Array2, Array4};

use crate::layers::{AvgPool2DLayer, BatchNormLayer, LayerTrait, MaxPool2DLayer};

// f32 loss differences are quantized at about value * 1e-7, so a smaller step is noise
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

fn spatial_input() -> Array4<f32> {
    // Distinct values so max pooling has an unambiguous argmax in every window
    Array4::from_shape_fn((1, 2, 4, 4), |(_, c, h, w)| {
        ((c * 16 + h * 4 + w) as f32 * 0.37).sin()
    })
}

#[test]
fn max_pool_routes_the_gradient_to_the_argmax() {
    let mut layer = MaxPool2DLayer::new((2, 2), Some((2, 2)));
    let input = spatial_input();

    let output = layer.forward_batch(input.view());
    let target = Array4::from_shape_fn(output.raw_dim(), |(_, c, h, w)| {
        ((c + h * 2 + w * 3) as f32 * 0.21).cos() * 0.5
    });

    let output_gradient = &output - &target;
    let analytic = layer.backward_batch(output_gradient.view());

    let mut loss = |layer: &mut MaxPool2DLayer, input: &Array4<f32>| -> f32 {
        let out = layer.forward_batch(input.view());
        0.5 * (&out - &target).mapv(|v| v * v).sum()
    };

    for (c, h, w) in [(0usize, 0usize, 0usize), (1, 2, 3), (0, 3, 1), (1, 1, 2)] {
        let mut perturbed = input.clone();
        perturbed[[0, c, h, w]] = input[[0, c, h, w]] + EPS;
        let plus = loss(&mut layer, &perturbed);
        perturbed[[0, c, h, w]] = input[[0, c, h, w]] - EPS;
        let minus = loss(&mut layer, &perturbed);

        let numerical = (plus - minus) / (2.0 * EPS);
        agrees(
            analytic[[0, c, h, w]],
            numerical,
            &format!("MaxPool2D dx [{}, {}, {}]", c, h, w),
        );
    }

    // Exactly one cell per window receives gradient, the rest are zero
    let nonzero = analytic.iter().filter(|v| v.abs() > 1e-9).count();
    assert_eq!(
        nonzero,
        output.len(),
        "one input per pooling window should carry gradient"
    );
}

#[test]
fn average_pool_spreads_the_gradient_evenly() {
    let mut layer = AvgPool2DLayer::new((2, 2), Some((2, 2)));
    let input = spatial_input();

    let output = layer.forward_batch(input.view());
    let target = Array4::from_shape_fn(output.raw_dim(), |(_, c, h, w)| {
        ((c + h * 2 + w * 3) as f32 * 0.17).cos() * 0.5
    });

    let output_gradient = &output - &target;
    let analytic = layer.backward_batch(output_gradient.view());

    let mut loss = |layer: &mut AvgPool2DLayer, input: &Array4<f32>| -> f32 {
        let out = layer.forward_batch(input.view());
        0.5 * (&out - &target).mapv(|v| v * v).sum()
    };

    for (c, h, w) in [(0usize, 0usize, 0usize), (1, 2, 3), (0, 3, 1)] {
        let mut perturbed = input.clone();
        perturbed[[0, c, h, w]] = input[[0, c, h, w]] + EPS;
        let plus = loss(&mut layer, &perturbed);
        perturbed[[0, c, h, w]] = input[[0, c, h, w]] - EPS;
        let minus = loss(&mut layer, &perturbed);

        let numerical = (plus - minus) / (2.0 * EPS);
        agrees(
            analytic[[0, c, h, w]],
            numerical,
            &format!("AvgPool2D dx [{}, {}, {}]", c, h, w),
        );
    }

    // Every input in a 2x2 window gets a quarter of that window's gradient
    assert!(analytic.iter().all(|v| v.abs() > 1e-9));
}

fn batch_norm_input() -> Array2<f32> {
    Array2::from_shape_fn((5, 3), |(i, j)| ((i * 3 + j) as f32 * 0.41).sin() + 0.2)
}

#[test]
fn batch_norm_backward_matches_finite_differences_in_training_mode() {
    let mut layer = BatchNormLayer::new(3, 0.1, 1e-5);
    layer.set_training(true);
    // Non-trivial scale and shift, so their gradients are not degenerate
    layer.gamma = ndarray::array![1.3, 0.7, 1.0];
    layer.beta = ndarray::array![0.1, -0.2, 0.0];

    let input = batch_norm_input();
    let output = layer.forward_batch(input.view());
    let target = Array2::from_shape_fn(output.raw_dim(), |(i, j)| ((i + j) as f32 * 0.3).cos());

    let output_gradient = &output - &target;
    let (grad_input, grad_gamma, grad_beta) = layer.backward_batch(output_gradient.view());

    // Running statistics update on every forward pass, so each probe has to start from
    // the same layer state
    let mut loss = |gamma: &ndarray::Array1<f32>,
                    beta: &ndarray::Array1<f32>,
                    input: &Array2<f32>| -> f32 {
        let mut probe = BatchNormLayer::new(3, 0.1, 1e-5);
        probe.set_training(true);
        probe.gamma = gamma.clone();
        probe.beta = beta.clone();
        let out = probe.forward_batch(input.view());
        0.5 * (&out - &target).mapv(|v| v * v).sum()
    };

    let gamma = layer.gamma.clone();
    let beta = layer.beta.clone();

    for j in 0..3 {
        let mut probe_gamma = gamma.clone();
        probe_gamma[j] = gamma[j] + EPS;
        let plus = loss(&probe_gamma, &beta, &input);
        probe_gamma[j] = gamma[j] - EPS;
        let minus = loss(&probe_gamma, &beta, &input);
        agrees(
            grad_gamma[[0, j]],
            (plus - minus) / (2.0 * EPS),
            &format!("BatchNorm gamma [{}]", j),
        );

        let mut probe_beta = beta.clone();
        probe_beta[j] = beta[j] + EPS;
        let plus = loss(&gamma, &probe_beta, &input);
        probe_beta[j] = beta[j] - EPS;
        let minus = loss(&gamma, &probe_beta, &input);
        agrees(
            grad_beta[j],
            (plus - minus) / (2.0 * EPS),
            &format!("BatchNorm beta [{}]", j),
        );
    }

    // The input gradient is the part the mean and variance paths run through, so it is
    // the one a wrong derivation gets wrong
    for (i, j) in [(0usize, 0usize), (2, 1), (4, 2), (1, 2)] {
        let mut perturbed = input.clone();
        perturbed[[i, j]] = input[[i, j]] + EPS;
        let plus = loss(&gamma, &beta, &perturbed);
        perturbed[[i, j]] = input[[i, j]] - EPS;
        let minus = loss(&gamma, &beta, &perturbed);

        agrees(
            grad_input[[i, j]],
            (plus - minus) / (2.0 * EPS),
            &format!("BatchNorm dx [{}, {}]", i, j),
        );
    }
}

#[test]
fn batch_norm_gamma_gradient_is_not_a_copy_of_beta_at_inference() {
    let mut layer = BatchNormLayer::new(3, 0.1, 1e-5);
    layer.gamma = ndarray::array![1.3, 0.7, 1.0];
    layer.running_mean = ndarray::array![0.1, 0.0, -0.2];
    layer.running_var = ndarray::array![0.5, 1.0, 2.0];
    layer.set_training(false);

    let input = batch_norm_input();
    let _ = layer.forward_batch(input.view());

    let output_gradient = Array2::from_shape_fn((5, 3), |(i, j)| ((i + j) as f32 * 0.5).sin());
    let (_, grad_gamma, grad_beta) = layer.backward_batch(output_gradient.view());

    // gamma multiplies the normalized value, beta does not, so their gradients differ
    let identical = grad_gamma
        .iter()
        .zip(grad_beta.iter())
        .all(|(a, b)| (a - b).abs() < 1e-9);
    assert!(!identical, "gamma gradient was a copy of the beta gradient");

    // gamma's gradient is sum over the batch of grad_output * normalized
    let std = layer.running_var.mapv(|v| (v + layer.epsilon).sqrt());
    for j in 0..3 {
        let expected: f32 = (0..5)
            .map(|i| output_gradient[[i, j]] * (input[[i, j]] - layer.running_mean[j]) / std[j])
            .sum();
        agrees(grad_gamma[[0, j]], expected, &format!("inference gamma [{}]", j));
    }
}

#[test]
fn batch_norm_backward_follows_the_branch_forward_took() {
    // Training mode with a single row falls back to the running statistics, because one
    // sample has no variance. Backward has to follow, rather than reading caches that
    // pass never wrote.
    let mut layer = BatchNormLayer::new(3, 0.1, 1e-5);
    layer.set_training(true);

    let single = Array2::from_shape_fn((1, 3), |(_, j)| j as f32 * 0.5);
    let _ = layer.forward_batch(single.view());

    let output_gradient = Array2::from_elem((1, 3), 1.0);
    let (grad_input, _, _) = layer.backward_batch(output_gradient.view());

    assert!(
        grad_input.iter().all(|v| v.is_finite()),
        "single-row backward produced non-finite gradients"
    );
}
