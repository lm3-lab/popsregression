"""Tests for POPSRegression with several targets, a fixed ridge and the public box."""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose

import popsregression._pops as pops_module
from popsregression import POPSRegression

EXACT = {"leverage_regularization": 0.0, "minimum_relative_correction": 0.0}


def _make_dynamics(n_samples=600, n_features=5, n_targets=3, seed=0):
    """Misspecified linear model of several targets sharing one design."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    M = rng.randn(n_targets, n_features)
    Y = X @ M.T + 0.1 * np.sin(2.0 * X[:, :n_targets]) + 0.05 * X[:, [0]] ** 2
    return X, Y


def _ridge(X, Y, ridge):
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y).T


# --- Shapes ---


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_attribute_shapes(fit_intercept):
    X, Y = _make_dynamics()
    n, f = X.shape
    k = Y.shape[1]
    p = f + int(fit_intercept)
    model = POPSRegression(fit_intercept=fit_intercept, random_state=0).fit(X, Y)
    assert model.n_targets_ == k
    assert model.coef_.shape == (k, f)
    assert np.shape(model.intercept_) == (k,)
    assert model.sigma_.shape == (p, p)
    assert model.misspecification_sigma_.shape == (k * p, k * p)
    assert model.posterior_samples_.shape[0] == k * p
    assert model.hypercube_support_.shape[0] == k * p
    low, high = model.hypercube_bounds_
    assert low.shape == high.shape == (model.hypercube_support_.shape[1],)
    assert model.pointwise_corrections_.shape == (n, k * p)
    assert model.leverage_.shape == model.selected_.shape == (n,)


def test_single_column_target_is_one_dimensional():
    X, Y = _make_dynamics(n_targets=1)
    model = POPSRegression(random_state=0).fit(X, Y)
    assert model.coef_.ndim == 1
    assert model.n_targets_ == 1
    assert model.predict(X).shape == (X.shape[0],)


# --- Mean fit ---


@pytest.mark.parametrize("n_targets", [1, 3])
def test_fixed_ridge_is_the_ridge_solution(n_targets):
    X, Y = _make_dynamics(n_targets=n_targets)
    Y = Y[:, 0] if n_targets == 1 else Y
    model = POPSRegression(ridge=0.7, posterior=None).fit(X, Y)
    expected = _ridge(X, Y[:, None] if n_targets == 1 else Y, 0.7)
    assert_allclose(
        model.coef_, expected[0] if n_targets == 1 else expected, rtol=1e-10
    )
    assert model.n_iter_ == 0
    assert_allclose(model.lambda_ / model.alpha_, 0.7, rtol=1e-12)


def test_fixed_ridge_satisfies_the_normal_equations():
    X, Y = _make_dynamics()
    ridge = 1e-3
    model = POPSRegression(ridge=ridge, posterior=None).fit(X, Y)
    Lambda = X.T @ X + ridge * np.eye(X.shape[1])
    assert_allclose(Lambda @ model.coef_.T, X.T @ Y, rtol=1e-10, atol=1e-10)


def test_fixed_ridge_with_prior_shape():
    X, Y = _make_dynamics()
    shape = np.linspace(0.5, 3.0, X.shape[1])
    model = POPSRegression(ridge=0.3, prior_precision=shape, posterior=None).fit(X, Y)
    expected = np.linalg.solve(X.T @ X + 0.3 * np.diag(shape), X.T @ Y).T
    assert_allclose(model.coef_, expected, rtol=1e-10)


def test_evidence_is_pooled_over_targets():
    """Duplicated targets give the precisions and coefficients of one target."""
    X, Y = _make_dynamics(n_targets=1)
    y = Y[:, 0]
    # A tight tolerance: the stopping test sums the change over the targets.
    # Flat hyperpriors: the one Gamma prior of the shared precisions would
    # otherwise weigh half as much against two copies of the data.
    flat = {name: 0.0 for name in ("alpha_1", "alpha_2", "lambda_1", "lambda_2")}
    params = {"posterior": None, "tol": 1e-14, "max_iter": 1000, **flat}
    single = POPSRegression(**params).fit(X, y)
    double = POPSRegression(**params).fit(X, np.column_stack([y, y]))
    assert_allclose(double.alpha_, single.alpha_, rtol=1e-8)
    assert_allclose(double.lambda_, single.lambda_, rtol=1e-8)
    assert_allclose(double.coef_, np.vstack([single.coef_] * 2), rtol=1e-8)


def test_targets_share_the_metric():
    X, Y = _make_dynamics()
    model = POPSRegression(posterior=None).fit(X, Y)
    ratio = model.lambda_ / model.alpha_
    assert_allclose(model.coef_, _ridge(X, Y, ratio), rtol=1e-8)


# --- Pointwise corrections ---


def test_joint_corrections_fit_every_target_of_every_point_exactly():
    X, Y = _make_dynamics()
    n, f = X.shape
    k = Y.shape[1]
    model = POPSRegression(ridge=1e-2, **EXACT).fit(X, Y)
    D = model.pointwise_corrections_.reshape(n, k, f)
    fitted = np.einsum("if,ikf->ik", X, model.coef_[None] + D)
    assert_allclose(fitted, Y, rtol=1e-9, atol=1e-9)


def test_joint_corrections_are_rank_one():
    """D_i = e_i g_i^T with the residual vector e_i and one direction g_i."""
    X, Y = _make_dynamics()
    n, f = X.shape
    k = Y.shape[1]
    ridge = 1e-2
    model = POPSRegression(ridge=ridge, **EXACT).fit(X, Y)
    D = model.pointwise_corrections_.reshape(n, k, f)
    errors = Y - X @ model.coef_.T
    g = np.linalg.solve(X.T @ X + ridge * np.eye(f), X.T).T
    g /= np.einsum("if,if->i", X, g)[:, None]
    assert_allclose(D, errors[:, :, None] * g[:, None, :], rtol=1e-8, atol=1e-12)


def test_joint_corrections_stack_the_single_target_corrections():
    X, Y = _make_dynamics()
    n, f = X.shape
    k = Y.shape[1]
    params = {"ridge": 0.05, "leverage_regularization": 1e-3}
    joint = POPSRegression(**params).fit(X, Y)
    D = joint.pointwise_corrections_.reshape(n, k, f)
    for j in range(k):
        single = POPSRegression(**params).fit(X, Y[:, j])
        assert_allclose(joint.coef_[j], single.coef_, rtol=1e-10)
        assert_allclose(D[:, j], single.pointwise_corrections_, rtol=1e-8, atol=1e-14)


def test_selection_uses_the_norm_of_the_joint_correction():
    X, Y = _make_dynamics()
    model = POPSRegression(ridge=1e-2, minimum_relative_correction=0.5).fit(X, Y)
    norm = np.linalg.norm(model.pointwise_corrections_, axis=1)
    rms = np.sqrt(np.mean(norm**2))
    assert_allclose(model.selected_, norm >= 0.5 * rms)


# --- Hypercube ---


def test_support_is_orthonormal():
    X, Y = _make_dynamics()
    W = POPSRegression(ridge=1e-2, **EXACT).fit(X, Y).hypercube_support_
    assert_allclose(W.T @ W, np.eye(W.shape[1]), atol=1e-10)


def test_hypercube_is_the_bounding_box_of_the_joint_corrections():
    X, Y = _make_dynamics()
    model = POPSRegression(ridge=1e-2, **EXACT).fit(X, Y)
    W = model.hypercube_support_
    low, high = model.hypercube_bounds_
    projected = model.pointwise_corrections_ @ W
    assert_allclose(low, projected.min(axis=0), rtol=1e-10, atol=1e-14)
    assert_allclose(high, projected.max(axis=0), rtol=1e-10, atol=1e-14)


def test_hypercube_contains_every_pointwise_fit():
    """Every correction lies in the span of the support and inside the box."""
    X, Y = _make_dynamics()
    model = POPSRegression(ridge=1e-2, mode_threshold=1e-14, **EXACT).fit(X, Y)
    W = model.hypercube_support_
    low, high = model.hypercube_bounds_
    D = model.pointwise_corrections_
    nu = D @ W
    assert_allclose(nu @ W.T, D, atol=1e-10 * np.abs(D).max())
    span = high - low
    assert np.all(nu >= low - 1e-12 * span) and np.all(nu <= high + 1e-12 * span)


def test_include_center_extends_the_box_to_the_mean():
    rng = np.random.RandomState(3)
    X = rng.randn(300, 3)
    # Positive residuals on one side only: a box that need not contain 0.
    Y = np.column_stack([X @ [1.0, -1.0, 0.5] + 0.3 * X[:, 0] ** 2, X[:, 1] ** 2])
    plain = POPSRegression(ridge=1e-2, **EXACT).fit(X, Y)
    centered = POPSRegression(ridge=1e-2, include_center=True, **EXACT).fit(X, Y)
    low, high = centered.hypercube_bounds_
    assert np.all(low <= 0.0) and np.all(high >= 0.0)
    assert_allclose(low, np.minimum(plain.hypercube_bounds_[0], 0.0))
    assert_allclose(high, np.maximum(plain.hypercube_bounds_[1], 0.0))
    assert_allclose(centered.hypercube_support_, plain.hypercube_support_)


def test_include_center_for_one_target():
    X, Y = _make_dynamics(n_targets=1)
    model = POPSRegression(include_center=True).fit(X, Y[:, 0])
    low, high = model.hypercube_bounds_
    assert np.all(low <= 0.0) and np.all(high >= 0.0)


def test_blocks_do_not_change_the_result(monkeypatch):
    X, Y = _make_dynamics()
    params = {"ridge": 1e-2, "random_state": 0, **EXACT}
    whole = POPSRegression(**params).fit(X, Y)
    monkeypatch.setattr(pops_module, "_BLOCK_ENTRIES", 7 * Y.shape[1] * X.shape[1])
    blocked = POPSRegression(**params).fit(X, Y)
    assert_allclose(blocked.selected_, whole.selected_)
    assert_allclose(blocked.pointwise_corrections_, whole.pointwise_corrections_)
    # Eigenvectors are defined up to sign: align the columns, then compare.
    signs = np.sign(np.sum(blocked.hypercube_support_ * whole.hypercube_support_, 0))
    assert_allclose(
        blocked.hypercube_support_ * signs, whole.hypercube_support_, atol=1e-8
    )
    low, high = blocked.hypercube_bounds_
    aligned_low = np.where(signs > 0, low, -high)
    aligned_high = np.where(signs > 0, high, -low)
    assert_allclose(aligned_low, whole.hypercube_bounds_[0], rtol=1e-8, atol=1e-14)
    assert_allclose(aligned_high, whole.hypercube_bounds_[1], rtol=1e-8, atol=1e-14)


def test_posterior_samples_lie_in_the_box_and_are_reproducible():
    X, Y = _make_dynamics()
    params = {"ridge": 1e-2, "resampling_method": "latin", "random_state": 4}
    model = POPSRegression(**params).fit(X, Y)
    again = POPSRegression(**params).fit(X, Y)
    assert_allclose(model.posterior_samples_, again.posterior_samples_)
    W = model.hypercube_support_
    low, high = model.hypercube_bounds_
    nu = W.T @ model.posterior_samples_
    assert np.all(nu >= low[:, None] - 1e-12) and np.all(nu <= high[:, None] + 1e-12)


def test_ensemble_posterior_with_several_targets():
    X, Y = _make_dynamics()
    model = POPSRegression(ridge=1e-2, posterior="ensemble").fit(X, Y)
    kept = model.pointwise_corrections_[model.selected_]
    assert_allclose(model.posterior_samples_, kept.T)


# --- Prediction ---


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_predict_with_several_targets(fit_intercept):
    X, Y = _make_dynamics()
    n, f = X.shape
    k = Y.shape[1]
    model = POPSRegression(fit_intercept=fit_intercept, random_state=0).fit(X, Y)
    mean, std, upper, lower, epistemic = model.predict(
        X[:20], return_std=True, return_bounds=True, return_epistemic_std=True
    )
    for array in (mean, std, upper, lower, epistemic):
        assert array.shape == (20, k)
    assert_allclose(mean, X[:20] @ model.coef_.T + model.intercept_)
    assert np.all(lower <= mean + 1e-12) and np.all(mean <= upper + 1e-12)
    # Each target uses its own block of the joint posterior.
    p = f + int(fit_intercept)
    design = np.hstack([X[:20], np.ones((20, 1))]) if fit_intercept else X[:20]
    for j in range(k):
        samples = model.posterior_samples_[j * p : (j + 1) * p]
        values = design @ samples
        assert_allclose(upper[:, j], mean[:, j] + values.max(axis=1))
        assert_allclose(lower[:, j], mean[:, j] + values.min(axis=1))


def test_predict_of_one_target_matches_the_single_target_model():
    X, Y = _make_dynamics()
    params = {"ridge": 1e-2, "random_state": 0}
    joint = POPSRegression(**params).fit(X, Y)
    single = POPSRegression(**params).fit(X, Y[:, 1])
    assert_allclose(joint.predict(X[:10])[:, 1], single.predict(X[:10]))
    # Same metric, but the noise precision is pooled over all targets.
    assert_allclose(
        joint.predict(X[:10], return_epistemic_std=True)[1][:, 1],
        single.predict(X[:10], return_epistemic_std=True)[1]
        * np.sqrt(single.alpha_ / joint.alpha_),
    )


def test_score_with_several_targets():
    X, Y = _make_dynamics()
    model = POPSRegression(posterior=None).fit(X, Y)
    per_target = [
        POPSRegression(ridge=model.lambda_ / model.alpha_, posterior=None)
        .fit(X, Y[:, j])
        .score(X, Y[:, j])
        for j in range(Y.shape[1])
    ]
    assert_allclose(model.score(X, Y), np.mean(per_target), rtol=1e-8)


# --- Mean-only fits ---


def test_mean_only_fit_has_no_posterior():
    X, Y = _make_dynamics()
    model = POPSRegression(posterior=None).fit(X, Y)
    for name in ("posterior_samples_", "misspecification_sigma_", "selected_"):
        assert not hasattr(model, name)
    mean, epistemic = model.predict(X[:5], return_epistemic_std=True)
    assert mean.shape == epistemic.shape == (5, Y.shape[1])
    with pytest.raises(ValueError, match="posterior=None"):
        model.predict(X[:5], return_std=True)
    with pytest.raises(ValueError, match="posterior=None"):
        model.predict(X[:5], return_bounds=True)


# --- Validation ---


@pytest.mark.parametrize("ridge", [0.0, -1.0, "a"])
def test_invalid_ridge(ridge):
    X, Y = _make_dynamics()
    with pytest.raises((ValueError, TypeError)):
        POPSRegression(ridge=ridge).fit(X, Y)


def test_invalid_targets():
    X, Y = _make_dynamics()
    with pytest.raises(ValueError, match="shape"):
        POPSRegression().fit(X, Y[:, :, None])
    with pytest.raises(ValueError, match="inconsistent"):
        POPSRegression().fit(X, Y[:-1])
    bad = Y.copy()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        POPSRegression().fit(X, bad)
