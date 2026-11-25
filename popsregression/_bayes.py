"""
Bayesian ridge regression by evidence maximization, in JAX.

The mean fit of :class:`~popsregression.POPSRegression`: a Gaussian prior
``N(0, 1 / lambda)`` on the coefficients and Gaussian noise of precision
``alpha``, both with Gamma hyperpriors, updated by the MacKay fixed point
[1]_. The design is reduced once to its triangular factor, so every evidence
iteration costs ``O(n_features**2)`` whatever the number of samples.

References
----------
.. [1] MacKay, D.J.C. (1992). "Bayesian interpolation." Neural Computation
       4(3), 415-447.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class RidgeFit(NamedTuple):
    """Evidence-maximized ridge fit, in the right singular basis of the design.

    ``V`` holds the right singular vectors (columns) and ``S`` the singular
    values of the (weighted) design, so that the regularized Gram matrix is
    ``A = V diag(S**2 + lambda / alpha) V^T`` on the row space, plus
    ``lambda / alpha`` on its complement when ``n_samples < n_features``.
    """

    coef: jnp.ndarray
    alpha: jnp.ndarray
    lam: jnp.ndarray
    n_iter: jnp.ndarray
    V: jnp.ndarray
    S: jnp.ndarray


@jax.jit
def _spectral_reduction(X, Y):
    """Singular basis of ``X`` and the part of every target it can and cannot reach.

    ``Y`` has shape ``(n_samples, n_targets)``. Returns ``(S, V, u, rho2)``
    with ``X = U diag(S) V^T``, ``u = U^T Y`` of shape ``(n_modes,
    n_targets)`` and ``rho2[j] = ||Y[:, j] - U u[:, j]||^2``. For tall designs
    the factorization goes through the triangular factor of ``[X, Y]``, which
    neither forms the Gram matrix (and so does not square the condition
    number) nor keeps an ``(n_samples, n_features)`` orthogonal factor.
    """
    n_samples, n_features = X.shape
    if n_samples > n_features:
        R = jnp.linalg.qr(jnp.column_stack([X, Y]), mode="r")
        U, S, Vt = jnp.linalg.svd(R[:n_features, :n_features], full_matrices=False)
        rho2 = jnp.sum(R[n_features:, n_features:] ** 2, axis=0)
        return S, Vt.T, U.T @ R[:n_features, n_features:], rho2
    U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
    u = U.T @ Y
    return S, Vt.T, u, jnp.sum((Y - U @ u) ** 2, axis=0)


@jax.jit
def bayesian_ridge(
    X, y, weights, alpha_init, lambda_init, hyper, tol, max_iter, ridge=jnp.nan
):
    """Evidence-maximized ridge fit of ``y ~ X @ coef``, one or several targets.

    With several targets (``y`` of shape ``(n_samples, n_targets)``) every
    target shares the design, the prior and the noise model, so the
    precisions ``alpha`` and ``lambda`` are pooled over the targets: the
    MacKay updates sum the effective number of parameters, the squared
    coefficients and the squared residuals of all targets. Every target then
    has the same regularized Gram matrix ``A = X^T W X + lambda / alpha``.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,) or (n_samples, n_targets)
    weights : array of shape (n_samples,)
        Non-negative sample weights.
    alpha_init, lambda_init : float
        Initial precisions; ``nan`` selects ``1 / var(y)`` and ``1``.
    hyper : array of shape (4,)
        ``(alpha_1, alpha_2, lambda_1, lambda_2)``, shape and rate of the
        Gamma hyperpriors on the noise and weight precisions.
    tol : float
        Stop once the coefficients move by less than ``tol`` in L1 norm.
    max_iter : int
        Maximum number of evidence iterations.
    ridge : float, default=nan
        Fixed ratio ``lambda / alpha``. ``nan`` learns it by evidence
        maximization; otherwise no evidence iteration runs, the coefficients
        are the ridge solution of this ratio, ``alpha`` is the MacKay noise
        precision of their residuals and ``lambda = ridge * alpha``.

    Returns
    -------
    RidgeFit
        ``coef`` has the shape of ``y`` with ``n_samples`` replaced by
        ``n_features``.
    """
    single = y.ndim == 1
    Y = y[:, None] if single else y
    n_targets = Y.shape[1]
    root = jnp.sqrt(weights)
    Xw, Yw = X * root[:, None], Y * root[:, None]
    # Weights act as fractional observation counts.
    n_effective = jnp.sum(weights)
    S, V, u, rho2 = _spectral_reduction(Xw, Yw)
    eig = S * S
    alpha_1, alpha_2, lambda_1, lambda_2 = hyper

    def solve(ratio):
        c = S[:, None] * u / (eig + ratio)[:, None]
        return V @ c, jnp.sum((u - S[:, None] * c) ** 2) + jnp.sum(rho2)

    def evidence(alpha, lam, coef, sse):
        gamma = n_targets * jnp.sum(alpha * eig / (lam + alpha * eig))
        lam_new = (gamma + 2.0 * lambda_1) / (jnp.sum(coef**2) + 2.0 * lambda_2)
        alpha_new = (n_targets * n_effective - gamma + 2.0 * alpha_1) / (
            sse + 2.0 * alpha_2
        )
        return alpha_new, lam_new

    def step(state):
        i, alpha, lam, coef_old, _ = state
        coef, sse = solve(lam / alpha)
        alpha_new, lam_new = evidence(alpha, lam, coef, sse)
        done = (i > 0) & (jnp.sum(jnp.abs(coef_old - coef)) < tol)
        return i + 1, alpha_new, lam_new, coef, done

    fixed = ~jnp.isnan(ridge)

    def running(state):
        return (state[0] < max_iter) & ~state[4] & ~fixed

    eps = jnp.finfo(X.dtype).eps
    mean = jnp.sum(weights[:, None] * Y, axis=0) / n_effective
    y_var = jnp.sum(weights[:, None] * (Y - mean) ** 2) / n_targets
    alpha0 = jnp.where(
        jnp.isnan(alpha_init), 1.0 / (y_var / n_effective + eps), alpha_init
    )
    lam0 = jnp.where(jnp.isnan(lambda_init), 1.0, lambda_init)
    start = (0, alpha0, lam0, jnp.zeros((X.shape[1], n_targets), X.dtype), False)
    n_iter, alpha, lam, _, _ = jax.lax.while_loop(running, step, start)
    coef, sse = solve(jnp.where(fixed, ridge, lam / alpha))
    alpha_fixed, _ = evidence(1.0, ridge, coef, sse)
    alpha = jnp.where(fixed, alpha_fixed, alpha)
    lam = jnp.where(fixed, ridge * alpha_fixed, lam)
    return RidgeFit(coef[:, 0] if single else coef, alpha, lam, n_iter, V, S)


@jax.jit
def scaled_covariance(fit):
    """``alpha * Sigma = (X^T X + lambda / alpha)^{-1}`` of a ridge fit."""
    ratio = fit.lam / fit.alpha
    V = fit.V
    row_space = (V / (fit.S**2 + ratio)) @ V.T
    n_features, n_modes = V.shape
    if n_modes < n_features:
        row_space = row_space + (jnp.eye(n_features) - V @ V.T) / ratio
    return row_space


def as_nan(value):
    """``None`` as ``nan``, the sentinel of an unset initial precision."""
    return np.nan if value is None else float(value)
