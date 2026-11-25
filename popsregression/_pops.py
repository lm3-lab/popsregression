"""
POPS (Pointwise Optimal Parameter Sets) Regression.

Bayesian regression for low-noise data accounting for model misspecification.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from ._base import (
    BaseEstimator,
    check_is_fitted,
    check_option,
    check_random_state,
    check_sample_weight,
    check_scalar,
    check_X,
    check_X_targets,
)
from ._bayes import as_nan, bayesian_ridge, scaled_covariance

_RESAMPLING = {"latin": qmc.LatinHypercube, "sobol": qmc.Sobol, "halton": qmc.Halton}


# Largest number of entries of a block of stacked corrections held at once.
_BLOCK_ENTRIES = 2**24


class _Corrections:
    """Pointwise corrections of every target, generated in blocks of rows.

    The correction of training point ``i`` for target ``j`` is the smallest
    parameter move, in the metric ``A = X^T W X + lambda / alpha`` of the
    mean fit, after which the model passes through ``y_ij``. In closed form
    it is ``e_ij A^-1 x_i / h_i`` with residual ``e_ij`` and leverage ``h_i
    = x_i^T A^-1 x_i``, which diverges for points of vanishing leverage.
    Asking instead for the minimizer of ``w_i (x_i @ d - e_ij)**2 / eps +
    d^T A d`` gives

        d_ij = e_ij A^-1 x_i * w_i / (w_i h_i + eps),

    which is bounded: ``||d_ij||_A <= |e_ij| sqrt(w_i / eps) / 2``. ``eps`` is
    ``leverage_regularization`` times the mean (weighted) leverage, so the
    regularization is dimensionless and only acts on points far less
    informative than the typical one.

    All targets share ``A`` (their precisions are pooled), so the
    corrections of one point are the rank-one matrix ``e_i (A^-1 x_i)^T``
    up to the scalar ``w_i / (w_i h_i + eps)``: the correction of point
    ``i`` to all targets at once, ``D_i = [d_i1; ...; d_ik]``, is the
    smallest move in the metric ``I_k (x) A`` after which the model passes
    through every target of point ``i``. Its target-major vectorization
    ``vec(D_i)`` is the row stored for point ``i``; with one target it is
    ``d_i``.

    Blocks are in the right singular basis ``fit.V`` of the design: a row of
    a block holds ``n_targets`` consecutive vectors of ``n_modes``
    coordinates.
    """

    def __init__(self, X, Y, weights, fit, leverage_regularization):
        self.T, self.inverse_gram, self.scale, self.leverage = _correction_factors(
            X, Y, weights, fit, leverage_regularization
        )
        self.n_samples, self.n_targets = self.scale.shape
        self.n_modes = self.T.shape[1]
        self.rows = max(1, _BLOCK_ENTRIES // (self.n_targets * self.n_modes))

    def blocks(self):
        """``(start, stop, block)`` with ``block`` of shape ``(stop - start,
        n_targets * n_modes)``."""
        for start in range(0, self.n_samples, self.rows):
            stop = min(start + self.rows, self.n_samples)
            yield start, stop, _correction_block(
                self.T[start:stop], self.scale[start:stop], self.inverse_gram
            )

    def in_parameters(self, basis):
        """Every correction in parameter coordinates, ``(n_samples, n_targets *
        n_parameters)`` (target-major)."""
        rows = [
            np.asarray(_to_parameters(block, basis, self.n_targets))
            for _, _, block in self.blocks()
        ]
        return np.concatenate(rows, axis=0)


@jax.jit
def _correction_factors(X, Y, weights, fit, leverage_regularization):
    """Singular-basis design ``T``, inverse Gram, scales ``w e / (w h + eps)``
    and leverages ``w h`` of the pointwise corrections."""
    T = X @ fit.V
    inverse_gram = 1.0 / (fit.S**2 + fit.lam / fit.alpha)
    errors = Y - X @ fit.coef.reshape(fit.coef.shape[0], -1)
    leverage = weights * jnp.sum(T * T * inverse_gram, axis=1)
    active = weights > 0
    mean_leverage = jnp.sum(jnp.where(active, leverage, 0.0)) / jnp.sum(active)
    denominator = (leverage + leverage_regularization * mean_leverage)[:, None]
    scale = jnp.where(denominator > 0, weights[:, None] * errors / denominator, 0.0)
    return T, inverse_gram, scale, leverage


@jax.jit
def _correction_block(T, scale, inverse_gram):
    """Rows ``vec(D_i)`` of a block of points, in the singular basis."""
    block = (scale[:, :, None] * T[:, None, :]) * inverse_gram
    return block.reshape(T.shape[0], -1)


def _to_parameters(block, basis, n_targets):
    """Rows of a block in parameter coordinates (``basis`` per target)."""
    n_rows = block.shape[0]
    per_target = block.reshape(n_rows, n_targets, basis.shape[1]) @ basis.T
    return per_target.reshape(n_rows, -1)


@jax.jit
def _block_norm2(block, metric):
    """Squared Euclidean norm of every row, ``metric = basis^T basis``."""
    rows = block.reshape(block.shape[0], -1, metric.shape[0])
    return jnp.sum((rows @ metric) * rows, axis=(1, 2))


@jax.jit
def _block_gram(block, keep):
    kept = jnp.where(keep[:, None], block, 0.0)
    return kept.T @ kept


@jax.jit
def _block_projection(block, keep, directions):
    """Coordinates of the kept rows along ``directions`` (NaN elsewhere)."""
    return jnp.where(keep[:, None], block @ directions, jnp.nan)


def _select_and_bound(corrections, basis, active, threshold, clipping):
    """Prune negligible corrections, then fit the PCA-aligned box.

    The corrections are the rows ``vec(D_i)`` of :class:`_Corrections`; in
    parameter coordinates every target block is mapped by ``basis``. Points
    are kept when their Euclidean norm ``||vec(D_i)||`` is at least
    ``threshold`` times the root-mean-square norm: these are the rows that
    determine the singular decomposition of the correction matrix.

    Returns the selection mask, the eigenvalues and eigenvectors (in
    parameter coordinates, target-major) of the pruned second-moment matrix
    and the box bounds along every eigenvector.
    """
    metric = basis.T @ basis
    norm2 = np.concatenate(
        [
            np.asarray(_block_norm2(block, metric))
            for _, _, block in corrections.blocks()
        ]
    )
    mean_norm2 = np.sum(np.where(active, norm2, 0.0)) / np.sum(active)
    keep = active & (norm2 >= threshold**2 * mean_norm2)
    keep = np.where(np.any(keep), keep, active)

    gram = None
    for start, stop, block in corrections.blocks():
        part = _block_gram(block, keep[start:stop])
        gram = part if gram is None else gram + part
    full_basis = (
        basis
        if corrections.n_targets == 1
        else jnp.kron(jnp.eye(corrections.n_targets), basis)
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(full_basis @ gram @ full_basis.T)
    directions = full_basis.T @ eigenvectors

    if clipping == 0.0:
        low = np.full(eigenvalues.shape[0], np.inf)
        high = np.full(eigenvalues.shape[0], -np.inf)
        for start, stop, block in corrections.blocks():
            projected = _block_projection(block, keep[start:stop], directions)
            low = np.fmin(low, np.asarray(jnp.nanmin(projected, axis=0)))
            high = np.fmax(high, np.asarray(jnp.nanmax(projected, axis=0)))
    else:
        projected = jnp.concatenate(
            [
                _block_projection(block, keep[start:stop], directions)
                for start, stop, block in corrections.blocks()
            ]
        )
        low = np.asarray(jnp.nanpercentile(projected, clipping, axis=0))
        high = np.asarray(jnp.nanpercentile(projected, 100.0 - clipping, axis=0))
    return keep, eigenvalues, eigenvectors, low, high


def prior_whitening(prior_precision, n_design, n_features):
    """Map ``T`` from whitened to parameter coordinates, ``T T^T = Lambda^-1``.

    With ``theta = T phi`` a Gaussian prior ``N(0, (lambda Lambda)^-1)`` on
    ``theta`` is the isotropic prior ``N(0, 1 / lambda)`` on ``phi``, so the
    fit of the design ``X @ T`` is the fit with prior shape ``Lambda``.

    ``prior_precision`` is a vector (diagonal ``Lambda``) or a symmetric
    positive definite matrix, over the ``n_design`` columns of the design or
    over its first ``n_features`` columns; any remaining column (the
    intercept) then gets unit precision. Returns ``None`` for no prior
    shape, a vector for a diagonal ``T`` and a matrix otherwise.
    """
    if prior_precision is None:
        return None
    precision = np.asarray(prior_precision, dtype=np.float64)
    size = precision.shape[0] if precision.ndim in (1, 2) else -1
    square = precision.ndim == 2 and precision.shape[0] == precision.shape[1]
    if size not in (n_design, n_features) or not (precision.ndim == 1 or square):
        raise ValueError(
            "prior_precision must have shape (n_features,) or (n_features, "
            f"n_features) with n_features = {n_features}, got {precision.shape}."
        )
    if not np.all(np.isfinite(precision)):
        raise ValueError("prior_precision contains NaN or infinite values.")
    if precision.ndim == 1:
        if np.any(precision <= 0):
            raise ValueError("prior_precision must be positive.")
        scale = np.ones(n_design)
        scale[:size] = 1.0 / np.sqrt(precision)
        return scale
    if not np.allclose(precision, precision.T, rtol=1e-10, atol=0.0):
        raise ValueError("prior_precision must be symmetric.")
    try:
        factor = np.linalg.cholesky(0.5 * (precision + precision.T))
    except np.linalg.LinAlgError as err:
        raise ValueError("prior_precision must be positive definite.") from err
    T = np.eye(n_design)
    T[:size, :size] = np.linalg.solve(factor.T, np.eye(size))
    return T


def prior_shape(prior_precision, n_design, n_features):
    """Dense prior shape ``Lambda`` over the columns of the design."""
    T = prior_whitening(prior_precision, n_design, n_features)
    if T is None:
        return np.eye(n_design)
    if T.ndim == 1:
        return np.diag(1.0 / (T * T))
    inverse = np.linalg.inv(T)
    return inverse.T @ inverse


def _from_whitened(T, M):
    """``T @ M`` for the map of :func:`prior_whitening`."""
    if T is None:
        return M
    return T[:, None] * M if T.ndim == 1 else T @ M


class POPSRegression(BaseEstimator):
    """Bayesian regression for low-noise data with misspecification uncertainty.

    Fits a linear model by Bayesian ridge regression, then estimates weight
    uncertainties accounting for model misspecification using the POPS
    (Pointwise Optimal Parameter Sets) algorithm [1]_. Unlike standard
    Bayesian regression, the aleatoric noise precision ``alpha_`` is not
    used for predictions, as it should be negligible in the low-noise regime.

    Standard Bayesian regression can only estimate epistemic and aleatoric
    uncertainties. In the low-noise limit, the weight uncertainties
    ``sigma_`` are significantly underestimated as they only account for
    epistemic uncertainties that decay with increasing data. POPS corrects
    this by estimating misspecification uncertainty from pointwise optimal
    parameter sets.

    Several targets that share the design, ``y`` of shape ``(n_samples,
    n_targets)``, are fitted jointly, for example the rows of a linear
    dynamical model ``z_next = M @ phi(z, u)``. The targets share the prior,
    the noise model and therefore the metric ``A`` of the mean fit (their
    precisions are pooled, or fixed by ``ridge``). The pointwise correction
    of a training point then moves all targets together, as the smallest
    change after which the model passes through every target of that point,
    and the posterior is one hypercube over the stacked coefficients
    ``vec(coef_)`` of all targets (target-major), not a product of
    per-target hypercubes.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of evidence iterations of the Bayesian ridge fit.

    tol : float, default=1e-3
        Convergence threshold. Stop the evidence iterations once the
        coefficient vector has converged.

    alpha_1 : float, default=1e-6
        Shape parameter for the Gamma distribution prior over ``alpha_``.

    alpha_2 : float, default=1e-6
        Inverse scale (rate) parameter for the Gamma distribution prior
        over ``alpha_``.

    lambda_1 : float, default=1e-6
        Shape parameter for the Gamma distribution prior over ``lambda_``.

    lambda_2 : float, default=1e-6
        Inverse scale (rate) parameter for the Gamma distribution prior
        over ``lambda_``.

    alpha_init : float, default=None
        Initial value for ``alpha_`` (precision of the noise). If None,
        ``alpha_init`` is ``1 / Var(y)``.

    lambda_init : float, default=None
        Initial value for ``lambda_`` (precision of the weights). If None,
        ``lambda_init`` is 1.

    fit_intercept : bool, default=False
        Whether to fit an intercept. If True, a constant column is appended
        to X (rather than centering) so that the intercept participates in
        the POPS posterior estimation.

    ridge : float, default=None
        Fixed ratio ``lambda_ / alpha_`` of the mean fit, which is then the
        ridge regression ``argmin ||y - X coef||^2 + ridge ||coef||^2`` (with
        ``prior_precision``, ``ridge * coef^T Lambda coef``). No evidence
        iteration runs: ``alpha_`` is the MacKay noise precision of the
        residuals of this fit and ``lambda_ = ridge * alpha_``. ``None``
        learns both precisions by evidence maximization.

    prior_precision : array-like of shape (n_features,) or \
            (n_features, n_features), default=None
        Shape ``Lambda`` of the Gaussian prior ``N(0, (lambda_ *
        Lambda)^-1)`` on the coefficients: a vector for a diagonal
        ``Lambda`` or a symmetric positive definite matrix. Only the shape
        is fixed; the overall scale ``lambda_`` is still learned by evidence
        maximization, so ``Lambda`` and ``c * Lambda`` give the same fit.
        The mean fit, the leverages and the pointwise corrections all use
        the metric ``A = X^T X + (lambda_ / alpha_) * Lambda``. ``None`` is
        the isotropic prior ``Lambda = I``. With ``fit_intercept=True`` the
        intercept keeps unit prior precision, unless ``Lambda`` is given
        with one more row (and column) for it.

    mode_threshold : float, default=1e-8
        Eigenvalue threshold (relative to max) for determining the effective
        dimensionality of the POPS posterior. Eigenvalues below
        ``mode_threshold * max_eigenvalue`` are discarded.

    resample_density : float, default=1.0
        Number of resampled points per training point. The actual number of
        samples is ``max(100, int(resample_density * n_samples))``.

    resampling_method : {'uniform', 'sobol', 'latin', 'halton'}, \
            default='uniform'
        Quasi-random sampling method for generating points within the
        POPS hypercube posterior.

    percentile_clipping : float, default=0.0
        Percentile to clip from each end when determining hypercube bounds.
        The hypercube spans the ``[percentile_clipping,
        100 - percentile_clipping]`` range. Should be between 0 and 50.

    leverage_regularization : float, default=1e-3
        Regularization of the leverage in the pointwise corrections, in
        units of the mean leverage of the training points. The correction
        of point ``i`` is ``e_i A^-1 x_i / (h_i + eps)`` with ``eps =
        leverage_regularization * mean(h)``: the exact pointwise fit
        (``eps = 0``) divides by the leverage ``h_i`` alone and is unstable
        for points the model is barely sensitive to. With ``eps > 0`` the
        correction is the minimizer of ``(x_i @ d - e_i)**2 / eps + d^T A
        d``, so the pointwise fit is satisfied up to the relative residual
        ``eps / (h_i + eps)``: about ``leverage_regularization`` for a
        typical point, and a large fraction only for points of leverage
        a thousand times below the mean at the default. ``0.0`` recovers
        the exact pointwise fit. The corrections of low-leverage points can
        dominate the posterior, so on designs whose leverages span many
        orders of magnitude this parameter sets its width: see
        ``leverage_``.

    minimum_relative_correction : float, default=1e-2
        Relative threshold for training point selection, in units of the
        root-mean-square Euclidean norm of the pointwise corrections. Only
        training points with ``||d_i|| >= minimum_relative_correction *
        rms(||d||)`` enter the singular decomposition and bounds of the
        POPS posterior; the others are too small to affect either. Use
        ``0.0`` to keep every training point. If no point passes the
        threshold, all points are used.

    include_center : bool, default=False
        Extend the hypercube bounds to contain the posterior mean ``coef_``,
        that is, the lower bound along every direction is at most 0 and the
        upper bound at least 0.

    posterior : {'hypercube', 'ensemble'} or None, default='hypercube'
        Form of the POPS parameter posterior:

        - ``'hypercube'``: fit an axis-aligned box in PCA space (default).
        - ``'ensemble'``: use raw pointwise corrections directly.
        - ``None``: fit the mean only; no pointwise correction or posterior
          is computed (``predict`` then returns only the mean and the
          epistemic standard deviation).

        The uniform-ellipsoid posterior, with its empirical-Bayes and PAC
        regularizations, is :class:`~popsregression.POPSEllipseRegression`.

    random_state : int, RandomState instance or None, default=None
        Seed for the posterior resampling. ``None`` uses the global NumPy
        random state for ``'uniform'`` resampling and fresh entropy for the
        quasi-random methods.

    minimum_relative_error : float, default='deprecated'
        Deprecated. Training points used to be selected by the size of
        their residual; they are now selected by the size of their
        pointwise correction through ``minimum_relative_correction``, to
        which a value passed here is forwarded with a
        :class:`FutureWarning`.

    Attributes
    ----------
    In the shapes below, ``n_parameters`` is ``n_features``, plus one with
    ``fit_intercept=True`` (the intercept is the last parameter of every
    target). With several targets, vectors over all coefficients are
    target-major: entry ``j * n_parameters + f`` is parameter ``f`` of
    target ``j``.

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Coefficients of the regression model (posterior mean).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in the decision function. Set to 0.0 if
        ``fit_intercept=False``.

    alpha_ : float
        Estimated precision of the noise (pooled over the targets). Not used
        for prediction.

    lambda_ : float
        Estimated precision of the weights (the scale of the prior).

    sigma_ : ndarray of shape (n_parameters, n_parameters)
        Estimated epistemic variance-covariance matrix of the weights of
        every target; the targets are independent under it. If
        ``fit_intercept=True`` the last row and column are the intercept.

    misspecification_sigma_ : ndarray of shape \
            (n_targets * n_parameters, n_targets * n_parameters)
        Estimated misspecification variance-covariance matrix from POPS.

    posterior_samples_ : ndarray of shape \
            (n_targets * n_parameters, n_posterior_samples)
        Samples from the POPS posterior, representing plausible weight
        perturbations.

    hypercube_support_ : ndarray of shape (n_targets * n_parameters, n_modes)
        Orthonormal directions of the hypercube (only for
        ``posterior='hypercube'``). A posterior sample is
        ``hypercube_support_ @ nu`` with ``nu`` inside ``hypercube_bounds_``.

    hypercube_bounds_ : tuple of two ndarrays of shape (n_modes,)
        Lower and upper coordinates of the hypercube along every direction.

    pointwise_corrections_ : ndarray of shape \
            (n_samples, n_targets * n_parameters)
        Pointwise corrections of every training point (lazy; computed on
        access).

    leverage_ : ndarray of shape (n_samples,)
        Leverage of every training point in the mean fit.

    selected_ : ndarray of shape (n_samples,), dtype=bool
        Training points that entered the POPS posterior.

    n_iter_ : int
        The actual number of iterations to reach convergence (0 with a fixed
        ``ridge``).

    n_targets_ : int
        Number of targets; 1 for a one-dimensional ``y``.

    n_features_in_ : int
        Number of features seen during ``fit``.

    See Also
    --------
    popsregression.POPSEllipseRegression : Uniform-ellipsoid POPS posterior
        with an exact predictive density.

    References
    ----------
    .. [1] Swinburne, T.D. and Perez, D. (2025).
           "Parameter uncertainties for imperfect surrogate models in the
           low-noise regime."
           Machine Learning: Science and Technology, 6, 015008.
           :doi:`10.1088/2632-2153/ad9fce`

    Examples
    --------
    >>> import numpy as np
    >>> from popsregression import POPSRegression
    >>> rng = np.random.RandomState(0)
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 0.01 * rng.randn(4)
    >>> reg = POPSRegression()
    >>> reg.fit(X, y)
    POPSRegression()
    >>> reg.predict(np.array([[3, 5]]))  # doctest: +ELLIPSIS
    array([...])
    """

    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        fit_intercept=False,
        ridge=None,
        prior_precision=None,
        mode_threshold=1.0e-8,
        resample_density=1.0,
        resampling_method="uniform",
        percentile_clipping=0.0,
        leverage_regularization=1.0e-3,
        minimum_relative_correction=1.0e-2,
        include_center=False,
        posterior="hypercube",
        random_state=None,
        minimum_relative_error="deprecated",
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.fit_intercept = fit_intercept
        self.ridge = ridge
        self.prior_precision = prior_precision
        self.mode_threshold = mode_threshold
        self.resample_density = resample_density
        self.resampling_method = resampling_method
        self.percentile_clipping = percentile_clipping
        self.leverage_regularization = leverage_regularization
        self.minimum_relative_correction = minimum_relative_correction
        self.include_center = include_center
        self.posterior = posterior
        self.random_state = random_state
        self.minimum_relative_error = minimum_relative_error

    def _validate(self):
        check_scalar("max_iter", self.max_iter, low=1)
        check_scalar("tol", self.tol, low=0.0)
        for name in ("alpha_1", "alpha_2", "lambda_1", "lambda_2"):
            check_scalar(name, getattr(self, name), low=0.0)
        if self.ridge is not None:
            check_scalar("ridge", self.ridge, low=0.0, low_open=True)
        check_scalar("mode_threshold", self.mode_threshold, low=0.0, low_open=True)
        check_scalar("resample_density", self.resample_density, low=0.0, low_open=True)
        check_option(
            "resampling_method", self.resampling_method, {"uniform", *_RESAMPLING}
        )
        check_scalar(
            "percentile_clipping", self.percentile_clipping, low=0.0, high=50.0
        )
        check_scalar("leverage_regularization", self.leverage_regularization, low=0.0)
        check_option("include_center", self.include_center, {True, False})
        check_option("posterior", self.posterior, {"hypercube", "ensemble", None})
        threshold = self.minimum_relative_correction
        if not isinstance(self.minimum_relative_error, str):
            warnings.warn(
                (
                    "'minimum_relative_error' is deprecated: training points are "
                    "now selected by the norm of their pointwise correction. Its "
                    "value is forwarded to 'minimum_relative_correction'."
                ),
                FutureWarning,
                stacklevel=3,
            )
            threshold = self.minimum_relative_error
        return check_scalar("minimum_relative_correction", threshold, low=0.0)

    def fit(self, X, y, sample_weight=None):
        """Fit the POPS regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Several targets are fitted jointly (see above).

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. They weight the mean fit
            and the leverage regularization; points of zero weight do not
            contribute to the POPS posterior.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_targets(X, y)
        weights = check_sample_weight(sample_weight, X.shape[0])
        self.n_features_in_ = X.shape[1]
        self.n_targets_ = 1 if y.ndim == 1 else y.shape[1]
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        coef = self._fit_hypercube(X, y, weights)
        if self.fit_intercept:
            self.coef_ = coef[..., :-1]
            self.intercept_ = float(coef[-1]) if y.ndim == 1 else coef[:, -1]
        else:
            self.coef_ = coef
            self.intercept_ = 0.0 if y.ndim == 1 else np.zeros(self.n_targets_)

        if self.posterior is None:
            return self
        if self.posterior == "ensemble":
            samples = self.pointwise_corrections_[self.selected_].T
            sigma = samples @ samples.T / samples.shape[1]
        else:
            samples, sigma = self._sample_hypercube()
        self.posterior_samples_, self.misspecification_sigma_ = samples, sigma
        return self

    def _fit_hypercube(self, X, y, weights):
        """Mean fit, pointwise corrections and the PCA-aligned box.

        ``X`` is the full design (any intercept column included). Returns
        the coefficients of that design, ``(n_parameters,)`` or
        ``(n_targets, n_parameters)``; no posterior samples are drawn.
        """
        threshold = self._validate()
        hyper = jnp.array([self.alpha_1, self.alpha_2, self.lambda_1, self.lambda_2])
        # A prior shape Lambda is an isotropic prior in the coordinates
        # phi = T^-1 theta: fit there, then map back.
        n_features = getattr(self, "n_features_in_", X.shape[1])
        T = prior_whitening(self.prior_precision, X.shape[1], n_features)
        if T is not None:
            X = X * T if T.ndim == 1 else X @ T
        fit = bayesian_ridge(
            X,
            y,
            weights,
            as_nan(self.alpha_init),
            as_nan(self.lambda_init),
            hyper,
            float(self.tol),
            int(self.max_iter),
            as_nan(self.ridge),
        )
        self.alpha_ = float(fit.alpha)
        self.lambda_ = float(fit.lam)
        self.n_iter_ = int(fit.n_iter)
        covariance = np.asarray(scaled_covariance(fit)) / self.alpha_
        self.sigma_ = _from_whitened(T, _from_whitened(T, covariance).T)
        coef = np.asarray(fit.coef)
        coef = _from_whitened(T, coef[:, None] if coef.ndim == 1 else coef).T
        coef = coef[0] if y.ndim == 1 else coef
        if self.posterior is None:
            return coef

        Y = y[:, None] if y.ndim == 1 else y
        corrections = _Corrections(
            X, Y, weights, fit, float(self.leverage_regularization)
        )
        basis = _from_whitened(T, np.asarray(fit.V))
        keep, eigenvalues, eigenvectors, low, high = _select_and_bound(
            corrections,
            basis,
            weights > 0,
            float(threshold),
            float(self.percentile_clipping),
        )
        self.leverage_ = np.asarray(corrections.leverage)
        self.selected_ = np.asarray(keep)
        self._basis = basis
        self._corrections = corrections

        eigenvalues = np.asarray(eigenvalues)
        modes = eigenvalues > self.mode_threshold * eigenvalues.max()
        low, high = np.asarray(low)[modes], np.asarray(high)[modes]
        if self.include_center:
            low, high = np.minimum(low, 0.0), np.maximum(high, 0.0)
        self._hypercube_support = np.asarray(eigenvectors)[:, modes]
        self._hypercube_bounds = [low, high]
        return coef

    @property
    def pointwise_corrections_(self):
        """Pointwise corrections of every training point, one per row.

        With several targets, row ``i`` is the target-major ``vec(D_i)``.
        """
        check_is_fitted(self, "_corrections")
        return self._corrections.in_parameters(self._basis)

    @property
    def hypercube_support_(self):
        """Orthonormal directions of the hypercube, one per column."""
        check_is_fitted(self, "_hypercube_support")
        return self._hypercube_support

    @property
    def hypercube_bounds_(self):
        """``(lower, upper)`` coordinates of the hypercube along its directions."""
        check_is_fitted(self, "_hypercube_bounds")
        low, high = self._hypercube_bounds
        return low, high

    def _sample_hypercube(self, size=None, resampling_method=None):
        """Sample points from the fitted POPS hypercube.

        Parameters
        ----------
        size : int, optional
            Number of samples. If None, determined by ``resample_density``.

        resampling_method : str, optional
            Override the instance's resampling method.

        Returns
        -------
        samples : ndarray of shape (n_targets * n_parameters, n_samples)
            Resampled weight perturbations.

        sigma : ndarray of shape \
                (n_targets * n_parameters, n_targets * n_parameters)
            Misspecification covariance estimated from the samples.
        """
        method = (
            self.resampling_method if resampling_method is None else resampling_method
        )
        low, high = self._hypercube_bounds
        if size is None:
            size = int(self.resample_density * self.selected_.size)
        n_resample = max(size, 100)

        # random_state=None keeps NumPy's global state for 'uniform' and the
        # sampler's own entropy for the quasi-random methods.
        seed = self.random_state
        if method == "uniform":
            unit = check_random_state(seed).uniform(size=(low.size, n_resample))
        else:
            if method == "sobol":
                n_resample = 2 ** int(np.log2(n_resample))
            unit = _RESAMPLING[method](d=low.size, seed=seed).random(n_resample).T

        samples = self._hypercube_support @ (
            low[:, None] + (high - low)[:, None] * unit
        )
        return samples, samples @ samples.T / samples.shape[1]

    def _design(self, X):
        X = check_X(X, self.n_features_in_)
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def predict(
        self,
        X,
        return_std=False,
        return_bounds=False,
        return_epistemic_std=False,
    ):
        """Predict using the POPS regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        return_std : bool, default=False
            If True, return the combined (misspecification + epistemic)
            standard deviation.

        return_bounds : bool, default=False
            If True, return the max and min predictions over the POPS
            posterior samples.

        return_epistemic_std : bool, default=False
            If True, return the epistemic-only standard deviation
            (from ``sigma_``, excluding misspecification).

        Returns
        -------
        Every array has shape ``(n_samples,)`` for a one-dimensional ``y``
        and ``(n_samples, n_targets)`` otherwise.

        y_mean : ndarray
            Predicted mean values.

        y_std : ndarray
            Combined standard deviation. Only returned if
            ``return_std=True``.

        y_max : ndarray
            Upper bound from posterior samples. Only returned if
            ``return_bounds=True``.

        y_min : ndarray
            Lower bound from posterior samples. Only returned if
            ``return_bounds=True``.

        y_epistemic_std : ndarray
            Epistemic-only standard deviation. Only returned if
            ``return_epistemic_std=True``.
        """
        check_is_fitted(self)
        if self.posterior is None and (return_std or return_bounds):
            raise ValueError(
                "return_std and return_bounds need a POPS posterior; this model "
                "was fitted with posterior=None."
            )
        X = self._design(X)
        y_mean = X[:, : self.n_features_in_] @ self.coef_.T + self.intercept_
        result = [y_mean]
        single = self.coef_.ndim == 1
        n_parameters = X.shape[1]

        def blocks(matrix):
            """Per-target diagonal blocks of a target-major covariance."""
            if single:
                return [matrix]
            return [
                matrix[j * n_parameters : (j + 1) * n_parameters][
                    :, j * n_parameters : (j + 1) * n_parameters
                ]
                for j in range(self.n_targets_)
            ]

        def stack(columns):
            return columns[0] if single else np.column_stack(columns)

        if return_std or return_epistemic_std:
            epistemic_var = np.einsum("ij,ij->i", X @ self.sigma_, X)
            if not single:
                epistemic_var = np.repeat(epistemic_var[:, None], self.n_targets_, 1)
        if return_std:
            misspecification_var = stack(
                [
                    np.einsum("ij,ij->i", X @ block, X)
                    for block in blocks(self.misspecification_sigma_)
                ]
            )
            result.append(np.sqrt(misspecification_var + epistemic_var))
        if return_bounds:
            samples = self.posterior_samples_
            extrema = [
                _sample_extrema(X, samples[j * n_parameters : (j + 1) * n_parameters])
                for j in range(self.n_targets_)
            ]
            upper = stack([e[0] for e in extrema])
            lower = stack([e[1] for e in extrema])
            result.extend([y_mean + upper, y_mean + lower])
        if return_epistemic_std:
            result.append(np.sqrt(epistemic_var))

        if len(result) == 1:
            return result[0]
        return tuple(result)


def _sample_extrema(X, samples, block=4096):
    """Row-wise max and min of ``X @ samples``, in blocks of posterior samples."""
    upper = np.full(X.shape[0], -np.inf)
    lower = np.full(X.shape[0], np.inf)
    for start in range(0, samples.shape[1], block):
        values = X @ samples[:, start : start + block]
        np.maximum(upper, values.max(axis=1), out=upper)
        np.minimum(lower, values.min(axis=1), out=lower)
    return upper, lower
