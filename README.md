popsregression
=================================================

[![tests](https://github.com/POPS-UQ/popsregression/actions/workflows/python-app.yml/badge.svg)](https://github.com/POPS-UQ/popsregression/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/POPS-UQ/popsregression/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/POPS-UQ/popsregression)
[![docs](https://img.shields.io/badge/docs-POPS--UQ.github.io%2Fpopsregression-blue)](https://POPS-UQ.github.io/popsregression)

**popsregression** provides `POPSRegression`, a Bayesian regression method for
low-noise data that accounts for model misspecification uncertainty, and
`POPSEllipseRegression`, its uniform-ellipsoid posterior with empirical-Bayes
and PAC-Bayes options. The numerical backend is [JAX](https://jax.readthedocs.io);
the estimators follow the scikit-learn conventions and work in its pipelines.

**Paper** *Parameter uncertainties for imperfect surrogate models in the low-noise regime*, T. D. Swinburne and D. Perez, [Machine Learning: Science and Technology 6, 015008 (2025)](http://iopscience.iop.org/article/10.1088/2632-2153/ad9fce)

**Documentation** 📖 [POPS-UQ.github.io/popsregression](https://POPS-UQ.github.io/popsregression) — API reference and usage for this Python package

**The POPS method** 🔬 [pops-uq.github.io](https://pops-uq.github.io) — concepts, algorithm, tutorials and the [Julia implementation](https://github.com/POPS-UQ/POPSRegression.jl)

**Try it out!** [online demo from the Kermode group](https://kermodegroup.github.io/demos/regression-demo.html) comparing multiple regression schemes.

## Misspecification-aware Bayesian regression

Standard Bayesian regression (e.g. `BayesianRidge`) estimates epistemic and
aleatoric uncertainties, but ignores model misspecification: errors arising
from a limited model form (see the example below). In the low-noise (weak
aleatoric, near-deterministic) limit, the weight uncertainties (`sigma_`) are
significantly underestimated, as they only capture epistemic uncertainty,
which decays with increasing data. Any remaining error is attributed to
aleatoric noise (`alpha_`), which is erroneous in low-noise settings.

`POPSRegression` efficiently estimates **model misspecification uncertainty**
with the Pointwise Optimal Parameter Sets (POPS) algorithm, which finds the
parameter perturbations that would fit each training point exactly. The result
is wider, more honest uncertainty estimates that cover the true function, even
when the model class cannot represent the target exactly.

The misspecified, near-deterministic regression problem that `POPSRegression`
addresses is particularly relevant to surrogate models in computational
science, such as interatomic potentials or learned dynamical models, where the
optimal surrogate is by construction unable to capture the target function
exactly.

## Example

Fitting a quartic polynomial (5 parameters) to an oscillatory function with
N = 10 (top row) and N = 100 (bottom row) training points. Columns are Bayesian
ridge, the POPS hypercube, the POPS ellipse, and the ellipse with its
empirical-Bayes (Ellipse+EB) and PAC-Bayes (Ellipse+PAC) layers. Every panel
shows the 95.45% and 99.9% intervals of the parameter-only predictive (no
aleatoric term) and reports the fraction of the truth they cover. The Bayesian
ridge uncertainty vanishes with more data, while POPS keeps uncertainty where
the polynomial deviates from the truth.

![Bayesian ridge vs POPS uncertainty](examples/example_polynomial.png)

The figure is produced by [examples/example_polynomial.py](examples/example_polynomial.py);
[examples/README.md](examples/README.md) lists the other studies.

## Installation

```bash
pip install popsregression
```

The development version (JAX backend, several targets):

```bash
pip install git+https://github.com/POPS-UQ/popsregression@jax-integration
```

**Dependencies**: jax >= 0.4.35, scipy >= 1.7.0, numpy >= 1.22.0 (Python >= 3.10).
scikit-learn is not required.

## Quick start

```python
from popsregression import POPSRegression

X_train, X_test, y_train, y_test = ...

# Fit POPSRegression
# fit_intercept=False by default
model = POPSRegression()
model.fit(X_train, y_train)

# Prediction with misspecification & epistemic uncertainty
y_pred, y_std = model.predict(X_test, return_std=True)

# Also return min/max bounds over the posterior
y_pred, y_std, y_max, y_min = model.predict(
    X_test, return_std=True, return_bounds=True
)

# Also return epistemic-only uncertainty separately
y_pred, y_std, y_max, y_min, y_epistemic_std = model.predict(
    X_test,
    return_std=True,
    return_bounds=True,
    return_epistemic_std=True,
)
```

## Several targets

Targets that share one design matrix, `y` of shape `(n_samples, n_targets)`,
are fitted jointly. A typical case is a linear dynamical model, such as a
Koopman or DMD predictor `z_next = M @ phi(z, u)`, whose rows are separate
targets of the same features `phi`.

- **One metric for all targets.** The targets share the prior and the noise
  model, so their precisions are pooled (or fixed by `ridge`) and every target
  has the same regularized Gram matrix `A = X^T X + (lambda_ / alpha_) I`.
- **One correction per training point.** The pointwise correction of point
  `i` is the smallest change of all coefficients after which the model fits
  every target of that point: the rank-one matrix `D_i = e_i (A⁻¹ x_i)^T / h_i`,
  with `e_i` the vector of residuals of point `i`.
- **One hypercube.** The posterior is a single box over the stacked
  coefficients `vec(coef_)` (target-major), so a posterior sample moves all
  targets together, as a training point does. Fitting each target separately
  would instead give a product of independent per-target boxes.

```python
import numpy as np
from popsregression import POPSRegression

# Phi: (n_samples, n_features) features; Z_next: (n_samples, n_targets) targets
model = POPSRegression(
    ridge=1e-6,                       # fixed lambda / alpha instead of evidence maximization
    leverage_regularization=0.0,      # exact pointwise fits
    minimum_relative_correction=0.0,  # every training point enters the box
    include_center=True,              # box contains the posterior mean
    resampling_method="latin",
    random_state=0,
).fit(Phi, Z_next)

M0 = model.coef_                              # (n_targets, n_features)
W = model.hypercube_support_                  # (n_targets * n_features, n_modes), orthonormal
lower, upper = model.hypercube_bounds_        # (n_modes,) each

# A model from the box: vec(M) = vec(M0) + W @ nu with lower <= nu <= upper
nu = lower + np.random.default_rng(1).uniform(size=lower.size) * (upper - lower)
M = M0 + (W @ nu).reshape(M0.shape)

Z_pred, Z_std = model.predict(Phi_test, return_std=True)   # (n_test, n_targets) each
```

The stacked corrections are generated in blocks, so the
`n_samples x (n_targets * n_features)` correction matrix is never held in
memory; `pointwise_corrections_` computes it on access. With one target the
results are those of the single-target fit.

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `posterior` | `'hypercube'` | Posterior form: `'hypercube'` (PCA-aligned box), `'ensemble'` (raw corrections) or `None` (mean fit only); the ellipsoid is `POPSEllipseRegression` (below) |
| `ridge` | `None` | Fixed ratio `lambda_ / alpha_` of the mean fit (ridge regression with this coefficient); `None` learns both precisions by evidence maximization |
| `include_center` | `False` | Extend the hypercube bounds to contain the posterior mean `coef_` |
| `random_state` | `None` | Seed for the posterior resampling; `None` uses the global NumPy state |
| `resampling_method` | `'uniform'` | Sampling method: `'uniform'`, `'sobol'`, `'latin'`, `'halton'` |
| `resample_density` | `1.0` | Number of posterior samples per training point |
| `prior_precision` | `None` | Shape `Λ` (vector or matrix) of the Gaussian prior `N(0, (λΛ)⁻¹)` on the coefficients; the scale `λ` is learned unless `ridge` is set. `None` is the isotropic prior |
| `leverage_regularization` | `0.001` | Regularizes the leverage in the pointwise corrections, `e_i A⁻¹x_i / (h_i + eps)` with `eps` this times the mean leverage; `0.0` is the exact (unstable for low-leverage points) pointwise fit |
| `minimum_relative_correction` | `0.01` | Relative threshold on the correction norm: only points with \|\|dθ_i\|\| >= this times the RMS norm enter the singular decomposition and bounds of the POPS posterior |
| `mode_threshold` | `1e-8` | Eigenvalue threshold for hypercube dimensionality |
| `percentile_clipping` | `0.0` | Percentile to clip from hypercube bounds (0–50) |

The Bayesian ridge parameters of the mean fit (`max_iter`, `tol`, `alpha_1`,
`alpha_2`, `lambda_1`, `lambda_2`, `alpha_init`, `lambda_init`,
`fit_intercept`) are also supported.

> [!WARNING]
> `minimum_relative_error` is deprecated: training points are now selected by
> the norm of their pointwise correction rather than by their residual. A
> value passed to it is forwarded to `minimum_relative_correction` with a
> `FutureWarning`. `leverage_percentile`, `compute_score`, `copy_X` and
> `verbose` have been removed.

## Key attributes (after fitting)

With several targets, vectors over all coefficients are target-major: entry
`j * n_features + f` is coefficient `f` of target `j` (plus one parameter per
target with `fit_intercept=True`).

| Attribute | Description |
|---|---|
| `coef_`, `intercept_` | Regression coefficients (posterior mean) and intercept; `(n_targets, n_features)` and `(n_targets,)` with several targets |
| `sigma_` | Epistemic variance-covariance matrix of the coefficients of every target |
| `misspecification_sigma_` | Misspecification variance-covariance matrix from POPS |
| `posterior_samples_` | Samples from the POPS posterior (perturbations of `vec(coef_)`) |
| `hypercube_support_`, `hypercube_bounds_` | Orthonormal hypercube directions and the lower and upper coordinates along them |
| `alpha_`, `lambda_` | Estimated noise precision (not used for prediction) and weight precision |
| `pointwise_corrections_`, `leverage_`, `selected_` | Pointwise corrections, leverages and the points kept for the posterior |
| `n_targets_` | Number of targets (1 for a one-dimensional `y`) |

## Ellipsoid posteriors: `POPSEllipseRegression`

`POPSEllipseRegression` fits a parameter distribution that is uniform on an
ellipsoid. For a linear model its predictive density at `x` has a closed form,
so the zero-noise generalization error is minimized directly. Predictive
densities, exact quantiles and parameter draws are available in closed form
(for the hierarchical layers, exactly for the stored finite mixture of
hyperparameter draws that represents the predictive). The `regularization`
parameter chooses how the finite-data uncertainty of the ellipsoid itself is
handled:

```python
from popsregression import POPSEllipseRegression

bare = POPSEllipseRegression().fit(X_train, y_train)

# Empirical-Bayes layer (Ellipse+EB): a Laplace hyperposterior over the
# ellipsoid, with its prior centered on the fit. Cheap; its diagnostic_bound_
# is not a PAC bound.
eb = POPSEllipseRegression(regularization="empirical-bayes").fit(X_train, y_train)

# PAC layer (Ellipse+PAC): the prior over the ellipsoid is fixed on a random
# half of the data, and a PAC-Bayes bound on the population log risk is
# evaluated on the other half (and vice versa). y_bounds must be known in
# advance (e.g. from a maximum principle), not read off the data.
pac = POPSEllipseRegression(regularization="PAC", y_bounds=(y_lower, y_upper))
pac.fit(X_train, y_train, groups=case_id)   # groups: rows of one simulator case
pac.certificate_.raw_bound, pac.certificate_.trivial_bound  # bound for the returned predictive

lo, hi = eb.predict_interval(X_test, level=0.9545)   # exact central interval
logp = eb.predict_logpdf(X_test, y_test)
fields = eb.sample_predictions(X_test, 1000)          # joint draws for propagation
```

All predictions refer to parameter uncertainty only; no noise term is added.
The PAC bound certifies the log risk of the predictive density, not interval
coverage. See the
[ellipse documentation](https://POPS-UQ.github.io/popsregression/ellipse/)
and the [glossary](https://POPS-UQ.github.io/popsregression/glossary/) for the
terms used above.

## Comparison methods and paper studies

PACm, PAC²-T, predictive variational inference and a finite
POPS-dictionary weighting ablation are implemented in `examples/comparisons/`,
outside the package. The scripts in `examples/` produce every figure and table
of the accompanying paper; see [examples/README.md](examples/README.md).

## Backend

All numerical kernels (the Bayesian ridge fit, the pointwise corrections, the
ellipsoid optimization, the PAC certification and the mixture predictives) are
written in [JAX](https://jax.readthedocs.io) and compiled on first use.

- The first call for a given problem shape pays a compilation cost of about a
  second; later calls with the same shapes reuse the compiled kernel.
- Importing the package enables JAX double precision
  (`jax_enable_x64`), a process-wide JAX setting.
- With a GPU build of JAX the same code runs on the accelerator.
- Inputs and fitted attributes (`coef_`, `sigma_`, ...) are NumPy arrays.

scikit-learn is not required. The estimators keep the familiar
`fit` / `predict` / `get_params` / `set_params` usage, so they still work
inside scikit-learn pipelines and parameter searches when it is installed.

## Pipeline compatibility

If scikit-learn is installed, both estimators can be used in its pipelines and
hyperparameter search:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe = make_pipeline(
    PolynomialFeatures(degree=4),
    POPSRegression(resampling_method='sobol'),
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

## Documentation

https://POPS-UQ.github.io/popsregression

## Development

The repository is managed with [uv](https://docs.astral.sh/uv/); `uv run`
resolves the pinned environment from `uv.lock` on first use.

```bash
uv run --group test pytest -vsl popsregression examples/comparisons  # tests
uv run --group lint ruff check popsregression examples               # linter
uv run --group lint black --check popsregression examples
uv run --group doc mkdocs serve                    # docs at localhost:8000
cd examples && uv run --extra examples python example_polynomial.py  # a figure
```

Without uv, `pip install -e ".[examples]"` and run the tools directly.

## Citation

If you use this package, please cite

```bibtex
@article{swinburne2025parameter,
  title   = {Parameter uncertainties for imperfect surrogate models in the low-noise regime},
  author  = {Swinburne, Thomas D. and Perez, Danny},
  journal = {Machine Learning: Science and Technology},
  volume  = {6},
  number  = {1},
  pages   = {015008},
  year    = {2025},
  doi     = {10.1088/2632-2153/ad9fce}
}
```

## AI Usage

LLM tools were used to produce fuller documentation and some test cases. All code was reviewed by a human.
The multi-target extension (several targets, `ridge`, `include_center`) was written with an LLM coding assistant.
