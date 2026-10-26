# Getting Started {#getting_started}

First, we define our model. Let's use logistic regression, by setting the
loss to `"logistic"`.

```cpp
#include <slope/slope.h>

slope::Slope model;

model.setLoss("logistic");
```

Next, we set the data matrix `x` and the response vector `y`. Here we use some
toy data.

```cpp
Eigen::MatrixXd x(3, 2);
Eigen::VectorXd y(3);

x << 1.1, 2.3, 0.2, 1.5, 0.5, 0.2;
y << 0, 1, 0;
```

Finally, we call the `path()` method to fit the full SLOPE path.

```cpp
auto res = model.path(x, y);
```

Now we can retrieve the coefficients by calling `res.getCoefs()`.

## Observation weights

Pass one observation weight per row to fit a weighted model. Weights must be
finite and nonnegative, and at least one weight must be positive. An empty
weight vector gives every observation the same weight. The weighted objective
is

\f[
  \frac{1}{\sum_i w_i}\sum_i w_i\ell_i(\beta_0,\beta)
  + \alpha\sum_j\lambda_j|\beta|_{(j)}.
\f]

Multiplying every weight by the same positive number leaves the fit unchanged.
Zero-weight rows do not contribute to the objective or to computed feature
centers and scales. Inputs must still satisfy the usual data validation rules.
Standardization uses the weighted mean and population standard deviation.
Manual centers and scales retain their supplied values. L1 and L2 feature
scales use weighted norms with weights normalized to the number of
positive-weight rows. Minima, ranges, and maximum absolute values use only
positive-weight rows. Integer weights reproduce repeated observations with
default standardization; L1 and L2 scales do not promise that equivalence.
Multinomial classes are inferred from all supplied responses, including
zero-weight rows.

Weights are the final optional argument to `fit()` and `path()`, after the
interrupt callback:

```cpp
Eigen::VectorXd weights(3);
weights << 1.0, 2.0, 0.5;
Eigen::ArrayXd lambda;
double alpha = 0.1;

auto fit = model.fit(
  x, y, alpha, lambda, slope::defaultInterruptChecker, weights);
auto path = model.path(
  x, y, {}, lambda, slope::defaultInterruptChecker, weights);
```

Supply weights again when relaxing a fit. For a single fit, weights follow the
optional intercept and coefficient warm starts. Pass the original predictors,
before normalization; relaxation uses just-in-time normalization and does not
modify them. For a path, weights follow `gamma`:

```cpp
auto relaxed_fit = model.relax(fit, x, y, 0.0, {}, {}, weights);
auto relaxed_path = model.relax(path, x, y, 0.0, weights);
```

Cross-validation accepts weights after its configuration:

```cpp
#include <slope/cv.h>

slope::CvConfig config;
config.n_folds = 3;
auto cv = slope::crossValidate(model, x, y, config, weights);
```

Each training and validation subset normalizes its own weights. Validation
metrics use observation weights, and the reported mean and standard error
continue to give each fold equal weight. Every training and validation subset
must have positive total weight.

The same row weights apply to every response column, including multiresponse
quadratic models and multinomial models.
Automatic noise-based alpha estimation and automatic Gaussian lambda
calibration do not support nonuniform weights. Use the default BH sequence,
another supported sequence, or an explicit lambda vector instead.

The Julia bindings retain the original entry points and add
`fit_slope_dense_weighted`, `fit_slope_sparse_weighted`,
`cv_slope_dense_weighted`, and `cv_slope_sparse_weighted`. These functions take
the same arguments as their original counterparts, followed by a
`Vector{Float64}` of observation weights.
