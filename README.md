# SLOPE <img src='https://raw.githubusercontent.com/jolars/libslope/refs/heads/main/assets/slope-logo.png' align="right" height="139" />

[![CI](https://github.com/jolars/libslope/actions/workflows/ci.yaml/badge.svg)](https://github.com/jolars/libslope/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/jolars/libslope/graph/badge.svg?token=y0mJN9eqYr)](https://codecov.io/gh/jolars/libslope)

Libslope is a C++ library for Sorted L-One Penalized Estimation (SLOPE). Its main
purpose is to serve as a backend for R and Python packages, but it can also be
used as a standalone library in the off-chance that you want to fit your models
entirely through C++.

## Getting Started

First, we define our model. Let's use logistic regression, by setting the
loss to `"logistic"`.

```cpp
#include "slope.h"

Slope::Model model;

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

## Dependencies

### Building

- A C++17 compiler
- CMake 3.15 or later
- Eigen 3.4 or later

### Documentation

- Doxygen

### Testing

- Catch2
