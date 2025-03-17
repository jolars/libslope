# SLOPE {#mainpage}

SLOPE is a C++ library for Sorted L-One Penalized Estimation (SLOPE). Its main
purpose is to serve as a backend for higher-level language packages in R and
Python, but it can also be used as a standalone library in the off-chance that
you want to fit your models entirely via C++.

SLOPE solves the following optimization problem:
\f[
\min_{\beta} \frac{1}{n} f(X\beta + \beta_0) + \alpha \sum_{j=1}^p \lambda_j |\beta_{(j)}|,
\f]

where \f(\beta_{(j)}\f) is the \f(j\f)th element of non-increasingly sorted vector
of \f(\beta\f), and \f(f\f) is a convex loss function. SLOPE is a generalization of
the lasso.

See the following sections for more information:
- @subpage getting_started
- @subpage dependencies


