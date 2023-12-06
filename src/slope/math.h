#pragma once

#include <Eigen/Core>
#include <numeric>
#include <vector>

namespace slope {

template<typename T>
int
sign(T val)
{
  return (T(0) < val) - (val < T(0));
}

template<typename T>
Eigen::ArrayXd
cumSum(const T& x)
{
  std::vector<double> cum_sum(x.size());
  std::partial_sum(
    x.data(), x.data() + x.size(), cum_sum.begin(), std::plus<double>());

  Eigen::Map<Eigen::ArrayXd> out(cum_sum.data(), cum_sum.size());

  return out;
}

template<typename T>
T
sigmoid(const T& x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

template<typename T>
T
clamp(const T& x, const T& lo, const T& hi)
{
  return x < lo ? lo : x > hi ? hi : x;
}

// Compute the gradient with respect to the coefficients, accounting for
// possible standardization of x
template<typename T>
Eigen::VectorXd
computeGradient(const T& x,
                const Eigen::VectorXd& residual,
                const Eigen::VectorXd& x_centers,
                const Eigen::VectorXd& x_scales,
                const bool standardize)
{
  const int n = x.rows();
  const int p = x.cols();

  Eigen::VectorXd gradient(p);

  if (standardize) {
    double wr_sum = residual.sum();
    for (int j = 0; j < p; ++j) {
      gradient(j) =
        -(x.col(j).dot(residual) - x_centers(j) * wr_sum) / (x_scales(j) * n);
    }
  } else {
    gradient = -(x.transpose() * residual) / n;
  }

  return gradient;
}

} // namespace slope
