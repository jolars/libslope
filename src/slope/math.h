/**
 * @internal
 * @file
 * @brief Mathematical support functions for the slope package.
 */

#pragma once

#include <Eigen/Core>
#include <numeric>
#include <vector>

namespace slope {

/**
 * @brief Returns the sign of a given value.
 *
 * This function determines the sign of the input value by comparing it to zero.
 * It returns -1 if the value is negative, 0 if the value is zero, and 1 if the
 * value is positive.
 *
 * @tparam T The type of the input value.
 * @param val The value for which the sign needs to be determined.
 * @return -1 if the value is negative, 0 if the value is zero, and 1 if the
 * value is positive.
 */
template<typename T>
int
sign(T val)
{
  return (T(0) < val) - (val < T(0));
}

/**
 * Calculates the cumulative sum of the elements in the input array.
 *
 * @tparam T The type of the input array.
 * @param x The input array.
 * @return An Eigen::ArrayXd containing the cumulative sum of the elements in
 * the input array.
 */
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

/**
 * Calculates the sigmoid function for the given input.
 *
 * The sigmoid function is defined as 1 / (1 + exp(-x)).
 *
 * @tparam T The type of the input.
 * @param x The input value.
 * @return The result of the sigmoid function.
 */
template<typename T>
T
sigmoid(const T& x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

/**
 * Returns the value of x clamped between the specified lower and upper bounds.
 *
 * @tparam T the type of the values being clamped
 * @param x the value to be clamped
 * @param lo the lower bound
 * @param hi the upper bound
 * @return the clamped value of x
 */
template<typename T>
T
clamp(const T& x, const T& lo, const T& hi)
{
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * LogSumExp
 *
 * @param a A matrix
 * @return \f$\log(\sum_i \exp(a_i))\f$
 */
Eigen::VectorXd
logSumExp(const Eigen::MatrixXd& a);

/**
 * Softmax
 *
 * Computes the softmax function for the given input matrix.
 *
 * @param a A matrix
 * @return \f$\exp(a) / \sum_i \exp(a_i)\f$
 */
Eigen::MatrixXd
softmax(const Eigen::MatrixXd& x);

/**
 * Computes the gradient of the objective with respect to \f(\beta\f).
 *
 * @tparam T The type of the input matrix.
 * @param x The input matrix.
 * @param residual The residual vector.
 * @param x_centers The vector of center values for each column of x.
 * @param x_scales The vector of scale values for each column of x.
 * @param standardize Flag indicating whether to standardize the gradient.
 * @return The computed gradient vector.
 */
template<typename T>
Eigen::MatrixXd
linearPredictor(const T& x,
                const Eigen::VectorXd& beta0,
                const Eigen::MatrixXd& beta,
                const Eigen::VectorXd& x_centers,
                const Eigen::VectorXd& x_scales,
                const bool standardize_jit,
                const bool intercept)
{
  int n = x.rows();
  int p = x.cols();
  int m = beta.cols();

  Eigen::MatrixXd eta(n, m);

  if (standardize_jit) {
    for (int k = 0; k < m; ++k) {
      eta.col(k) = x * beta.col(k).cwiseQuotient(x_scales);
      eta.col(k).array() -= x_centers.cwiseQuotient(x_scales).dot(beta.col(k));
    }
  } else {
    eta = x * beta;
  }

  if (intercept) {
    eta.rowwise() += beta0.transpose();
  }

  return eta;
}

/**
 * Computes the gradient of the objective with respect to \f(\beta\f).
 *
 * @tparam T The type of the input matrix.
 * @param x The input matrix.
 * @param residual The residual vector.
 * @param x_centers The vector of center values for each column of x.
 * @param x_scales The vector of scale values for each column of x.
 * @param standardize Flag indicating whether to standardize the gradient.
 * @return The computed gradient vector.
 */
template<typename T>
Eigen::MatrixXd
computeGradient(const T& x,
                const Eigen::MatrixXd& residual,
                const Eigen::VectorXd& x_centers,
                const Eigen::VectorXd& x_scales,
                const Eigen::VectorXd& w,
                const bool standardize_jit)
{
  const int n = x.rows();
  const int p = x.cols();
  const int m = residual.cols();

  Eigen::MatrixXd weighted_residual(residual.rows(), residual.cols());

  for (int k = 0; k < m; ++k) {
    weighted_residual.col(k) = residual.col(k).cwiseProduct(w);
  }

  if (standardize_jit) {
    Eigen::MatrixXd gradient(p, m);

    for (int k = 0; k < m; ++k) {
      double wr_sum = weighted_residual.col(k).sum();

      for (int j = 0; j < p; ++j) {
        gradient(j, k) =
          -(x.col(j).dot(weighted_residual.col(k)) - x_centers(j) * wr_sum) /
          (x_scales(j) * n);
      }
    }

    return gradient;
  }

  // No standardization or already standardized in place
  return -(x.transpose() * weighted_residual) / n;
}

/**
 * Computes the gradient of the objective with respect to \f(\beta\f).
 *
 * @tparam T The type of the input matrix.
 * @param x The input matrix.
 * @param residual The residual vector.
 * @param x_centers The vector of center values for each column of x.
 * @param x_scales The vector of scale values for each column of x.
 * @param standardize Flag indicating whether to standardize the gradient.
 * @return The computed gradient vector.
 */
template<typename T>
Eigen::MatrixXd
computeGradientOffset(const T& x,
                      const Eigen::VectorXd& offset,
                      const Eigen::VectorXd& x_centers,
                      const Eigen::VectorXd& x_scales,
                      const bool standardize_jit)
{
  const int n = x.rows();
  const int p = x.cols();
  const int m = offset.rows();

  Eigen::MatrixXd out(p, m);

  if (standardize_jit) {
    // This is not necessary if x_centers are already the means, but
    // it is included in case later on we want to use something
    // other than means for the centers.
    for (int k = 0; k < m; ++k) {
      for (int j = 0; j < p; ++j) {
        out(j) = offset(k) * (x.col(j).sum() / n - x_centers(j)) / x_scales(j);
      }
    }
  } else {
    // return x.colwise().mean().array() * offset;
    for (int k = 0; k < m; ++k) {
      for (int j = 0; j < p; ++j) {
        out(j, k) = offset(k) * x.col(j).sum() / n;
      }
    }
  }

  // No standardization or already standardized in place
  return out;
}

} // namespace slope
