/**
 * @internal
 * @file
 * @brief Mathematical support functions for the slope package.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
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
 * The logit
 *
 * The logit function is defined as \f$\log(\frac{x}{1 - x})\f$.
 *
 * @tparam T The type of the input.
 * @param x The input value.
 * @return The result of the logit function.
 */
template<typename T>
T
logit(const T& x)
{
  assert(x > 0 && x < 1 && "Input must be in (0, 1)");

  return std::log(x) - std::log1p(-x);
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
                const std::vector<int>& active_set,
                const Eigen::VectorXd& beta0,
                const Eigen::MatrixXd& beta,
                const Eigen::VectorXd& x_centers,
                const Eigen::VectorXd& x_scales,
                const bool standardize_jit,
                const bool intercept)
{
  int n = x.rows();
  int m = beta.cols();

  Eigen::MatrixXd eta = Eigen::MatrixXd::Zero(n, m);

  if (standardize_jit) {
    for (int k = 0; k < m; ++k) {
      for (const auto& j : active_set) {
        eta.col(k) += x.col(j) * beta(j, k) / x_scales(j);
        eta.col(k).array() -= beta(j, k) * x_centers(j) / x_scales(j);
      }
    }
  } else {
    for (int k = 0; k < m; ++k) {
      for (const auto& j : active_set) {
        eta.col(k) += x.col(j) * beta(j, k);
      }
    }
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
void
updateGradient(Eigen::MatrixXd& gradient,
               const T& x,
               const Eigen::MatrixXd& residual,
               const std::vector<int>& active_set,
               const Eigen::VectorXd& x_centers,
               const Eigen::VectorXd& x_scales,
               const Eigen::VectorXd& w,
               const bool standardize_jit)
{
  const int n = x.rows();
  const int p = x.cols();
  const int m = residual.cols();

  assert(gradient.rows() == p && gradient.cols() == m &&
         "Gradient matrix has incorrect dimensions");

  Eigen::MatrixXd weighted_residual(n, m);

  for (int k = 0; k < m; ++k) {
    weighted_residual.col(k) = residual.col(k).cwiseProduct(w);
  }

  if (standardize_jit) {
    for (int k = 0; k < m; ++k) {
      double wr_sum = weighted_residual.col(k).sum();

      for (const auto& j : active_set) {
        gradient(j, k) =
          (x.col(j).dot(weighted_residual.col(k)) - x_centers(j) * wr_sum) /
          (x_scales(j) * n);
      }
    }
  } else {
    for (int k = 0; k < m; ++k) {
      for (const auto& j : active_set) {
        gradient(j, k) = x.col(j).dot(weighted_residual.col(k)) / n;
      }
    }
  }
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
void
offsetGradient(Eigen::MatrixXd& gradient,
               const T& x,
               const Eigen::VectorXd& offset,
               const std::vector<int>& active_set,
               const Eigen::VectorXd& x_centers,
               const Eigen::VectorXd& x_scales,
               const bool standardize_jit)
{
  const int n = x.rows();
  const int p = x.cols();
  const int m = offset.rows();

  assert(gradient.rows() == p && gradient.cols() == m &&
         "Gradient matrix has incorrect dimensions");

  if (standardize_jit) {
    // This is not necessary if x_centers are already the means, but
    // it is included in case later on we want to use something
    // other than means for the centers.
    for (int k = 0; k < m; ++k) {
      for (const auto& j : active_set) {
        gradient(j, k) -=
          offset(k) * (x.col(j).sum() / n - x_centers(j)) / x_scales(j);
      }
    }
  } else {
    for (int k = 0; k < m; ++k) {
      for (const auto& j : active_set) {
        gradient(j, k) -= offset(k) * x.col(j).sum() / n;
      }
    }
  }
}

std::vector<int>
setUnion(const std::vector<int>& a, const std::vector<int>& b);

std::vector<int>
setDiff(const std::vector<int>& a, const std::vector<int>& b);

template<typename T>
int
whichMax(const T& x)
{
  return std::distance(x.begin(), std::max_element(x.begin(), x.end()));
}

} // namespace slope
