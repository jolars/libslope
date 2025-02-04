/**
 * @file
 * @brief Functions for generating regularization sequences for SLOPE.
 * */

#pragma once

#include "math.h"
#include "sorted_l1_norm.h"
#include <Eigen/Sparse>
#include <string>

namespace slope {

/**
 * Generates a sequence of regularization weights for the sorted L1 norm.
 *
 * @param p The number of lambda values to generate (number of features)
 * @param q The false discovery rate (FDR) level or quantile value (typically
 * between 0 and 1)
 * @param type The type of sequence to generate:
 *            - "bh": Benjamini-Hochberg sequence
 *            - "gaussian": Gaussian sequence
 *            - "oscar": Octagonal Shrinkage and Clustering Algorithm for
 * Regression
 * @param n Number of observations (only used for gaussian type)
 * @param theta1 First parameter for OSCAR weights (default: 1.0)
 * @param theta2 Second parameter for OSCAR weights (default: 1.0)
 * @return Eigen::ArrayXd containing the generated lambda sequence in decreasing
 * order
 */
Eigen::ArrayXd
lambdaSequence(const int p,
               const double q,
               const std::string& type,
               const int n = -1,
               const double theta1 = 1.0,
               const double theta2 = 1.0);

/**
 * Computes a sequence of regularization weights for the SLOPE path.
 *
 * @tparam T Matrix type (dense or sparse)
 * @param x The design matrix
 * @param w Sample weights
 * @param z Response variable
 * @param x_centers Column means of x
 * @param x_scales Column scales of x
 * @param penalty The SortedL1Norm penalty object
 * @param path_length Number of points in the regularization path
 * @param alpha_min_ratio Ratio of minimum to maximum alpha (if < 0, defaults
 * based on n > p)
 * @param intercept Whether to fit an intercept
 * @param standardize_jit Whether to standardize features just-in-time
 * @return Eigen::ArrayXd containing the sequence of regularization parameters
 *         from strongest (alpha_max) to weakest (alpha_max * alpha_min_ratio)
 */
template<typename T>
Eigen::ArrayXd
regularizationPath(const T& x,
                   const Eigen::VectorXd& w,
                   const Eigen::VectorXd& z,
                   const Eigen::VectorXd& x_centers,
                   const Eigen::VectorXd& x_scales,
                   const SortedL1Norm& penalty,
                   const int path_length,
                   double alpha_min_ratio,
                   const bool intercept,
                   const bool standardize_jit)
{
  const int n = x.rows();
  const int p = x.cols();

  if (alpha_min_ratio < 0) {
    alpha_min_ratio = n > p ? 1e-4 : 1e-2;
  }

  auto gradient =
    computeGradient(x, z, x_centers, x_scales, w, standardize_jit);

  double alpha_max = penalty.dualNorm(gradient);

  Eigen::ArrayXd alpha(path_length);

  double div = path_length - 1;

  for (int i = 0; i < path_length; ++i) {
    alpha[i] = alpha_max * std::pow(alpha_min_ratio, i / div);
  }

  return alpha;
}

} // namespace slope
