/**
 * @file
 * @brief Functions for generating regularization sequences for SLOPE.
 * */

#pragma once

#include "sorted_l1_norm.h"
#include <Eigen/Sparse>
#include <string>

namespace slope {

/**
 * Generates a sequence of regularization weights for the sorted L1 norm.
 *
 * @param p The number of lambda values to generate (number of features)
 * @param q The false discovery rate (FDR) level or quantile value (in (0, 1))
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
std::tuple<Eigen::ArrayXd, double, int>
regularizationPath(const Eigen::ArrayXd& alpha_in,
                   const Eigen::MatrixXd& gradient,
                   const SortedL1Norm& penalty,
                   const Eigen::ArrayXd& lambda,
                   const int n,
                   const int path_length,
                   double alpha_min_ratio);

} // namespace slope
