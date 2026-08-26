/**
 * @file
 * @brief Diagnostics for SLOPE optimization
 */

#pragma once

#include "jit_normalization.h"
#include "losses/loss.h"
#include "math.h"
#include "sorted_l1_norm.h"
#include <Eigen/Dense>
#include <algorithm>
#include <memory>

namespace slope {

/**
 * @brief Scales a candidate into the SLOPE dual constraint and evaluates it.
 *
 * @tparam MatrixType The type of the design matrix.
 * @param beta Current coefficient vector, used to determine the dual-gradient
 * dimensions.
 * @param theta Candidate dual point satisfying the loss conjugate domain.
 * @param loss Pointer to the loss function object.
 * @param sl1_norm Sorted L1 norm object.
 * @param lambda Vector of penalty parameters.
 * @param x Design matrix.
 * @param y Response matrix.
 * @param x_centers Vector of feature means for centering.
 * @param x_scales Vector of feature scales for normalization.
 * @param jit_normalization Just-in-time normalization settings.
 * @return The dual objective at the scaled feasible point.
 */
template<typename MatrixType>
double
computeDualFromPoint(const Eigen::VectorXd& beta,
                     Eigen::MatrixXd theta,
                     const std::unique_ptr<Loss>& loss,
                     const SortedL1Norm& sl1_norm,
                     const Eigen::ArrayXd& lambda,
                     const MatrixType& x,
                     const Eigen::MatrixXd& y,
                     const Eigen::VectorXd& x_centers,
                     const Eigen::VectorXd& x_scales,
                     const JitNormalization& jit_normalization)
{
  const int n = x.rows();
  Eigen::VectorXd gradient(beta.size());

  updateGradient(gradient,
                 x,
                 theta,
                 x_centers,
                 x_scales,
                 Eigen::VectorXd::Ones(n),
                 jit_normalization);

  const double dual_norm = sl1_norm.dualNorm(gradient, lambda);
  theta.array() /= std::max(1.0, dual_norm);

  return loss->dual(theta, y, Eigen::VectorXd::Ones(n));
}

/**
 * @brief Computes the dual objective function value for SLOPE optimization
 *
 * @tparam MatrixType The type of the design matrix
 * @param beta Current coefficient vector
 * @param eta Current linear predictor
 * @param loss Pointer to the loss function object
 * @param sl1_norm Sorted L1 norm object
 * @param lambda Vector of penalty parameters
 * @param x Design matrix
 * @param y Response matrix
 * @param x_centers Vector of feature means for centering
 * @param x_scales Vector of feature scales for normalization
 * @param jit_normalization Just-in-time normalization settings
 * @param intercept Boolean indicating if intercept is included in the model
 *
 * @return double The computed dual objective value
 *
 * @details This function computes the dual objective value for the SLOPE
 * optimization problem. It handles both cases with and without intercept terms,
 * applying appropriate normalization and gradient computations.
 */
template<typename MatrixType>
double
computeDual(const Eigen::VectorXd& beta,
            const Eigen::MatrixXd& eta,
            const std::unique_ptr<Loss>& loss,
            const SortedL1Norm& sl1_norm,
            const Eigen::ArrayXd& lambda,
            const MatrixType& x,
            const Eigen::MatrixXd& y,
            const Eigen::VectorXd& x_centers,
            const Eigen::VectorXd& x_scales,
            const JitNormalization& jit_normalization,
            const bool intercept)
{
  Eigen::MatrixXd theta = loss->dualPoint(eta, y, intercept);
  return computeDualFromPoint(beta,
                              theta,
                              loss,
                              sl1_norm,
                              lambda,
                              x,
                              y,
                              x_centers,
                              x_scales,
                              jit_normalization);
}

} // namespace slope
