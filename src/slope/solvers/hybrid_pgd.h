/**
 * @file
 * @brief An implementation of a proximal gradient descent step
 */

#pragma once

#include "../sorted_l1_norm.h"
#include "math.h"
#include <Eigen/Dense>

namespace slope {
namespace solvers {

/**
 * @brief Performs proximal gradient descent with line search.
 *
 * This function updates the beta values using the proximal gradient descent
 * algorithm with line search. It also updates the residual, learning rate, and
 * other variables as necessary. It assumes that the gradient has already been
 * computed.
 *
 * @tparam T The type of the input data.
 * @param beta0 The intercept value.
 * @param beta The coefficient vector.
 * @param residual The residual vector.
 * @param learning_rate The learning rate.
 * @param gradient The gradient vector.
 * @param x The input data matrix.
 * @param w The weight vector.
 * @param z The response vector.
 * @param sl1_norm The sorted L1 norm object.
 * @param x_centers The center values of the input data.
 * @param x_scales The scale values of the input data.
 * @param g_old The previous value of the objective function.
 * @param intercept Flag indicating whether to include an intercept term.
 * @param normalize_jit Flag indicating wheter we are normalizing just-in-time.
 * @param learning_rate_decr The learning rate decrement factor.
 *
 * @see SortedL1Norm
 */
template<typename T>
void
proximalGradientDescent(Eigen::VectorXd& beta0,
                        Eigen::MatrixXd& beta,
                        Eigen::VectorXd& residual,
                        double& learning_rate,
                        const Eigen::ArrayXd& lambda,
                        const Eigen::VectorXd& gradient,
                        const std::vector<int>& working_set,
                        const T& x,
                        const Eigen::VectorXd& w,
                        const Eigen::VectorXd& z,
                        const SortedL1Norm& sl1_norm,
                        const Eigen::VectorXd& x_centers,
                        const Eigen::VectorXd& x_scales,
                        const double g_old,
                        const bool intercept,
                        const bool normalize_jit,
                        const double learning_rate_decr)
{
  const int n = x.rows();

  // Proximal gradient descent with line search

  Eigen::VectorXd beta_old = beta(working_set, 0);
  Eigen::VectorXd gradient_active = gradient(working_set, 0);

  while (true) {
    beta(working_set, 0) =
      sl1_norm.prox(beta_old - learning_rate * gradient_active,
                    lambda.head(beta_old.size()) * learning_rate);

    Eigen::VectorXd beta_diff = beta(working_set, 0) - beta_old;

    if (intercept) {
      double intercept_update = residual.dot(w) / n;
      beta0(0) -= intercept_update;
    }

    residual = linearPredictor(x,
                               working_set,
                               beta0,
                               beta,
                               x_centers,
                               x_scales,
                               normalize_jit,
                               intercept) -
               z;

    double g = (0.5 / n) * residual.cwiseAbs2().dot(w);
    double q = g_old + beta_diff.dot(gradient_active) +
               (1.0 / (2 * learning_rate)) * beta_diff.squaredNorm();

    if (q >= g * (1 - 1e-12)) {
      learning_rate *= 1.1;
      break;
    } else {
      learning_rate *= learning_rate_decr;
    }
  }
}

} // namespace solvers
} // namespace slope
