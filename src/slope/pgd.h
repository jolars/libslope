#pragma once

#include "helpers.h"
#include "math.h"
#include "parameters.h"
#include "sorted_l1_norm.h"
#include <Eigen/Dense>
#include <iostream>

namespace slope {
template<typename T>
void
proximalGradientDescent(double& beta0,
                        Eigen::VectorXd& beta,
                        Eigen::VectorXd& residual,
                        double& learning_rate,
                        const Eigen::VectorXd& gradient,
                        const T& x,
                        const Eigen::VectorXd& w,
                        const Eigen::VectorXd& z,
                        const SortedL1Norm& sl1_norm,
                        const Eigen::VectorXd& x_centers,
                        const Eigen::VectorXd& x_scales,
                        const double g_old,
                        const SlopeParameters& params)
{
  const int n = x.rows();
  const int p = x.cols();

  // Proximal gradient descent with line search
  if (params.print_level > 2) {
    std::cout << "        Starting line search" << std::endl;
  }

  Eigen::VectorXd beta_old = beta;

  while (true) {
    beta = sl1_norm.prox(beta_old - learning_rate * gradient, learning_rate);

    Eigen::VectorXd beta_diff = beta - beta_old;

    if (params.standardize) {
      residual = z - x * beta.cwiseQuotient(x_scales);
      residual.array() += x_centers.cwiseQuotient(x_scales).dot(beta);
    } else {
      residual = z - x * beta;
    }

    if (params.intercept) {
      beta0 = residual.dot(w) / w.sum();
      residual.array() -= beta0;
    }

    double g = (0.5 / n) * residual.cwiseAbs2().dot(w);
    double q = g_old + beta_diff.dot(gradient) +
               (1.0 / (2 * learning_rate)) * beta_diff.squaredNorm();

    if (q >= g * (1 - 1e-12)) {
      break;
    } else {
      learning_rate *= params.learning_rate_decr;
    }
  }
}

} // namespace slope
