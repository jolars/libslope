/**
 * @file
 * @brief Proximal Gradient Descent solver implementation for SLOPE
 */

#pragma once

#include "../sorted_l1_norm.h"
#include "math.h"
#include "slope/clusters.h"
#include "slope/math.h"
#include "slope/objectives/objective.h"
#include "solver.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>

namespace slope {
namespace solvers {

/**
 * @brief Proximal Gradient Descent solver for SLOPE optimization
 *
 * This solver implements the proximal gradient descent algorithm with line
 * search for solving the SLOPE optimization problem. It uses backtracking line
 * search to automatically adjust the learning rate for optimal convergence.
 */
class PGD : public SolverBase
{
public:
  PGD(double tol,
      int max_it,
      bool standardize_jit,
      int print_level,
      bool intercept,
      bool update_clusters,
      int pgd_freq)
    : SolverBase(tol,
                 max_it,
                 standardize_jit,
                 print_level,
                 intercept,
                 update_clusters,
                 pgd_freq)
    , learning_rate(1.0)
    , learning_rate_decr(0.5)
  {
  }

  /**
   * @brief Implementation of the PGD solver algorithm
   *
   * @tparam MatrixType Type of the design matrix
   * @param beta0 Intercept term (scalar)
   * @param beta Coefficient matrix
   * @param residual Residual vector
   * @param gradient Gradient vector
   * @param x Design matrix
   * @param w Weight vector
   * @param z Response vector
   * @param sl1_norm Sorted L1 norm object
   * @param x_centers Feature centers for standardization
   * @param x_scales Feature scales for standardization
   * @param g_old Previous value of objective function
   */
  void run(Eigen::VectorXd& beta0,
           Eigen::MatrixXd& beta,
           Eigen::MatrixXd& eta,
           Clusters& clusters,
           const Eigen::ArrayXd& lambda,
           const std::unique_ptr<Objective>& objective,
           SortedL1Norm& penalty,
           Eigen::MatrixXd& gradient,
           const std::vector<int>& active_set,
           const Eigen::MatrixXd& x,
           const Eigen::VectorXd& x_centers,
           const Eigen::VectorXd& x_scales,
           const Eigen::MatrixXd& y) override;

  /**
   * @brief Implementation of the PGD solver algorithm
   *
   * @tparam MatrixType Type of the design matrix
   * @param beta0 Intercept term (scalar)
   * @param beta Coefficient matrix
   * @param residual Residual vector
   * @param gradient Gradient vector
   * @param x Design matrix
   * @param w Weight vector
   * @param z Response vector
   * @param sl1_norm Sorted L1 norm object
   * @param x_centers Feature centers for standardization
   * @param x_scales Feature scales for standardization
   * @param g_old Previous value of objective function
   */
  void run(Eigen::VectorXd& beta0,
           Eigen::MatrixXd& beta,
           Eigen::MatrixXd& eta,
           Clusters& clusters,
           const Eigen::ArrayXd& lambda,
           const std::unique_ptr<Objective>& objective,
           SortedL1Norm& penalty,
           Eigen::MatrixXd& gradient,
           const std::vector<int>& active_set,
           const Eigen::SparseMatrix<double>& x,
           const Eigen::VectorXd& x_centers,
           const Eigen::VectorXd& x_scales,
           const Eigen::MatrixXd& y) override;

private:
  template<typename MatrixType>
  void runImpl(Eigen::VectorXd& beta0,
               Eigen::MatrixXd& beta,
               Eigen::MatrixXd& eta,
               Clusters& clusters,
               const Eigen::ArrayXd& lambda,
               const std::unique_ptr<Objective>& objective,
               const SortedL1Norm& penalty,
               Eigen::MatrixXd& gradient,
               const std::vector<int>& active_set,
               const MatrixType& x,
               const Eigen::VectorXd& x_centers,
               const Eigen::VectorXd& x_scales,
               const Eigen::MatrixXd& y)
  {
    using Eigen::all;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    if (this->print_level > 2) {
      std::cout << "        Starting line search" << std::endl;
    }

    MatrixXd beta_old = beta(active_set, all);

    double g_old = objective->loss(eta, y);

    while (true) {
      beta(active_set, all) =
        penalty.prox(beta_old - this->learning_rate * gradient(active_set, all),
                     this->learning_rate * lambda.head(beta_old.size()));

      if (intercept) {
        objective->updateIntercept(beta0, eta, y);
      }

      Eigen::MatrixXd beta_diff = beta(active_set, all) - beta_old;

      eta = linearPredictor(x,
                            active_set,
                            beta0,
                            beta,
                            x_centers,
                            x_scales,
                            standardize_jit,
                            intercept);

      double g = objective->loss(eta, y);
      double q =
        g_old + beta_diff.reshaped().dot(gradient(active_set, all).reshaped()) +
        (1.0 / (2 * this->learning_rate)) * beta_diff.reshaped().squaredNorm();

      if (q >= g * (1 - 1e-12)) {
        this->learning_rate *= 1.1;
        break;
      } else {
        this->learning_rate *= this->learning_rate_decr;
      }
    }
  }

  double learning_rate;      ///< Current learning rate for gradient steps
  double learning_rate_decr; ///< Learning rate decrease factor for line search
};

} // namespace solvers
} // namespace slope
