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
#include <Eigen/SparseCore>
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
  /**
   * @brief Constructs Proximal Gradient Descent solver for SLOPE optimization
   * @param tol Convergence tolerance threshold
   * @param max_it Maximum number of iterations
   * @param jit_normalization Feature normalization strategy
   * @param intercept If true, fits intercept term
   * @param update_clusters If true, updates clusters during optimization
   * @param pgd_freq Frequency of proximal gradient descent updates
   * @param update_type Type of update strategy to use
   */
  PGD(double tol,
      int max_it,
      JitNormalization jit_normalization,
      bool intercept,
      bool update_clusters,
      int pgd_freq,
      const std::string& update_type)
    : SolverBase(tol,
                 max_it,
                 jit_normalization,
                 intercept,
                 update_clusters,
                 pgd_freq)
    , learning_rate(1.0)
    , learning_rate_decr(0.5)
    , update_type{ update_type }
    , t(1.0)
  {
  }

  /// @copydoc SolverBase::run
  void run(Eigen::VectorXd& beta0,
           Eigen::MatrixXd& beta,
           Eigen::MatrixXd& eta,
           Clusters& clusters,
           const Eigen::ArrayXd& lambda,
           const std::unique_ptr<Objective>& objective,
           SortedL1Norm& penalty,
           Eigen::MatrixXd& gradient,
           const std::vector<int>& working_set,
           const Eigen::MatrixXd& x,
           const Eigen::VectorXd& x_centers,
           const Eigen::VectorXd& x_scales,
           const Eigen::MatrixXd& y) override;

  /// @copydoc SolverBase::run
  void run(Eigen::VectorXd& beta0,
           Eigen::MatrixXd& beta,
           Eigen::MatrixXd& eta,
           Clusters& clusters,
           const Eigen::ArrayXd& lambda,
           const std::unique_ptr<Objective>& objective,
           SortedL1Norm& penalty,
           Eigen::MatrixXd& gradient,
           const std::vector<int>& working_set,
           const Eigen::SparseMatrix<double>& x,
           const Eigen::VectorXd& x_centers,
           const Eigen::VectorXd& x_scales,
           const Eigen::MatrixXd& y) override;

private:
  template<typename MatrixType>
  void runImpl(Eigen::VectorXd& beta0,
               Eigen::MatrixXd& beta,
               Eigen::MatrixXd& eta,
               Clusters&,
               const Eigen::ArrayXd& lambda,
               const std::unique_ptr<Objective>& objective,
               const SortedL1Norm& penalty,
               Eigen::MatrixXd& gradient,
               const std::vector<int>& working_set,
               const MatrixType& x,
               const Eigen::VectorXd& x_centers,
               const Eigen::VectorXd& x_scales,
               const Eigen::MatrixXd& y)
  {
    using Eigen::all;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    beta_old = beta(working_set, all);

    double g_old = objective->loss(eta, y);
    double t_old = t;

    Eigen::MatrixXd beta_diff(beta_old.rows(), beta_old.cols());

    while (true) {
      beta(working_set, all) = penalty.prox(
        beta_old - this->learning_rate * gradient(working_set, all),
        this->learning_rate * lambda.head(beta_old.size()));

      if (intercept) {
        objective->updateIntercept(beta0, eta, y);
      }

      beta_diff = beta(working_set, all) - beta_old;

      eta = linearPredictor(x,
                            working_set,
                            beta0,
                            beta,
                            x_centers,
                            x_scales,
                            jit_normalization,
                            intercept);

      double g = objective->loss(eta, y);
      double q =
        g_old +
        beta_diff.reshaped().dot(gradient(working_set, all).reshaped()) +
        (1.0 / (2 * this->learning_rate)) * beta_diff.reshaped().squaredNorm();

      if (q >= g * (1 - 1e-12)) {
        this->learning_rate *= 1.1;
        break;
      } else {
        this->learning_rate *= this->learning_rate_decr;
      }
    }

    if (update_type == "fista") {
      this->t = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t_old * t_old));
      beta(working_set, all) += beta_diff * (t_old - 1.0) / this->t;
      eta = linearPredictor(x,
                            working_set,
                            beta0,
                            beta,
                            x_centers,
                            x_scales,
                            jit_normalization,
                            intercept);
    }
  }

  double learning_rate;      ///< Current learning rate for gradient steps
  double learning_rate_decr; ///< Learning rate decrease factor for line search
  std::string update_type;   ///< Update type for PGD
  double t;                  ///< FISTA step size
  Eigen::MatrixXd beta_old;  ///< Old beta values
};

} // namespace solvers
} // namespace slope
