/**
 * @file
 * @brief Hybrid numerical solver for SLOPE combining coordinate descent and
 * proximal gradient descent
 */

#pragma once

#include "hybrid_cd.h"
#include "hybrid_pgd.h"
#include "slope/clusters.h"
#include "slope/constants.h"
#include "slope/helpers.h"
#include "slope/objectives/objective.h"
#include "slope/sorted_l1_norm.h"
#include "solver.h"
#include <memory>

namespace slope {
namespace solvers {

/**
 * @brief Hybrid CD-PGD solver for SLOPE
 *
 * This solver alternates between coordinate descent (CD) and proximal gradient
 * descent (PGD) steps to solve the SLOPE optimization problem. The hybrid
 * approach aims to combine the benefits of both methods:
 * - CD: Efficient updates for sparse solutions
 * - PGD: Better handling of correlated features and faster convergence in some
 * cases
 *
 * The switching between methods is controlled by the pgd_freq parameter, which
 * determines how often PGD steps are taken versus CD steps.
 */
class Hybrid : public SolverBase
{
public:
  /**
   * @brief Construct a new Hybrid Solver
   *
   * @param args Arguments forwarded to base solver constructor
   */
  Hybrid(double tol,
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
  {
  }

  // Override for dense matrices
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

  // Override for sparse matrices
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
  /**
   * @brief Implementation of the hybrid solver algorithm
   *
   * @tparam MatrixType Type of the design matrix
   * @param beta0 Intercept term (scalar)
   * @param beta Coefficient matrix
   * @param eta Linear predictor
   * @param clusters Coefficient clustering information
   * @param objective Pointer to the objective function
   * @param penalty SLOPE penalty object
   * @param x Design matrix
   * @param x_centers Feature centers for standardization
   * @param x_scales Feature scales for standardization
   * @param y Response variable
   */
  template<typename MatrixType>
  void runImpl(Eigen::VectorXd& beta0,
               Eigen::MatrixXd& beta,
               Eigen::MatrixXd& eta,
               Clusters& clusters,
               const Eigen::ArrayXd& lambda,
               const std::unique_ptr<Objective>& objective,
               const SortedL1Norm& penalty,
               Eigen::MatrixXd& gradient_in,
               const std::vector<int>& working_set,
               const MatrixType& x,
               const Eigen::VectorXd& x_centers,
               const Eigen::VectorXd& x_scales,
               const Eigen::VectorXd& y)
  {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    const int n = x.rows();

    VectorXd w = VectorXd::Ones(n);
    VectorXd z = y;
    objective->updateWeightsAndWorkingResponse(w, z, eta, y);
    VectorXd w_sqrt = w.cwiseSqrt();

    VectorXd residual = z - eta;
    MatrixXd gradient = gradient_in;

    for (int it = 0; it < this->max_it_inner; ++it) {
      if (it % this->pgd_freq == 0) {
        double g = residual.cwiseAbs2().dot(w) / (2.0 * n);
        double h = penalty.eval(beta(working_set, Eigen::all).reshaped(),
                                lambda.head(working_set.size()));
        double primal_inner = g + h;

        updateGradient(gradient,
                       x,
                       residual,
                       working_set,
                       x_centers,
                       x_scales,
                       w,
                       this->standardize_jit);

        VectorXd theta = residual;

        // First compute gradient with potential offset for intercept case
        MatrixXd dual_gradient = gradient;
        if (this->intercept) {
          Eigen::VectorXd theta_mean(1);
          theta_mean(0) = theta.mean();
          theta.array() -= theta_mean(0);

          offsetGradient(dual_gradient,
                         x,
                         theta_mean,
                         working_set,
                         x_centers,
                         x_scales,
                         standardize_jit);
        }

        // Obtain a feasible dual point by dual scaling
        double dual_norm =
          penalty.dualNorm(dual_gradient(working_set, Eigen::all).reshaped(),
                           lambda.head(working_set.size()));
        theta.array() /= std::max(1.0, dual_norm);

        double dual_inner = (theta.cwiseProduct(w).dot(z) -
                             0.5 * theta.cwiseAbs2().cwiseProduct(w).sum()) /
                            n;

        double dual_gap_inner = primal_inner - dual_inner;

        assert(dual_gap_inner > -1e-5 && "Inner dual gap should be positive");

        double tol_inner =
          (std::abs(primal_inner) + constants::EPSILON) * this->tol;

        if (this->print_level > 2) {
          std::cout << indent(2) << "iteration: " << it << std::endl
                    << indent(3) << "primal (inner): " << primal_inner
                    << std::endl
                    << indent(3) << "duality gap (inner): " << dual_gap_inner
                    << ", tol: " << tol_inner << std::endl;
        }

        if (std::max(dual_gap_inner, 0.0) <= tol_inner) {
          break;
        }

        if (this->print_level > 2) {
          std::cout << indent(3) << "Running PGD step" << std::endl;
        }

        proximalGradientDescent(beta0,
                                beta,
                                residual,
                                this->pgd_learning_rate,
                                lambda,
                                gradient,
                                working_set,
                                x,
                                w,
                                z,
                                penalty,
                                x_centers,
                                x_scales,
                                g,
                                this->intercept,
                                this->standardize_jit,
                                this->pgd_learning_rate_decr,
                                this->print_level);

        // TODO: We might be able to speed up cluster updating since we know
        // betas outside the active set cannot have changed
        clusters.update(beta);
      } else {
        if (this->print_level > 2) {
          std::cout << indent(3) << "Running CD step" << std::endl;
        }

        coordinateDescent(beta0,
                          beta,
                          residual,
                          clusters,
                          lambda,
                          x,
                          w,
                          x_centers,
                          x_scales,
                          this->intercept,
                          this->standardize_jit,
                          this->update_clusters);
      }
    }

    // The residual is kept up to date, but not eta. So we need to compute
    // it here.
    eta = z - residual;
    // TODO: register convergence status
  }

  double pgd_learning_rate =
    1.0; ///< Learning rate for proximal gradient descent steps
  double pgd_learning_rate_decr =
    0.5; ///< Learning rate decrease factor on failed PGD steps
  int max_it_inner = 10000; ///< Maximum number of inner iterations
};

} // namespace solvers
} // namespace slope
