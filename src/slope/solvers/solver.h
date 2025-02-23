/**
 * @file
 * @brief Numerical solver class for SLOPE (Sorted L-One Penalized Estimation)
 */

#pragma once

#include "slope/clusters.h"
#include "slope/normalize.h"
#include "slope/objectives/objective.h"
#include "slope/sorted_l1_norm.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>

namespace slope {
namespace solvers {

/**
 * @class SolverBase
 * @brief Abstract base class for SLOPE optimization solvers
 *
 * Provides the interface and common functionality for different SLOPE (Sorted
 * L-One Penalized Estimation) optimization algorithms. Derived classes
 * implement specific optimization strategies like coordinate descent or
 * proximal gradient descent.
 *
 * @see slope::solvers::PGD
 * @see slope::solvers::CD
 */
class SolverBase
{
public:
  /**
   * @brief Constructs a base solver for SLOPE optimization
   *
   * @param tol Convergence tolerance for the optimization
   * @param max_it Maximum number of iterations allowed
   * @param jit_normalization Type of just-in-time normalization to apply (None,
   * Center, Scale, or Both)
   * @param intercept Whether to fit an intercept term
   * @param update_clusters Whether to update coefficient clusters during
   * optimization
   * @param pgd_freq Frequency of proximal gradient descent updates (0 for
   * coordinate descent only)
   */
  SolverBase(double tol,
             int max_it,
             JitNormalization jit_normalization,
             bool intercept,
             bool update_clusters,
             int pgd_freq)
    : tol(tol)
    , max_it(max_it)
    , jit_normalization(jit_normalization)
    , intercept(intercept)
    , update_clusters(update_clusters)
    , pgd_freq(pgd_freq)
  {
  }

  /// Default desstructor
  virtual ~SolverBase() = default;

  /**
   * @brief Pure virtual function defining the solver's optimization routine
   *
   * @param beta0 Intercept terms for each response
   * @param beta Coefficient matrix (p predictors x m responses)
   * @param eta Linear predictor matrix (n samples x m responses)
   * @param clusters Coefficient clustering structure
   * @param lambda Vector of regularization parameters
   * @param objective Pointer to objective function object
   * @param penalty Sorted L1 norm object for proximal operations
   * @param gradient Gradient matrix for objective function
   * @param working_set Vector of indices for active predictors
   * @param x Input feature matrix (n samples x p predictors)
   * @param x_centers Vector of feature means for centering
   * @param x_scales Vector of feature scales for normalization
   * @param y Response matrix (n samples x m responses)
   */
  virtual void run(Eigen::VectorXd& beta0,
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
                   const Eigen::MatrixXd& y) = 0;

  /**
   * @brief Pure virtual function defining the solver's optimization routine
   *
   * @param beta0 Intercept terms for each response
   * @param beta Coefficient matrix (p predictors x m responses)
   * @param eta Linear predictor matrix (n samples x m responses)
   * @param clusters Coefficient clustering structure
   * @param lambda Vector of regularization parameters
   * @param objective Pointer to objective function object
   * @param penalty Sorted L1 norm object for proximal operations
   * @param gradient Gradient matrix for objective function
   * @param working_set Vector of indices for active predictors
   * @param x Input feature matrix (n samples x p predictors)
   * @param x_centers Vector of feature means for centering
   * @param x_scales Vector of feature scales for normalization
   * @param y Response matrix (n samples x m responses)
   */
  virtual void run(Eigen::VectorXd& beta0,
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
                   const Eigen::MatrixXd& y) = 0;

protected:
  double tol;                         ///< Convergence tolerance threshold
  int max_it;                         ///< Maximum iterations
  JitNormalization jit_normalization; ///< JIT feature normalization strategy
  bool intercept;                     ///< If true, fits intercept term
  bool update_clusters; ///< If true, updates clusters during optimization
  int pgd_freq;         ///< Proximal gradient descent update frequency
};

} // namespace solvers
} // namespace slope
