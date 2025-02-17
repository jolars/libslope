/**
 * @file
 * @brief Numerical solver class for SLOPE (Sorted L-One Penalized Estimation)
 */

#pragma once

#include "slope/clusters.h"
#include "slope/objectives/objective.h"
#include "slope/sorted_l1_norm.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>

namespace slope {
namespace solvers {

/**
 * @brief Base class template for SLOPE numerical solvers
 *
 * @tparam Derived CRTP derived class implementing the specific solver algorithm
 *
 * This class implements the base functionality for Sorted L-One Penalized
 * Estimation (SLOPE) solvers using the Curiously Recurring Template Pattern
 * (CRTP). SLOPE is a method that generalizes the Lasso by penalizing the sorted
 * magnitudes of the coefficients with decreasing weights.
 */
class SolverBase
{
public:
  SolverBase(double tol,
             int max_it,
             bool standardize_jit,
             int print_level,
             bool intercept,
             bool update_clusters,
             int pgd_freq)
    : tol(tol)
    , max_it(max_it)
    , standardize_jit(standardize_jit)
    , print_level(print_level)
    , intercept(intercept)
    , update_clusters(update_clusters)
    , pgd_freq(pgd_freq)
  {
  }
  virtual ~SolverBase() = default;

  // All solver-specific arguments are given as parameters
  virtual void run(Eigen::VectorXd& beta0,
                   Eigen::MatrixXd& beta,
                   Eigen::MatrixXd& eta,
                   Clusters& clusters,
                   const std::unique_ptr<Objective>& objective,
                   SortedL1Norm& sl1_norm,
                   Eigen::MatrixXd& gradient,
                   const std::vector<int>& active_set,
                   const Eigen::MatrixXd& x,
                   const Eigen::VectorXd& x_centers,
                   const Eigen::VectorXd& x_scales,
                   const Eigen::MatrixXd& y) = 0;

  // All solver-specific arguments are given as parameters
  virtual void run(Eigen::VectorXd& beta0,
                   Eigen::MatrixXd& beta,
                   Eigen::MatrixXd& eta,
                   Clusters& clusters,
                   const std::unique_ptr<Objective>& objective,
                   SortedL1Norm& sl1_norm,
                   Eigen::MatrixXd& gradient,
                   const std::vector<int>& active_set,
                   const Eigen::SparseMatrix<double>& x,
                   const Eigen::VectorXd& x_centers,
                   const Eigen::VectorXd& x_scales,
                   const Eigen::MatrixXd& y) = 0;

protected:
  double tol;
  int max_it;
  bool standardize_jit;
  int print_level;
  bool intercept;
  bool update_clusters;
  int pgd_freq;
};

} // namespace solvers
} // namespace slope
