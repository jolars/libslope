/**
 * @file
 * @brief Proximal Gradient Descent solver implementation for SLOPE
 */

#include "pgd.h"
#include "../sorted_l1_norm.h"
#include "math.h"
#include "slope/clusters.h"
#include "slope/objectives/objective.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

namespace slope {
namespace solvers {

// Override for dense matrices
void
PGD::run(Eigen::VectorXd& beta0,
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
         const Eigen::MatrixXd& y)
{
  runImpl(beta0,
          beta,
          eta,
          clusters,
          lambda,
          objective,
          penalty,
          gradient,
          active_set,
          x,
          x_centers,
          x_scales,
          y);
}

// Override for sparse matrices
void
PGD::run(Eigen::VectorXd& beta0,
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
         const Eigen::MatrixXd& y)
{
  runImpl(beta0,
          beta,
          eta,
          clusters,
          lambda,
          objective,
          penalty,
          gradient,
          active_set,
          x,
          x_centers,
          x_scales,
          y);
}

} // namespace solvers
} // namespace slope
