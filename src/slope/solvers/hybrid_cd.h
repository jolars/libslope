/**
 * @file
 * @brief An implementation of the coordinate descent step in the hybrid
 * algorithm for solving SLOPE.
 */

#pragma once

#include "../clusters.h"
#include "../math.h"
#include "slope_threshold.h"
#include <Eigen/Core>
#include <vector>

namespace slope {
namespace solvers {

template<typename T>
std::pair<double, double>
computeGradientAndHessian(const T& x,
                          int k,
                          const Eigen::VectorXd& w,
                          const Eigen::VectorXd& residual,
                          const Eigen::VectorXd& x_centers,
                          const Eigen::VectorXd& x_scales,
                          double s,
                          bool normalize_jit,
                          int n)
{
  double gradient, hessian;

  if (normalize_jit) {
    gradient = s *
               (x.col(k).cwiseProduct(w).dot(residual) -
                w.dot(residual) * x_centers(k)) /
               (n * x_scales(k));

    hessian =
      (x.col(k).cwiseAbs2().dot(w) - 2 * x_centers(k) * x.col(k).dot(w) +
       std::pow(x_centers(k), 2) * w.sum()) /
      (std::pow(x_scales(k), 2) * n);
  } else {
    gradient = s * (x.col(k).cwiseProduct(w).dot(residual)) / n;
    hessian = x.col(k).cwiseAbs2().dot(w) / n;
  }

  return { gradient, hessian };
}

/**
 * Coordinate Descent Step
 *
 * This function takes a coordinate descent step in the hybrid CD/PGD algorithm
 * for SLOPE.
 *
 * @tparam T The type of the design matrix. This can be either a dense or
 * sparse.
 * @param beta0 The intercept
 * @param beta The coefficients
 * @param residual The residual vector
 * @param clusters The cluster information, stored in a Cluster object.
 * @param x The design matrix
 * @param w The weight vector
 * @param sl1_norm The sorted L1 norm object
 * @param x_centers The center values of the data matrix columns
 * @param x_scales The scale values of the data matrix columns
 * @param intercept Shuold an intervept be fit?
 * @param normalize_jit Flag indicating whether we are standardizing `x`
 *   just-in-time.
 * @param update_clusters Flag indicating whether to update the cluster
 * information
 *
 * @see Clusters
 * @see SortedL1Norm
 */
template<typename T>
void
coordinateDescent(Eigen::VectorXd& beta0,
                  Eigen::MatrixXd& beta,
                  Eigen::VectorXd& residual,
                  Clusters& clusters,
                  const Eigen::ArrayXd& lambda,
                  const T& x,
                  const Eigen::VectorXd& w,
                  const Eigen::VectorXd& x_centers,
                  const Eigen::VectorXd& x_scales,
                  const bool intercept,
                  const bool normalize_jit,
                  const bool update_clusters)
{
  using namespace Eigen;

  const int n = x.rows();

  for (int j = 0; j < clusters.n_clusters(); ++j) {
    double c_old = clusters.coeff(j);

    if (c_old == 0) {
      // We do not update the zero cluster because it can be very large, but
      // often does not change.
      continue;
    }

    std::vector<int> s;
    int cluster_size = clusters.cluster_size(j);
    s.reserve(cluster_size);

    double hessian_j = 1;
    double gradient_j = 0;
    VectorXd x_s(n);

    if (cluster_size == 1) {
      int k = *clusters.cbegin(j);
      double s_k = sign(beta(k, 0));
      s.emplace_back(s_k);

      std::tie(gradient_j, hessian_j) = computeGradientAndHessian(
        x, k, w, residual, x_centers, x_scales, s_k, normalize_jit, n);
    } else {
      // There's no reasonable just-in-time standardization approach for sparse
      // design matrices when there are clusters in the data, so we need to
      // reduce to a dense column vector.
      x_s.setZero();

      for (auto c_it = clusters.cbegin(j); c_it != clusters.cend(j); ++c_it) {
        int k = *c_it;
        double s_k = sign(beta(k, 0));
        s.emplace_back(s_k);

        if (normalize_jit) {
          x_s += x.col(k) * (s_k / x_scales(k));
          x_s.array() -= x_centers(k) * s_k / x_scales(k);
        } else {
          x_s += x.col(k) * s_k;
        }
      }

      hessian_j = x_s.cwiseAbs2().dot(w) / n;
      gradient_j = x_s.cwiseProduct(w).dot(residual) / n;
    }

    auto [c_tilde, new_index] = slopeThreshold(
      c_old - gradient_j / hessian_j, j, lambda / hessian_j, clusters);

    auto s_it = s.cbegin();
    auto c_it = clusters.cbegin(j);
    for (; c_it != clusters.cend(j); ++c_it, ++s_it) {
      beta(*c_it, 0) = c_tilde * (*s_it);
    }

    double c_diff = c_old - c_tilde;

    if (c_diff != 0) {
      if (cluster_size == 1) {
        int k = *clusters.cbegin(j);

        if (normalize_jit) {
          residual -= x.col(k) * (s[0] * c_diff / x_scales(k));
          residual.array() += x_centers(k) * s[0] * c_diff / x_scales(k);
        } else {
          residual -= x.col(k) * (s[0] * c_diff);
        }
      } else {
        residual -= x_s * c_diff;
      }
    }

    if (update_clusters) {
      clusters.update(j, new_index, std::abs(c_tilde));
    } else {
      clusters.setCoeff(j, std::abs(c_tilde));
    }

    if (intercept) {
      // TODO: Consider whether this should only be done once per loop instead
      // of after each coordinate update
      double beta0_update = residual.dot(w) / n;
      residual.array() -= beta0_update;
      beta0(0) -= beta0_update;
    }
  }
}

} // namespace solvers
} // namespace slope
