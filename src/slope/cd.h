#pragma once

#include "clusters.h"
#include "slope_threshold.h"
#include "sorted_l1_norm.h"
#include <RcppEigen.h>
#include <vector>

namespace slope {

template<typename T>
void
coordinateDescent(double& beta0,
                  Eigen::VectorXd& beta,
                  Eigen::VectorXd& residual,
                  Clusters& clusters,
                  const T& x,
                  const Eigen::VectorXd& w,
                  const Eigen::VectorXd& z,
                  const SortedL1Norm& sl1_norm,
                  const Eigen::VectorXd& x_centers,
                  const Eigen::VectorXd& x_scales,
                  bool intercept,
                  bool standardize,
                  bool update_clusters,
                  int print_level)
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
      double s_k = sign(beta(k));
      s.emplace_back(s_k);

      if (standardize) {
        gradient_j = -s_k *
                     (x.col(k).cwiseProduct(w).dot(residual) -
                      w.dot(residual) * x_centers(k)) /
                     (n * x_scales(k));
        // TODO: Consider caching the hessian values. We need to invalidate the
        // cache every time the cluster is updated or the  signs flip relatively
        // inside the cluster.
        hessian_j =
          (x.col(k).cwiseAbs2().dot(w) - 2 * x_centers(k) * x.col(k).dot(w) +
           std::pow(x_centers(k), 2) * w.sum()) /
          (std::pow(x_scales(k), 2) * n);
      } else {
        gradient_j = -s_k * x.col(k).cwiseProduct(w).dot(residual) / n;
        hessian_j = x.col(k).cwiseAbs2().dot(w) / n;
      }
    } else {
      // There's no reasonable just-in-time standardization approach for sparse
      // design matrices when there are clusters in the data, so we need to
      // reduce to a dense column vector.
      x_s.setZero();

      for (auto c_it = clusters.cbegin(j); c_it != clusters.cend(j); ++c_it) {
        int k = *c_it;
        double s_k = sign(beta(k));
        s.emplace_back(s_k);

        if (standardize) {
          x_s += x.col(k) * (s_k / x_scales(k));
          x_s.array() -= x_centers(k) * s_k / x_scales(k);
        } else {
          x_s += x.col(k) * s_k;
        }
      }

      hessian_j = x_s.cwiseAbs2().dot(w) / n;
      gradient_j = -x_s.cwiseProduct(w).dot(residual) / n;
    }

    auto thresholding_results =
      slopeThreshold(c_old - gradient_j / hessian_j,
                     j,
                     sl1_norm.lambda * sl1_norm.getAlpha() / (hessian_j),
                     clusters);

    double c_tilde = thresholding_results.value;
    int new_index = thresholding_results.new_index;

    auto s_it = s.cbegin();
    auto c_it = clusters.cbegin(j);
    for (; c_it != clusters.cend(j); ++c_it, ++s_it) {
      beta(*c_it) = c_tilde * (*s_it);
    }

    double c_diff = c_old - c_tilde;

    if (c_diff != 0) {
      if (cluster_size == 1) {
        int k = *clusters.cbegin(j);

        if (standardize) {
          residual += x.col(k) * (s[0] * c_diff / x_scales(k));
          residual.array() -= x_centers(k) * s[0] * c_diff / x_scales(k);
        } else {
          residual += x.col(k) * (s[0] * c_diff);
        }
      } else {
        residual += x_s * c_diff;
      }
    }

    if (update_clusters) {
      clusters.update(j, new_index, std::abs(c_tilde));
    } else {
      clusters.setCoeff(j, std::abs(c_tilde));
    }

    if (intercept) {
      double beta0_update = residual.dot(w) / w.sum();
      residual.array() -= beta0_update;
      beta0 += beta0_update;
    }

    Rcpp::checkUserInterrupt();
  }
}
}
