#pragma once

#include "cd.h"
#include "clusters.h"
#include "helpers.h"
#include "objectives/objective.h"
#include "pgd.h"
#include "regularization_sequence.h"
#include "results.h"
#include "setup_objective.h"
#include "slope_threshold.h"
#include "sorted_l1_norm.h"
#include "standardize.h"
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <vector>

namespace slope {

// This is the convergence criterion from glmnet.
template<typename T>
double
computeMaxDelta(const T& x,
                const Eigen::VectorXd& x_centers,
                const Eigen::VectorXd& x_scales,
                const Eigen::VectorXd& w,
                const Eigen::VectorXd& beta_old,
                const Eigen::VectorXd& beta,
                bool standardize)
{
  const int p = x.cols();

  double max_delta = 0.0;

  // TODO: There's no need to traverse all the coefficients if we find one that
  // violates the stopping rule.
  for (int j = 0; j < p; ++j) {
    double v_j;

    if (standardize) {
      v_j = (x.col(j).cwiseAbs2().dot(w) - 2 * x_centers(j) * x.col(j).dot(w) +
             std::pow(x_centers(j), 2) * w.sum()) /
            std::pow(x_scales(j), 2);
    } else {
      v_j = x.col(j).cwiseAbs2().dot(w);
    }
    double delta_j = v_j * std::pow(beta_old(j) - beta(j), 2);
    max_delta = std::max(max_delta, delta_j);
  }

  return max_delta;
}

template<typename T>
Results
slope(const T& x,
      const Eigen::MatrixXd& y,
      std::string objective_choice,
      Eigen::ArrayXd alpha,
      const Eigen::ArrayXd& lambda,
      bool intercept,
      bool standardize,
      int path_length,
      double alpha_min_ratio,
      int pgd_freq,
      double tol,
      int max_it,
      int max_it_outer,
      bool update_clusters,
      int print_level)
{
  using Eigen::VectorXd;

  const int n = x.rows();
  const int p = x.cols();

  SortedL1Norm sl1_norm{ lambda };

  // standardize
  VectorXd x_centers(p);
  VectorXd x_scales(p);

  if (standardize) {
    std::tie(x_centers, x_scales) = computeMeanAndStdDev(x);
  }

  std::unique_ptr<Objective> objective = setupObjective(objective_choice);

  VectorXd eta = VectorXd::Zero(n);
  VectorXd w(n);
  VectorXd z(n);

  objective->updateWeightsAndWorkingResponse(w, z, eta, y);

  // The intercept can be fit directly for the null model.
  double beta0 = intercept ? z.dot(w) / w.sum() : 0.0;
  eta.array() += beta0;
  VectorXd residual = z - eta;

  if (alpha.size() == 0) {
    // No user-supplied alpha sequence, so generate one.
    alpha = regularizationPath(x,
                               w,
                               z,
                               x_centers,
                               x_scales,
                               sl1_norm,
                               path_length,
                               alpha_min_ratio,
                               intercept,
                               standardize);
  } else {
    path_length = alpha.size();
  }

  if (print_level > 0) {
    printContents(alpha, "alpha");
  }

  VectorXd beta = VectorXd::Zero(p);

  VectorXd beta0s(path_length);
  std::vector<Eigen::Triplet<double>> beta_triplets;

  std::vector<double> primals;

  double learning_rate = 1.0;
  double learning_rate_decr = 0.5;

  VectorXd beta_old_outer = beta;

  Clusters clusters(beta);

  // Regularization path loop
  for (int path_step = 0; path_step < path_length; ++path_step) {
    if (print_level > 0) {
      std::cout << "Path step: " << path_step << ", alpha: " << alpha(path_step)
                << std::endl;
    }

    sl1_norm.setAlpha(alpha(path_step));

    // IRLS loop
    for (int it_outer = 0; it_outer < max_it_outer; ++it_outer) {
      eta = z - residual;

      double primal = objective->loss(eta, y) + sl1_norm.eval(beta);
      primals.emplace_back(primal);

      beta_old_outer = beta;

      objective->updateWeightsAndWorkingResponse(w, z, eta, y);
      residual = z - eta;

      if (print_level > 1) {
        std::cout << "  IRLS iteration: " << it_outer << std::endl;
        std::cout << "    primal (main problem): " << primal << std::endl;
      }

      if (print_level > 3) {
        printContents(w, "    weights");
        printContents(z, "    working response");
      }

      double max_update_inner = 0;

      for (int it = 0; it < max_it; ++it) {
        max_update_inner = 0;

        double g = (0.5 / n) * residual.cwiseAbs2().dot(w);
        double h = sl1_norm.eval(beta);

        if (print_level > 2) {
          std::cout << "    iteration: " << it << std::endl;
          std::cout << "      primal (sub problem): " << g + h << std::endl;
        }

        if (it % pgd_freq == 0) {
          VectorXd beta_old = beta;
          if (print_level > 2) {
            std::cout << "      Running PGD step" << std::endl;
          }

          proximalGradientDescent(beta0,
                                  beta,
                                  residual,
                                  learning_rate,
                                  x,
                                  w,
                                  z,
                                  sl1_norm,
                                  x_centers,
                                  x_scales,
                                  g,
                                  intercept,
                                  standardize,
                                  learning_rate_decr,
                                  print_level);

          clusters.update(beta);

          max_update_inner = computeMaxDelta(
            x, x_centers, x_scales, w, beta_old, beta, standardize);

          // TODO: Consider changing this criterion to use the duality gap
          // instead.
          if (print_level > 2) {
            std::cout << "      max inner update change: " << max_update_inner
                      << ", tol: " << tol << std::endl;
          }

          if (max_update_inner <= tol) {
            break;
          }
        } else {
          if (print_level > 2) {
            std::cout << "      Running CD step" << std::endl;
          }

          coordinateDescent(beta0,
                            beta,
                            residual,
                            clusters,
                            x,
                            w,
                            z,
                            sl1_norm,
                            x_centers,
                            x_scales,
                            intercept,
                            standardize,
                            update_clusters,
                            print_level);
        }
      }

      double max_update_outer = computeMaxDelta(
        x, x_centers, x_scales, w, beta_old_outer, beta, standardize);

      if (print_level > 1) {
        std::cout << "    max outer update change: " << max_update_outer
                  << ", tol: " << tol << std::endl;
      }

      if (max_update_outer <= tol) {
        break;
      }
    }

    double beta0_out;
    VectorXd beta_out;

    if (standardize) {
      std::tie(beta0_out, beta_out) =
        unstandardizeCoefficients(beta0, beta, x_centers, x_scales, intercept);
    } else {
      beta0_out = beta0;
      beta_out = beta;
    }

    // Store intercept and coefficients
    beta0s(path_step) = beta0_out;

    for (int j = 0; j < p; ++j) {
      if (beta_out(j) != 0) {
        beta_triplets.emplace_back(j, path_step, beta_out(j));
      }
    }
  }

  Eigen::SparseMatrix<double> betas(p, path_length);
  betas.setFromTriplets(beta_triplets.begin(), beta_triplets.end());

  return { beta0s, betas, primals };
}

} // namespace slope
