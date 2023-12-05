#pragma once

#include "cd.h"
#include "clusters.h"
#include "helpers.h"
#include "objectives.h"
#include "parameters.h"
#include "pgd.h"
#include "regularization_sequence.h"
#include "results.h"
#include "slope_threshold.h"
#include "sorted_l1_norm.h"
#include "standardize.h"
#include <Eigen/Sparse>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace slope {

/**
 * Computes the maximum delta value for the given inputs.
 *
 * @tparam T The type of the input data.
 * @param x The input data.
 * @param x_centers The centers of the input data.
 * @param x_scales The scales of the input data.
 * @param w The weights.
 * @param beta_old The old beta values.
 * @param beta The new beta values.
 * @param standardize Flag indicating whether to standardize the data.
 * @return The maximum delta value.
 */
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

/**
 * Calculates the slope coefficients for a linear regression model using the
 * SortedL1Norm regularization.
 *
 * @tparam T The type of the input matrix x.
 * @param x The input matrix of size n x p, where n is the number of
 *   observations and p is the number of predictors.
 * @param y The response matrix of size n x 1.
 * @param alpha The regularization parameter sequence. If not provided, it will
 *   be generated automatically.
 * @param lambda The regularization parameter for the SortedL1Norm
 *   regularization. If not provided, it will be set to zero.
 * @return The slope coefficients, intercept values, and primal values for each
 *   step in the regularization path.
 */
template<typename T>
Results
fitSlope(const T& x,
         const Eigen::MatrixXd& y,
         Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(0),
         Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(0),
         const SlopeParameters& params = SlopeParameters())
{
  using Eigen::VectorXd;

  const int n = x.rows();
  const int p = x.cols();

  // standardize
  VectorXd x_centers(p);
  VectorXd x_scales(p);

  if (params.standardize) {
    std::tie(x_centers, x_scales) = computeMeanAndStdDev(x);
  }

  std::unique_ptr<Objective> objective = setupObjective(params.objective);

  // initialize coeficients
  double beta0 = 0.0;
  VectorXd beta = VectorXd::Zero(p);

  VectorXd eta = x * beta;
  eta.array() += beta0;

  VectorXd w(n); // weights
  VectorXd z(n); // working response

  objective->updateWeightsAndWorkingResponse(w, z, eta, y);

  VectorXd residual = z - eta;

  // Setup the regularization sequence and path
  SortedL1Norm sl1_norm{ lambda };

  if (lambda.size() == 0) {
    lambda = lambdaSequence(p, params.q, params.lambda_type);
  } else {
    if (lambda.size() != p) {
      throw std::invalid_argument(
        "lambda must be the same length as the number of predictors");
    }
    if (lambda.minCoeff() < 0) {
      throw std::invalid_argument("lambda must be non-negative");
    }
  }

  int path_length = params.path_length;

  if (alpha.size() == 0) {
    alpha = regularizationPath(x,
                               w,
                               z,
                               x_centers,
                               x_scales,
                               sl1_norm,
                               params.path_length,
                               params.alpha_min_ratio,
                               params.intercept,
                               params.standardize);
  } else {
    path_length = alpha.size();
  }

  if (params.print_level > 0) {
    printContents(alpha, "alpha");
  }

  VectorXd beta0s(path_length);
  std::vector<Eigen::Triplet<double>> beta_triplets;

  std::vector<double> primals;

  double learning_rate = 1.0;

  VectorXd beta_old_outer = beta;

  Clusters clusters(beta);

  // Regularization path loop
  for (int path_step = 0; path_step < path_length; ++path_step) {
    if (params.print_level > 0) {
      std::cout << "Path step: " << path_step << ", alpha: " << alpha(path_step)
                << std::endl;
    }

    sl1_norm.setAlpha(alpha(path_step));

    // IRLS loop
    for (int it_outer = 0; it_outer < params.max_it_outer; ++it_outer) {
      eta = z - residual;

      double primal = objective->loss(eta, y) + sl1_norm.eval(beta);
      primals.emplace_back(primal);

      beta_old_outer = beta;

      objective->updateWeightsAndWorkingResponse(w, z, eta, y);
      residual = z - eta;

      if (params.print_level > 1) {
        std::cout << "  IRLS iteration: " << it_outer << std::endl;
        std::cout << "    primal (main problem): " << primal << std::endl;
      }

      if (params.print_level > 3) {
        printContents(w, "    weights");
        printContents(z, "    working response");
      }

      double max_update_inner = 0;

      for (int it = 0; it < params.max_it; ++it) {
        max_update_inner = 0;

        double g = (0.5 / n) * residual.cwiseAbs2().dot(w);
        double h = sl1_norm.eval(beta);

        if (params.print_level > 2) {
          std::cout << "    iteration: " << it << std::endl;
          std::cout << "      primal (sub problem): " << g + h << std::endl;
        }

        if (it % params.pgd_freq == 0) {
          VectorXd beta_old = beta;
          if (params.print_level > 2) {
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
                                  params);

          clusters.update(beta);

          max_update_inner = computeMaxDelta(
            x, x_centers, x_scales, w, beta_old, beta, params.standardize);

          // TODO: Consider changing this criterion to use the duality gap
          // instead.
          if (params.print_level > 2) {
            std::cout << "      max inner update change: " << max_update_inner
                      << ", tol: " << params.tol << std::endl;
          }

          if (max_update_inner <= params.tol) {
            break;
          }
        } else {
          if (params.print_level > 2) {
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
                            params);
        }
      }

      double max_update_outer = computeMaxDelta(
        x, x_centers, x_scales, w, beta_old_outer, beta, params.standardize);

      if (params.print_level > 1) {
        std::cout << "    max outer update change: " << max_update_outer
                  << ", tol: " << params.tol << std::endl;
      }

      if (max_update_outer <= params.tol) {
        break;
      }
    }

    double beta0_out;
    VectorXd beta_out;

    if (params.standardize) {
      std::tie(beta0_out, beta_out) = unstandardizeCoefficients(
        beta0, beta, x_centers, x_scales, params.intercept);
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

  return { beta0s, betas, alpha, lambda, primals };
}

} // namespace slope
