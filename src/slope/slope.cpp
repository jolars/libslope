#include "slope.h"
#include "clusters.h"
#include "constants.h"
#include "helpers.h"
#include "math.h"
#include "objectives/objective.h"
#include "objectives/setup_objective.h"
#include "regularization_sequence.h"
#include "solvers/hybrid.h"
#include "solvers/pgd.h"
#include "sorted_l1_norm.h"
#include "standardize.h"
#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <set>

namespace slope {

void
Slope::reset()
{
  this->dual_gaps_path.clear();
  this->primals_path.clear();
  this->betas.clear();
  this->beta0s.clear();
  this->it_total = 0;
}

/**
 * Fits a SLOPE model to the given data.
 *
 * This method fits a regularized regression model using the SLOPE penalty,
 * which combines the benefits of Lasso-type regularization with FDR control.
 * The algorithm can handle multiple objective functions and automatically
 * generates regularization paths if not provided.
 *
 * @param x The design matrix, where each column represents a feature
 * @param y The response vector
 * @param alpha Sequence of multipliers for the sorted L1 norm. If empty,
 *             automatically generated
 * @param lambda Weights for the sorted L1 norm. If empty, computed using
 *              the specified lambda_type
 */
template<typename SolverType, typename T>
void
Slope::fit(T& x,
           const Eigen::MatrixXd& y_in,
           Eigen::ArrayXd alpha,
           Eigen::ArrayXd lambda)
{
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  const int n = x.rows();
  const int p = x.cols();

  reset(); // Reset all internal variables

  if (n != y_in.rows()) {
    throw std::invalid_argument("x and y must have the same number of rows");
  }

  const bool sparse_x = std::is_base_of_v<Eigen::SparseMatrixBase<T>, T>;
  const bool standardize_jit =
    (!sparse_x && this->standardize && !this->modify_x) ||
    (sparse_x && this->standardize);

  auto [x_centers, x_scales] = computeCentersAndScales(x, this->standardize);

  if (this->standardize && this->modify_x && !sparse_x) {
    standardizeFeatures(x, x_centers, x_scales);
  }

  std::unique_ptr<Objective> objective = setupObjective(this->objective);

  MatrixXd y = objective->preprocessResponse(y_in);

  const int m = y.cols();

  beta0.resize(m);
  beta0.setZero();
  beta.resize(p, m);
  beta.setZero();
  MatrixXd eta = MatrixXd::Zero(n, m); // linear predictor

  if (lambda.size() == 0) {
    lambda = lambdaSequence(p * m, this->q, this->lambda_type);
  } else {
    if (lambda.size() != p * m) {
      throw std::invalid_argument(
        "lambda must be the same length as the number of coefficients");
    }
    if (lambda.minCoeff() < 0) {
      throw std::invalid_argument("lambda must be non-negative");
    }
    if (!lambda.isFinite().all()) {
      throw std::invalid_argument("lambda must be finite");
    }
  }

  // Setup the regularization sequence and path
  SortedL1Norm sl1_norm{ lambda };

  auto solver = SolverType(this->tol,
                           this->max_it_inner,
                           standardize_jit,
                           this->print_level,
                           this->intercept,
                           this->update_clusters,
                           this->pgd_freq);

  if (alpha.size() == 0) {
    alpha = regularizationPath(x,
                               y,
                               x_centers,
                               x_scales,
                               sl1_norm,
                               this->path_length,
                               this->alpha_min_ratio,
                               this->intercept,
                               standardize_jit);
  } else {
    if (alpha.minCoeff() < 0) {
      throw std::invalid_argument("alpha must be non-negative");
    }
    if (!alpha.isFinite().all()) {
      throw std::invalid_argument("alpha must be finite");
    }
    this->path_length = alpha.size();
  }

  std::vector<double> dual_gaps;
  std::vector<double> primals;

  // TODO: We should not do this for all solvers.
  Clusters clusters(beta.reshaped());

  this->it_total = 0;

  // Regularization path loop
  for (int path_step = 0; path_step < this->path_length; ++path_step) {
    if (this->print_level > 0) {
      std::cout << "Path step: " << path_step << ", alpha: " << alpha(path_step)
                << std::endl;
    }

    sl1_norm.setAlpha(alpha(path_step));

    for (int it = 0; it < this->max_it; ++it) {
      // Compute primal, dual, and gap
      double primal = objective->loss(eta, y) + sl1_norm.eval(beta);
      primals.emplace_back(primal);

      MatrixXd residual = objective->residual(eta, y);
      MatrixXd gradient = computeGradient(x,
                                          residual,
                                          x_centers,
                                          x_scales,
                                          Eigen::VectorXd::Ones(n),
                                          standardize_jit);
      MatrixXd theta = residual;

      // First compute gradient with potential offset for intercept case
      MatrixXd dual_gradient = gradient;
      if (this->intercept) {
        VectorXd theta_mean = theta.colwise().mean();
        theta.rowwise() -= theta_mean.transpose();

        MatrixXd gradient_offset = computeGradientOffset(
          x, theta_mean, x_centers, x_scales, standardize_jit);
        dual_gradient = gradient + gradient_offset;
      }

      // Common scaling operation
      double dual_norm = sl1_norm.dualNorm(dual_gradient);
      theta.array() /= std::max(1.0, dual_norm);

      double dual = objective->dual(theta, y, Eigen::VectorXd::Ones(n));

      double dual_gap = primal - dual;

      assert(dual_gap > -1e-5 && "Dual gap should be positive");

      dual_gaps.emplace_back(dual_gap);

      double tol_scaled = (std::abs(primal) + EPSILON) * this->tol;

      if (this->print_level > 1) {
        std::cout << indent(1) << "Outer iteration: " << it << std::endl
                  << indent(2) << "primal (main problem): " << primal
                  << std::endl
                  << indent(2) << "duality gap (main problem): " << dual_gap
                  << ", tol: " << tol_scaled << std::endl;
      }

      if (std::max(dual_gap, 0.0) <= tol_scaled) {
        break;
      }

      solver.run(beta0,
                 beta,
                 eta,
                 clusters,
                 objective,
                 sl1_norm,
                 gradient,
                 x,
                 x_centers,
                 x_scales,
                 y);
    }

    // Store everything for this step of the path
    auto [beta0_out, beta_out] = rescaleCoefficients(
      beta0, beta, x_centers, x_scales, intercept, standardize);

    std::vector<Eigen::Triplet<double>> beta_triplets;

    for (int k = 0; k < m; ++k) {
      for (int j = 0; j < p; ++j) {
        if (beta_out(j, k) != 0) {
          beta_triplets.emplace_back(j, k, beta_out(j, k));
        }
      }
    }

    Eigen::SparseMatrix<double> beta_out_sparse(p, m);
    beta_out_sparse.setFromTriplets(beta_triplets.begin(), beta_triplets.end());

    beta0s.emplace_back(beta0_out);
    betas.emplace_back(beta_out_sparse);

    primals_path.emplace_back(primals);
    dual_gaps_path.emplace_back(dual_gaps);
  }

  alpha_out = alpha;
  lambda_out = lambda;
}

void
Slope::setIntercept(bool intercept)
{
  this->intercept = intercept;
}

void
Slope::setStandardize(bool standardize)
{
  this->standardize = standardize;
}

void
Slope::setUpdateClusters(bool update_clusters)
{
  this->update_clusters = update_clusters;
}

void
Slope::setAlphaMinRatio(double alpha_min_ratio)
{
  if (alpha_min_ratio <= 0 || alpha_min_ratio >= 1) {
    throw std::invalid_argument("alpha_min_ratio must be in (0, 1)");
  }
  this->alpha_min_ratio = alpha_min_ratio;
}

void
Slope::setLearningRateDecr(double learning_rate_decr)
{
  if (learning_rate_decr <= 0 || learning_rate_decr >= 1) {
    throw std::invalid_argument("learning_rate_decr must be in (0, 1)");
  }
  this->learning_rate_decr = learning_rate_decr;
}

void
Slope::setQ(double q)
{
  if (q < 0 || q > 1) {
    throw std::invalid_argument("q must be between 0 and 1");
  }
  this->q = q;
}

void
Slope::setTol(double tol)
{
  if (tol < 0) {
    throw std::invalid_argument("tol must be non-negative");
  }
  this->tol = tol;
}

void
Slope::setMaxIt(int max_it)
{
  if (max_it < 1) {
    throw std::invalid_argument("max_it_outer must be >= 1");
  }
  this->max_it = max_it;
}

void
Slope::setMaxItInner(int max_it_inner)
{
  if (max_it_inner < 1) {
    throw std::invalid_argument("max_it_inner must be >= 1");
  }
  this->max_it_inner = max_it_inner;
}

void
Slope::setPathLength(int path_length)
{
  if (path_length < 1) {
    throw std::invalid_argument("path_length must be >= 1");
  }
  this->path_length = path_length;
}

void
Slope::setPgdFreq(int pgd_freq)
{
  if (pgd_freq < 1) {
    throw std::invalid_argument("pgd_freq must be > 1");
  }
  this->pgd_freq = pgd_freq;
}

void
Slope::setPrintLevel(int print_level)
{
  if (print_level < 0) {
    throw std::invalid_argument("print_level must be >= 0");
  }
  this->print_level = print_level;
}

void
Slope::setLambdaType(const std::string& lambda_type)
{
  validateOption(
    lambda_type, { "bh", "gaussian", "oscar", "lasso" }, "lambda_type");

  this->lambda_type = lambda_type;
}

void
Slope::setObjective(const std::string& objective)
{
  validateOption(objective,
                 { "gaussian", "binomial", "poisson", "multinomial" },
                 "objective");
  this->objective = objective;
}

void
Slope::setModifyX(const bool modify_x)
{
  this->modify_x = modify_x;
}

const Eigen::ArrayXd&
Slope::getAlpha() const
{
  return alpha_out;
}

const Eigen::ArrayXd&
Slope::getLambda() const
{
  return lambda_out;
}

const std::vector<Eigen::SparseMatrix<double>>
Slope::getCoefs() const
{
  return betas;
}

const std::vector<Eigen::VectorXd>
Slope::getIntercepts() const
{
  return beta0s;
}

int
Slope::getTotalIterations() const
{
  return it_total;
}

const std::vector<std::vector<double>>&
Slope::getDualGaps() const
{
  return dual_gaps_path;
}

const std::vector<std::vector<double>>&
Slope::getPrimals() const
{
  return primals_path;
}

// Explicit instantiations for common matrix/solver combinations
template void
Slope::fit<solvers::Hybrid, Eigen::MatrixXd>(Eigen::MatrixXd&,
                                             const Eigen::MatrixXd&,
                                             Eigen::ArrayXd,
                                             Eigen::ArrayXd);

template void
Slope::fit<solvers::Hybrid, Eigen::SparseMatrix<double>>(
  Eigen::SparseMatrix<double>&,
  const Eigen::MatrixXd&,
  Eigen::ArrayXd,
  Eigen::ArrayXd);

template void
Slope::fit<solvers::PGD, Eigen::MatrixXd>(Eigen::MatrixXd&,
                                          const Eigen::MatrixXd&,
                                          Eigen::ArrayXd,
                                          Eigen::ArrayXd);

template void
Slope::fit<solvers::PGD, Eigen::SparseMatrix<double>>(
  Eigen::SparseMatrix<double>&,
  const Eigen::MatrixXd&,
  Eigen::ArrayXd,
  Eigen::ArrayXd);

// slope.cpp

} // namespace slope
