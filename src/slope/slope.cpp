#include "slope.h"
#include "clusters.h"
#include "constants.h"
#include "helpers.h"
#include "kkt_check.h"
#include "math.h"
#include "objectives/objective.h"
#include "objectives/setup_objective.h"
#include "regularization_sequence.h"
#include "screening.h"
#include "solvers/setup_solver.h"
#include "sorted_l1_norm.h"
#include "standardize.h"
#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <numeric>
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
template<typename T>
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

  std::vector<int> full_set(p);
  std::iota(full_set.begin(), full_set.end(), 0);

  std::unique_ptr<Objective> objective = setupObjective(this->objective);

  MatrixXd y = objective->preprocessResponse(y_in);

  const int m = y.cols();

  beta0.resize(m);
  beta0.setZero();
  beta.resize(p, m);
  beta.setZero();
  MatrixXd eta = MatrixXd::Zero(n, m); // linear predictor
  MatrixXd residual = objective->residual(eta, y);
  MatrixXd gradient(p, m);

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

  // TODO: Make this part of the slope class
  auto solver = setupSolver(this->solver_type,
                            this->objective,
                            this->tol,
                            this->max_it_inner,
                            standardize_jit,
                            this->print_level,
                            this->intercept,
                            this->update_clusters,
                            this->pgd_freq);

  updateGradient(gradient,
                 x,
                 residual,
                 full_set,
                 x_centers,
                 x_scales,
                 Eigen::VectorXd::Ones(n),
                 standardize_jit);

  int alpha_max_ind = whichMax(gradient.reshaped().cwiseAbs());
  alpha_max_ind = alpha_max_ind % p;

  double alpha_max;
  std::tie(alpha, alpha_max, this->path_length) =
    regularizationPath(alpha,
                       gradient,
                       sl1_norm,
                       n,
                       this->path_length,
                       this->alpha_min_ratio,
                       this->intercept,
                       standardize_jit);

  // Screening stuff
  std::vector<int> strong_set, previous_set, working_set, inactive_set;
  if (this->screening_type == "none") {
    working_set = full_set;
  } else {
    working_set = { alpha_max_ind };
  }

  std::vector<double> dual_gaps, primals;

  // TODO: We should not do this for all solvers.
  Clusters clusters(beta.reshaped());

  double alpha_prev = std::max(alpha_max, alpha(0));

  this->it_total = 0;

  // Regularization path loop
  for (int path_step = 0; path_step < this->path_length; ++path_step) {
    if (this->print_level > 0) {
      std::cout << "Path step: " << path_step << ", alpha: " << alpha(path_step)
                << std::endl;
    }

    double alpha_curr = alpha(path_step);

    assert(alpha_curr <= alpha_prev && "Alpha must be decreasing");

    sl1_norm.setAlpha(alpha_curr);

    Eigen::ArrayXd lambda_curr = alpha_curr * lambda;
    Eigen::ArrayXd lambda_prev = alpha_prev * lambda;

    if (screening_type == "strong") {
      // TODO: Only update for inactive set, making sure gradient
      // is updated for the active set
      updateGradient(gradient,
                     x,
                     residual,
                     full_set,
                     x_centers,
                     x_scales,
                     Eigen::VectorXd::Ones(n),
                     standardize_jit);

      previous_set = previouslyActiveSet(beta);
      strong_set = strongSet(gradient, lambda_curr, lambda_prev);
      strong_set = setUnion(strong_set, previous_set);
      working_set = setUnion(previous_set, { alpha_max_ind });
    }

    for (int it = 0; it < this->max_it; ++it) {
      assert(it < this->max_it - 1 && "Exceeded maximum number of iterations");
      // Compute primal, dual, and gap
      residual = objective->residual(eta, y);
      updateGradient(gradient,
                     x,
                     residual,
                     working_set,
                     x_centers,
                     x_scales,
                     Eigen::VectorXd::Ones(n),
                     standardize_jit);

      double primal = objective->loss(eta, y) +
                      sl1_norm.eval(beta(working_set, Eigen::all).reshaped());
      primals.emplace_back(primal);

      MatrixXd theta = residual;

      // First compute gradient with potential offset for intercept case
      MatrixXd dual_gradient = gradient;

      // TODO: Can we avoid this copy? Maybe revert offset afterwards or,
      // alternatively, solve intercept until convergence and then no longer
      // need the offset at all.
      if (this->intercept) {
        VectorXd theta_mean = theta.colwise().mean();
        theta.rowwise() -= theta_mean.transpose();

        offsetGradient(dual_gradient,
                       x,
                       theta_mean,
                       working_set,
                       x_centers,
                       x_scales,
                       standardize_jit);
      }

      // Common scaling operation
      double dual_norm =
        sl1_norm.dualNorm(dual_gradient(working_set, Eigen::all).reshaped());
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
        if (screening_type == "strong") {
          updateGradient(gradient,
                         x,
                         residual,
                         strong_set,
                         x_centers,
                         x_scales,
                         Eigen::VectorXd::Ones(n),
                         standardize_jit);

          auto violations = setDiff(
            kktCheck(gradient, beta, lambda_curr, strong_set), working_set);

          if (this->print_level > 1) {
            std::cout << indent(2) << "N active: " << working_set.size()
                      << std::endl
                      << indent(2) << "KKT violations: " << violations.size()
                      << std::endl;
          }

          if (violations.empty()) {
            updateGradient(gradient,
                           x,
                           residual,
                           full_set,
                           x_centers,
                           x_scales,
                           Eigen::VectorXd::Ones(n),
                           standardize_jit);

            violations = setDiff(
              kktCheck(gradient, beta, lambda_curr, full_set), working_set);
            if (violations.empty()) {
              break;
            } else {
              working_set = setUnion(working_set, violations);
            }
          } else {
            working_set = setUnion(working_set, violations);
          }
        } else {
          break;
        }
      }

      solver->run(beta0,
                  beta,
                  eta,
                  clusters,
                  objective,
                  sl1_norm,
                  gradient,
                  working_set,
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

    alpha_prev = alpha_curr;
  }

  alpha_out = alpha;
  lambda_out = lambda;
}

void
Slope::setSolver(const std::string& solver)
{
  validateOption(solver, { "pgd", "hybrid" }, "solver");
  this->solver_type = solver;
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
Slope::setScreening(const std::string& screening_type)
{
  validateOption(screening_type, { "strong", "none" }, "screening_type");
  this->screening_type = screening_type;
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

// Explicit instantiations for dense and sparse matrices
template void
Slope::fit<Eigen::MatrixXd>(Eigen::MatrixXd&,
                            const Eigen::MatrixXd&,
                            Eigen::ArrayXd,
                            Eigen::ArrayXd);

template void
Slope::fit<Eigen::SparseMatrix<double>>(Eigen::SparseMatrix<double>&,
                                        const Eigen::MatrixXd&,
                                        Eigen::ArrayXd,
                                        Eigen::ArrayXd);

// slope.cpp

} // namespace slope
