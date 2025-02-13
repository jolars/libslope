/**
 * @file
 * @brief The actual function that fits SLOPE
 */

#pragma once

#include "clusters.h"
#include "constants.h"
#include "helpers.h"
#include "math.h"
#include "objectives/objective.h"
#include "objectives/setup_objective.h"
#include "regularization_sequence.h"
#include "solvers/hybrid.h"
#include "sorted_l1_norm.h"
#include "standardize.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace slope {

/**
 * Class representing SLOPE (Sorted L-One Penalized Estimation) optimization.
 *
 * This class implements the SLOPE algorithm for regularized regression
 * problems. It supports different loss functions (gaussian, binomial, poisson)
 * and provides functionality for fitting models with sorted L1 regularization
 * along a path of regularization parameters.
 */
class Slope
{
public:
  /**
   * Default constructor for the Slope class.
   *
   * Initializes the Slope object with default parameter values.
   */
  Slope()
    : intercept(true)
    , standardize(true)
    , update_clusters(false)
    , alpha_min_ratio(-1)
    , learning_rate_decr(0.5)
    , q(0.1)
    , tol(1e-4)
    , max_it_inner(1e6)
    , max_it(1e4)
    , path_length(100)
    , pgd_freq(10)
    , modify_x(false)
    , print_level(0)
    , lambda_type("bh")
    , objective("gaussian")
  {
  }

  /**
   * @brief Sets the intercept flag.
   *
   * @param intercept Should an intercept be fitted?
   */
  void setIntercept(bool intercept);

  /**
   * @brief Sets the standardize flag.
   *
   * @param standardize Should the design matrix be standardized?
   */
  void setStandardize(bool standardize);

  /**
   * @brief Sets the update clusters flag.
   *
   * @param update_clusters Selects whether the coordinate descent keeps the
   * clusters updated.
   */
  void setUpdateClusters(bool update_clusters);

  /**
   * @brief Sets the alpha min ratio.
   *
   * @param alpha_min_ratio The value to set for the alpha min ratio. A negative
   * value means that the program automatically chooses 1e-4 if the number of
   * observations is larger than the number of features and 1e-2 otherwise.
   */
  void setAlphaMinRatio(double alpha_min_ratio);

  /**
   * @brief Sets the learning rate decrement.
   *
   * @param learning_rate_decr The value to set for the learning rate decrement
   * for the proximal gradient descent step.
   */
  void setLearningRateDecr(double learning_rate_decr);

  /**
   * @brief Sets the q value.
   *
   * @param q The value to set for the q value for use in automatically
   * generating the lambda sequence. values between 0 and 1 are allowed..
   */
  void setQ(double q);

  /**
   * @brief Sets the tolerance value.
   *
   * @param tol The value to set for the tolerance value. Must be positive.
   */
  void setTol(double tol);

  /**
   * @brief Sets the maximum number of iterations.
   *
   * @param max_it The value to set for the maximum number of iterations. Must
   * be positive. If negative (the default), then the value will be decided by
   * the solver.
   */
  void setMaxIt(int max_it);

  /**
   * @brief Sets the maximum number of inner iterations.
   *
   * @param max_it Sets the maximum number of inner iterations for solvers
   * that use an inner loop. Must be positive.
   */
  void setMaxItInner(int max_it_inner);

  /**
   * @brief Sets the path length.
   *
   * @param path_length The value to set for the path length.
   */
  void setPathLength(int path_length);

  /**
   * @brief Sets the frequence of proximal gradient descent steps.
   *
   * @param pgd_freq The frequency of the proximal gradient descent steps (or
   * the inverse of that actually). A value of 1 means that the algorithm only
   * runs proximal gradient descent steps.
   */
  void setPgdFreq(int pgd_freq);

  /**
   * @brief Sets the print level.
   *
   * @param print_level The value to set for the print level. A print level of 1
   * prints values from the outer loop, a level of 2 from the inner loop, and a
   * level of 3 some extra debugging information. A level of 0 means no
   * printing.
   */
  void setPrintLevel(int print_level);

  /**
   * @brief Sets the lambda type for regularization weights.
   *
   * @param lambda_type The method used to compute regularization weights.
   * Currently "bh" (Benjamini-Hochberg), "gaussian", "oscar", and "lasso" are
   * supported.
   */
  void setLambdaType(const std::string& lambda_type);

  /**
   * @brief Sets the objective function type.
   *
   * @param objective The type of objective function to use. Supported values
   * are:
   *                 - "gaussian": Gaussian regression
   *                 - "binomial": Logistic regression
   *                 - "poisson": Poisson regression
   *                 - "multinomial": Multinomial logistic regression
   */
  void setObjective(const std::string& objective);

  /**
   * @brief Controls if `x` should be modified-in-place.
   * @details If `true`, then `x` will be modified in place if
   *   it is standardized. In case when `x` is dense, it will be both
   *   centered and scaled. If `x` is sparse, it will be only scaled.
   * @param modify_x Whether to modfiy `x` in place or not
   */
  void setModifyX(const bool objective);

  /**
   * @brief Get the alpha sequence.
   *
   * @return The sequence of weights for the regularization path.
   */
  const Eigen::ArrayXd& getAlpha() const;

  /**
   * @brief Get the lambda sequence.
   *
   * @return The sequence of lambda values for the weights of the sorted L1
   * norm.
   */
  const Eigen::ArrayXd& getLambda() const;

  /**
   * Get the coefficients from the path.
   *
   * @return The coefficients from the path, stored in a sparse matrix.
   */
  const std::vector<Eigen::SparseMatrix<double>> getCoefs() const;

  /**
   * Get the intercepts from the path.
   *
   * @return The coefficients from the path, stored in an Eigen vector. If no
   * intercepts were fit, this is a vector of zeros.
   */
  const std::vector<Eigen::VectorXd> getIntercepts() const;

  /**
   * Get the total number of (inner) iterations.
   *
   * @return The toral number of iterations from the inner loop, computed across
   * the path.
   */
  int getTotalIterations() const;

  /**
   * Get the duality gaps.
   *
   * @return Get the duality gaps from the path.
   */
  const std::vector<std::vector<double>>& getDualGaps() const;

  /**
   * Get the primal objective values.
   *
   * @return Get the primal objective values from the path.
   */
  const std::vector<std::vector<double>>& getPrimals() const;

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
  template<typename SolverType = solvers::Hybrid, typename T>
  void fit(T& x,
           const Eigen::MatrixXd& y_in,
           Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(0),
           Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(0))
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
        std::cout << "Path step: " << path_step
                  << ", alpha: " << alpha(path_step) << std::endl;
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
      beta_out_sparse.setFromTriplets(beta_triplets.begin(),
                                      beta_triplets.end());

      beta0s.emplace_back(beta0_out);
      betas.emplace_back(beta_out_sparse);

      primals_path.emplace_back(primals);
      dual_gaps_path.emplace_back(dual_gaps);
    }

    alpha_out = alpha;
    lambda_out = lambda;
  }

private:
  // Reset the output values, but not the current coefficients
  void reset();

  // parameters
  bool intercept;
  bool standardize;
  bool update_clusters;
  bool modify_x;
  double alpha_min_ratio;
  double learning_rate_decr;
  double q;
  double tol;
  int max_it;
  int max_it_inner;
  int path_length;
  int pgd_freq;
  int print_level;
  std::string lambda_type;
  std::string objective;

  // estimates
  Eigen::ArrayXd alpha_out;
  Eigen::ArrayXd lambda_out;
  std::vector<Eigen::SparseMatrix<double>> betas;
  std::vector<Eigen::VectorXd> beta0s;
  Eigen::MatrixXd beta;
  Eigen::VectorXd beta0;
  int it_total;
  std::vector<std::vector<double>> dual_gaps_path;
  std::vector<std::vector<double>> primals_path;
};

} // namespace slope
