/**
 * @file
 * @brief The actual function that fits SLOPE
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cassert>
#include <optional>
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
    , modify_x(false)
    , standardize(true)
    , update_clusters(false)
    , alpha_min_ratio(-1) // TODO: Use std::optional for alpha_min_ratio
    , dev_change_tol(1e-5)
    , dev_ratio_tol(0.999)
    , learning_rate_decr(0.5)
    , q(0.1)
    , tol(1e-4)
    , max_it(1e4)
    , max_it_inner(1e6)
    , path_length(100)
    , pgd_freq(10)
    , print_level(0)
    , max_clusters(std::optional<int>())
    , lambda_type("bh")
    , objective("gaussian")
    , screening_type("strong")
    , solver_type("hybrid")
  {
  }

  /**
   * @brief Sets the intercept flag.
   *
   * @param intercept Should an intercept be fitted?
   */
  void setSolver(const std::string& solver);

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
   * @brief Sets the type of feature screening used, which discards predictors
   * that are unlikely to be active.
   *
   * @param screening_type Type of screening. Supported values are:
   * are:
   *   - "strong": Strong screening rule ()
   *   - "none": No screening
   */
  void setScreening(const std::string& screening_type);

  /**
   * @brief Controls if `x` should be modified-in-place.
   * @details If `true`, then `x` will be modified in place if
   *   it is standardized. In case when `x` is dense, it will be both
   *   centered and scaled. If `x` is sparse, it will be only scaled.
   * @param modify_x Whether to modfiy `x` in place or not
   */
  void setModifyX(const bool objective);

  /**
   * @brief Sets tolerance in deviance change for early stopping.
   * @param dev_change_tol The tolerance for the change in deviance.
   */
  void setDevChangeTol(const double dev_change_tol);

  /**
   * @brief Sets tolerance in deviance change for early stopping.
   * @param dev_ratio_tol The tolerance for the dev ratio. If the deviance
   * exceeds this value, the path will terminate.
   */
  void setDevRatioTol(const double dev_ratio_tol);

  /**
   * @brief Sets tolerance in deviance change for early stopping.
   * @param max_clusters The maximum number of clusters. SLOPE
   * can (theoretically) select at most select min(n, p) clusters (unique
   * non-zero betas). By default, this is set to -1, which means that the number
   * of clusters will be automatically set to the number of observations + 1.
   */
  void setMaxClusters(const int max_clusters);

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
   * Get the deviances
   *
   * @return Get the primal objective values from the path.
   */
  const std::vector<double>& getDeviances() const;

  /**
   * Get the deviance for the null model
   *
   * @return Get the primal objective values from the path.
   */
  const double& getNullDeviance() const;

  // Declaration of the templated fit() method.
  template<typename T>
  void fit(T& x,
           const Eigen::MatrixXd& y_in,
           Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(0),
           Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(0));

private:
  // Reset the output values, but not the current coefficients
  void reset();

  // parameters
  bool intercept;
  bool modify_x;
  bool standardize;
  bool update_clusters;
  double alpha_min_ratio;
  double dev_change_tol;
  double dev_ratio_tol;
  double learning_rate_decr;
  double q;
  double tol;
  int max_it;
  int max_it_inner;
  int path_length;
  int pgd_freq;
  int print_level;
  std::optional<int> max_clusters;
  std::string lambda_type;
  std::string objective;
  std::string screening_type;
  std::string solver_type;

  // estimates
  Eigen::ArrayXd alpha_out;
  Eigen::ArrayXd lambda_out;
  Eigen::MatrixXd beta;
  Eigen::VectorXd beta0;
  double null_deviance;
  int it_total;
  std::vector<Eigen::SparseMatrix<double>> betas;
  std::vector<Eigen::VectorXd> beta0s;
  std::vector<double> deviances;
  std::vector<std::vector<double>> dual_gaps_path;
  std::vector<std::vector<double>> primals_path;
};

} // namespace slope
