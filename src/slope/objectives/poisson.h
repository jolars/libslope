/**
 * @file
 * @brief Poisson objective function implementation for SLOPE algorithm
 * @details This file contains the Gaussian class which implements a Poisson
 * objective function used in the SLOPE (Sorted L-One Penalized Estimation)
 * algorithm.
 */

#pragma once

#include "objective.h"

namespace slope {
/**
 * @class Poisson
 * @brief The Poisson class represents a Poisson regression objective function.
 * @details The Poisson regression objective function is used for modeling count
 * data. It assumes the response variable follows a Poisson distribution. The
 * log-likelihood for a single observation is:
 * \f[ \ell(y_i|\eta_i) = y_i\eta_i - e^{\eta_i} - \log(y_i!) \f]
 * where \f$\eta_i\f$ is the linear predictor and \f$y_i\f$ is the observed
 * count.
 */
class Poisson : public Objective
{

public:
  explicit Poisson()
    : Objective(std::numeric_limits<double>::infinity())
  {
  }
  /**
   * @brief Calculates the negative log-likelihood loss for the Poisson
   * regression.
   * @param eta The linear predictor vector \f$\eta\f$
   * @param y The observed counts vector \f$y\f$
   * @return The negative log-likelihood: \f$-\sum_i(y_i\eta_i - e^{\eta_i})\f$
   * (ignoring constant terms)
   */
  double loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y) override;

  /**
   * @brief Calculates the Fenchel conjugate (dual) of the Poisson loss.
   * @param theta The dual variables
   * @param y The observed counts vector
   * @param w The weights vector
   * @return The dual objective value
   */
  double dual(const Eigen::MatrixXd& theta,
              const Eigen::MatrixXd& y,
              const Eigen::VectorXd& w) override;

  /**
   * @brief Calculates the residual (negative gradient) for the Poisson
   * regression.
   * @param eta The linear predictor vector \f$\eta\f$
   * @param y The observed counts vector \f$y\f$
   * @return The residual vector: \f$y - e^{\eta}\f$
   */
  Eigen::MatrixXd residual(const Eigen::MatrixXd& eta,
                           const Eigen::MatrixXd& y) override;

  /**
   * @brief Updates the weights and working response for IRLS (Iteratively
   * Reweighted Least Squares).
   * @param w The weights vector: \f$w = e^{\eta}\f$
   * @param z The working response vector: \f$z = \eta + (y -
   * e^{\eta})/e^{\eta}\f$
   * @param eta The current linear predictor
   * @param y The observed counts vector
   */
  void updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                       Eigen::VectorXd& z,
                                       const Eigen::VectorXd& eta,
                                       const Eigen::VectorXd& y) override;

  /**
   * @brief Preprocesses the response for the Poisson model
   * @details Checks if the response is non-negative and throws an error
   * otherwise.
   *
   * @param y Predicted values vector (n x 1)
   * @return Vector of residuals (n x 1)
   */
  Eigen::MatrixXd preprocessResponse(const Eigen::MatrixXd& y) override;

  /**
   * @brief Updates the intercept with a
   * gradient descent update. Unlike the Gaussian and Binomial cases, the
   * Poisson regression intercept update is not quite as simple since the
   * gradient is not Lipschitz continuous. Instead we use
   * a backtracking line search here.
   * @param beta0 The current intercept
   * @param eta The current linear predictor
   * @param y The observed counts vector
   */
  void updateIntercept(Eigen::VectorXd& beta0,
                       const Eigen::MatrixXd& eta,
                       const Eigen::MatrixXd& y) override;
};

} // namespace slope
