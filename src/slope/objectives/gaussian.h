/**
 * @file
 * @brief Gaussian objective function implementation for SLOPE algorithm
 * @details This file contains the Gaussian class which implements a Gaussian
 * objective function used in the SLOPE (Sorted L-One Penalized Estimation)
 * algorithm. The Gaussian objective function is particularly useful for
 * regression problems with normally distributed errors.
 */

#pragma once

#include "objective.h"

namespace slope {

/**
 * @class Gaussian
 * @brief Implementation of the Gaussian objective function
 * @details The Gaussian class provides methods for computing loss, dual
 * function, residuals, and weight updates for the Gaussian case in the SLOPE
 * algorithm. It is particularly suited for regression problems where the error
 * terms are assumed to follow a normal distribution.
 *
 * @note This class inherits from the base Objective class and implements
 * all required virtual functions.
 */
class Gaussian : public Objective
{
public:
  /**
   * @brief Calculates the Gaussian loss function value
   * @details Computes the squared error loss between predicted and actual
   * values, normalized by twice the number of observations.
   *
   * @param eta Vector of predicted values (n x 1)
   * @param y Matrix of actual values (n x 1)
   * @return Double precision loss value
   *
   * @note The loss is calculated as: \f$ \frac{1}{2n} \sum_{i=1}^n (\eta_i -
   * y_i)^2 \f$
   */
  double loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y);

  /**
   * @brief Computes the dual function for the Gaussian objective
   * @details Calculates the Fenchel conjugate of the Gaussian loss function
   *
   * @param theta Dual variables vector (n x 1)
   * @param y Observed values vector (n x 1)
   * @param w Observation weights vector (n x 1)
   * @return Double precision dual value
   *
   * @see loss() for the primal function
   */
  double dual(const Eigen::VectorXd& theta,
              const Eigen::VectorXd& y,
              const Eigen::VectorXd& w);

  /**
   * @brief Calculates residuals for the Gaussian model
   * @details Computes the difference between observed and predicted values
   *
   * @param eta Predicted values vector (n x 1)
   * @param y Actual values vector (n x 1)
   * @return Vector of residuals (n x 1)
   *
   * @note Residuals are calculated as: \f$ r_i = y_i - \eta_i \f$
   */
  Eigen::VectorXd residual(const Eigen::VectorXd& eta,
                           const Eigen::VectorXd& y);

  /**
   * @brief Updates weights and working response for IRLS algorithm
   * @details For Gaussian case, weights are set to 1 and working response
   * equals the original response. This implementation is particularly simple
   * compared to other GLM families.
   *
   * @param[out] w Weights vector to be updated (n x 1)
   * @param[out] z Working response vector to be updated (n x 1)
   * @param[in] eta Current predictions vector (n x 1)
   * @param[in] y Matrix of observed values (n x 1)
   *
   * @note For Gaussian regression, this is particularly simple as weights
   * remain constant and working response equals the original response
   */
  void updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                       Eigen::VectorXd& z,
                                       const Eigen::VectorXd& eta,
                                       const Eigen::MatrixXd& y);
};

} // namespace slope
