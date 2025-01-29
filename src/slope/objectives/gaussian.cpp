/**
 * @file
 * @brief Implementation of Gaussian loss function and related operations
 *
 * This file implements the Gaussian loss function and its related operations
 * for the SLOPE (Sorted L-One Penalized Estimation) algorithm.
 */

#include "gaussian.h"

namespace slope {

/**
 * @brief Calculates the Gaussian loss function
 *
 * @param eta The linear predictor
 * @param y The observed values (response matrix)
 * @return double The computed loss value
 */
double
Gaussian::loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y)
{
  const int n = y.rows();
  return (eta - y.col(0)).squaredNorm() / (2.0 * n);
}

/**
 * @brief Computes the dual function for the Gaussian loss
 *
 * @param theta The dual variables
 * @param y The observed values
 * @param w The weights vector
 * @return double The computed dual value
 */
double
Gaussian::dual(const Eigen::VectorXd& theta,
               const Eigen::VectorXd& y,
               const Eigen::VectorXd& w)
{
  const int n = y.rows();
  const Eigen::VectorXd eta = y - theta;
  const Eigen::VectorXd w_sqrt = w.cwiseSqrt();

  return (y.cwiseProduct(w_sqrt).squaredNorm() -
          eta.cwiseProduct(w_sqrt).squaredNorm()) /
         (2.0 * n);
}

/**
 * @brief Calculates the residual (difference between observed and predicted
 * values)
 *
 * @param eta The linear predictor
 * @param y The observed values
 * @return Eigen::VectorXd The residual vector
 */
Eigen::VectorXd
Gaussian::residual(const Eigen::VectorXd& eta, const Eigen::VectorXd& y)
{
  return y - eta;
}

/**
 * @brief Updates the weights and working response for the Gaussian case
 *
 * For Gaussian regression, weights are set to 1 and working response equals the
 * original response
 *
 * @param w Output parameter for weights
 * @param z Output parameter for working response
 * @param eta The predicted values
 * @param y The observed values
 */
void
Gaussian::updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                          Eigen::VectorXd& z,
                                          const Eigen::VectorXd& eta,
                                          const Eigen::MatrixXd& y)
{
  w.setOnes();
  z = y;
}

} // namespace slope
