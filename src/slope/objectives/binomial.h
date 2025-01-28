#pragma once

#include "objective.h"

namespace slope {
/**
 * @class Binomial
 * @brief The Binomial class represents a binomial objective function.
 * @details The binomial objective function is used for binary classification
 * problems. It calculates the loss, dual, residual, and updates weights and
 * working response.
 */
class Binomial : public Objective
{
private:
  double p_min = 1e-5; /**< The minimum probability value. */

public:
  /**
   * @brief Calculates the loss for the binomial objective function.
   * @param eta The predicted values.
   * @param y The true labels.
   * @return The loss value.
   */
  double loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y);

  /**
   * @brief Calculates the dual for the binomial objective function.
   * @param theta The dual variables.
   * @param y The true labels.
   * @return The dual value.
   */
  double dual(const Eigen::VectorXd& theta,
              const Eigen::VectorXd& y,
              const Eigen::VectorXd& w);

  /**
   * @brief Calculates the residual for the binomial objective function.
   * @param eta The predicted values.
   * @param y The true labels.
   * @return The residual vector.
   */
  Eigen::VectorXd residual(const Eigen::VectorXd& eta,
                           const Eigen::VectorXd& y);

  /**
   * @brief Updates the weights and working response for the binomial objective
   * function.
   * @param w The weights.
   * @param z The working response.
   * @param eta The predicted values.
   * @param y The true labels.
   */
  void updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                       Eigen::VectorXd& z,
                                       const Eigen::VectorXd& eta,
                                       const Eigen::MatrixXd& y);
};

} // namespace slope
