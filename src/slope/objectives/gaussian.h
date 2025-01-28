#pragma once

#include "objective.h"

namespace slope {
/**
 * @class Gaussian
 * @brief The Gaussian class represents a Gaussian objective function.
 */
class Gaussian : public Objective
{
public:
  /**
   * @brief Calculates the loss function for the Gaussian objective.
   * @param eta The predicted values.
   * @param y The actual values.
   * @return The loss value.
   */
  double loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y);

  /**
   * @brief Calculates the dual function for the Gaussian objective.
   * @param theta The dual variables.
   * @param y The actual values.
   * @param w The observation weights.
   * @return The dual value.
   */
  double dual(const Eigen::VectorXd& theta,
              const Eigen::VectorXd& y,
              const Eigen::VectorXd& w);

  /**
   * @brief Calculates the residuals for the Gaussian objective.
   * @param eta The predicted values.
   * @param y The actual values.
   * @return The residuals.
   */
  Eigen::VectorXd residual(const Eigen::VectorXd& eta,
                           const Eigen::VectorXd& y);

  /**
   * @brief Updates the weights and working response for the Gaussian objective.
   * @param w The weights.
   * @param z The working response.
   * @param eta The predicted values.
   * @param y The actual values.
   */
  void updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                       Eigen::VectorXd& z,
                                       const Eigen::VectorXd& eta,
                                       const Eigen::MatrixXd& y);
};

} // namespace slope
