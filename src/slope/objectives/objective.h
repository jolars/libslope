/**
 * @file
 * @brief The declartion of the Objctive class and its subclasses, which
 * represent the data-fitting part of the composite objective function.
 */

#pragma once

#include <Eigen/Core>

namespace slope {

/**
 * Abstract class representing an objective function.
 *
 * This class defines the interface for an objective function, which is used in
 * optimization algorithms. The objective function calculates the loss, dual,
 * residual, and updates the weights and working response.
 */
class Objective
{
public:
  /**
   * @brief Destructor for the Objective class.
   */
  virtual ~Objective() = default;

  /**
   * @brief Calculates the loss function
   *
   * This function calculates the loss function given the predicted values (eta)
   * and the true values (y).
   *
   * @param eta The predicted values.
   * @param y The true values.
   * @return The loss value.
   */
  virtual double loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y) = 0;

  /**
   * @brief Calculates the dual objective
   *
   * This function calculates the dual function given the estimated parameters
   * (theta) and the true values (y).
   *
   * @param theta The estimated parameters.
   * @param y The true values.
   * @return The dual value.
   */
  virtual double dual(const Eigen::VectorXd& theta,
                      const Eigen::VectorXd& y,
                      const Eigen::VectorXd& w) = 0;

  /**
   * @brief Calculates the residual
   *
   * This function calculates the residual given the predicted values (eta) and
   * the true values (y).
   *
   * @param eta The predicted values.
   * @param y The true values.
   * @return The residual vector.
   */
  virtual Eigen::VectorXd residual(const Eigen::VectorXd& eta,
                                   const Eigen::VectorXd& y) = 0;

  /**
   * @brief Updates the weights and working response
   *
   * This function updates the weights and working response given the predicted
   * values (eta) and the true values (y).
   *
   * @param weights The weights to be updated.
   * @param working_response The working response to be updated.
   * @param eta The predicted values.
   * @param y The true values.
   */
  virtual void updateWeightsAndWorkingResponse(
    Eigen::VectorXd& weights,
    Eigen::VectorXd& working_response,
    const Eigen::VectorXd& eta,
    const Eigen::MatrixXd& y) = 0;
};

} // namespace slope
