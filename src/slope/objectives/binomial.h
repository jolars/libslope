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
  double p_min = 1e-9; /**< The minimum probability value. */

public:
  explicit Binomial()
    : Objective(0.25)
  {
  }

  /**
   * @brief Calculates the loss for the binomial objective function.
   * @param eta The predicted values.
   * @param y The true labels.
   * @return The loss value.
   */
  double loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y);

  /**
   * @brief Calculates the dual for the binomial objective function.
   * @param theta The dual variables.
   * @param y The true labels.
   * @return The dual value.
   */
  double dual(const Eigen::MatrixXd& theta,
              const Eigen::MatrixXd& y,
              const Eigen::VectorXd& w);

  /**
   * @brief Calculates the residual for the binomial objective function.
   * @param eta The predicted values.
   * @param y The true labels.
   * @return The residual vector.
   */
  Eigen::MatrixXd residual(const Eigen::MatrixXd& eta,
                           const Eigen::MatrixXd& y);

  /**
   * @brief Preprocesses the response for the Gaussian model
   * @details Checks if the response is in {0, 1} and converts it otherwise
   *
   * @param y Response vector (in {0,1})
   * @return Modified response.
   */
  Eigen::MatrixXd preprocessResponse(const Eigen::MatrixXd& y);

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
                                       const Eigen::VectorXd& y);

  /**
   * @brief The link function
   * @param eta Linear predictor.
   * @return The result of applyin the link function.
   */
  Eigen::MatrixXd link(const Eigen::MatrixXd& eta);
};

} // namespace slope
