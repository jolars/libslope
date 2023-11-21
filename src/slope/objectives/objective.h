#pragma once

#include <Eigen/Core>
#include <memory>

namespace slope {

class Objective
{
public:
  virtual ~Objective() = default;

  virtual double loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y) = 0;

  virtual void updateWeightsAndWorkingResponse(
    Eigen::VectorXd& weights,
    Eigen::VectorXd& working_response,
    const Eigen::VectorXd& eta,
    const Eigen::MatrixXd& y) = 0;
};

} // namespace slope
