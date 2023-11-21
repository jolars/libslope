#pragma once

#include "objective.h"
#include <Eigen/Core>

namespace slope {

class Gaussian : public Objective
{
public:
  double loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y);

  void updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                       Eigen::VectorXd& z,
                                       const Eigen::VectorXd& eta,
                                       const Eigen::MatrixXd& y);
};

}
