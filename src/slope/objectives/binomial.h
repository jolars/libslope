#pragma once

#include "objective.h"
#include <Eigen/Core>

namespace slope {

class Binomial : public Objective
{
private:
  double p_min = 1e-5;

public:
  double loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y);

  void updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                       Eigen::VectorXd& z,
                                       const Eigen::VectorXd& eta,
                                       const Eigen::MatrixXd& y);
};

}
