#pragma once

#include "gaussian.h"

namespace slope {

double
Gaussian::loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y)
{
  const int n = y.rows();
  return (eta - y.col(0)).squaredNorm() / (2.0 * n);
}

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

Eigen::VectorXd
Gaussian::residual(const Eigen::VectorXd& eta, const Eigen::VectorXd& y)
{
  return y - eta;
}

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
