#include "poisson.h"

namespace slope {

double
Poisson::loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y)
{
  return -(y.array() * eta.array() - eta.array().exp()).mean();
}

double
Poisson::dual(const Eigen::VectorXd& theta,
              const Eigen::VectorXd& y,
              const Eigen::VectorXd& w)
{
  const Eigen::ArrayXd r = y - theta;
  return -(r * (r.log() - 1.0)).mean();
}

Eigen::VectorXd
Poisson::residual(const Eigen::VectorXd& eta, const Eigen::VectorXd& y)
{
  return y.array() - eta.array().exp();
}

void
Poisson::updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                         Eigen::VectorXd& z,
                                         const Eigen::VectorXd& eta,
                                         const Eigen::MatrixXd& y)
{
  w = eta.array().exp();
  z = eta.array() - 1.0 + y.array() / w.array();
}

} // namespace slope
