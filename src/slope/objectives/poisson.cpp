#include "poisson.h"

namespace slope {

double
Poisson::loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return -(y.col(0).array() * eta.col(0).array() - eta.col(0).array().exp())
            .mean();
}

double
Poisson::dual(const Eigen::MatrixXd& theta,
              const Eigen::MatrixXd& y,
              const Eigen::VectorXd& w)
{
  const Eigen::ArrayXd r = y.col(0) - theta.col(0);
  return -(r * (r.log() - 1.0)).mean();
}

Eigen::MatrixXd
Poisson::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return y.col(0).array() - eta.col(0).array().exp();
}

void
Poisson::updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                         Eigen::VectorXd& z,
                                         const Eigen::VectorXd& eta,
                                         const Eigen::VectorXd& y)
{
  w = eta.array().exp();
  z = eta.array() - 1.0 + y.array() / w.array();
}

} // namespace slope
