#include "poisson.h"

namespace slope {

double
Poisson::loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return -(y.array() * eta.array() - eta.array().exp()).mean();
}

double
Poisson::dual(const Eigen::MatrixXd& theta,
              const Eigen::MatrixXd& y,
              const Eigen::VectorXd& w)
{
  const Eigen::ArrayXd e = y - theta;

  assert((e >= 0).all() &&
         "Dual function is not defined for negative residuals");

  return (e * (1.0 - e.log())).mean();
}

Eigen::MatrixXd
Poisson::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return y.array() - eta.array().exp();
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

void
Poisson::updateIntercept(Eigen::VectorXd& beta0,
                         const Eigen::MatrixXd& eta,
                         const Eigen::MatrixXd& y)
{
  Eigen::VectorXd residual = this->residual(eta, y);
  double grad = -residual.mean();
  double hess = eta.array().exp().mean();

  beta0(0) -= grad / hess;
}

} // namespace slope
