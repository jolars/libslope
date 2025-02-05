#include "gaussian.h"

namespace slope {

double
Gaussian::loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return (eta - y).squaredNorm() / (2.0 * y.rows());
}

double
Gaussian::dual(const Eigen::MatrixXd& theta,
               const Eigen::MatrixXd& y,
               const Eigen::VectorXd& w)
{
  const int n = y.rows();

  return (y.squaredNorm() - (y - theta).squaredNorm()) / (2.0 * n);
}

Eigen::MatrixXd
Gaussian::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return y - eta;
}

Eigen::MatrixXd
Gaussian::preprocessResponse(const Eigen::MatrixXd& y)
{
  return y;
}

void
Gaussian::updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                          Eigen::VectorXd& z,
                                          const Eigen::VectorXd& eta,
                                          const Eigen::VectorXd& y)
{
  w.setOnes();
  z = y;
}

} // namespace slope
