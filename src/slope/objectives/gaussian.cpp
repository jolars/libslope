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

  return (y.squaredNorm() - (theta + y).squaredNorm()) / (2.0 * n);
}

Eigen::MatrixXd
Gaussian::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return eta - y;
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

Eigen::MatrixXd
Gaussian::link(const Eigen::MatrixXd& eta)
{
  return eta;
}

// double
// Gaussian::nullDeviance(const Eigen::MatrixXd& y, const bool intercept)
// {
//   double beta0 = intercept ? y.mean() : 0.0;
//
//   Eigen::MatrixXd eta(y.rows(), y.cols());
//   eta.setConstant(beta0);
//
//   return deviance(eta, y);
// }

} // namespace slope
