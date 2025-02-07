#include "gaussian.h"

namespace slope {

double
Gaussian::loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  const int n = y.rows();
  return (eta.reshaped() - y.reshaped()).squaredNorm() / (2.0 * n);
}

double
Gaussian::dual(const Eigen::MatrixXd& theta,
               const Eigen::MatrixXd& y,
               const Eigen::VectorXd& w)
{
  const int n = y.rows();

  return theta.reshaped().cwiseProduct(w).dot(y.reshaped()) / n -
         0.5 * theta.reshaped().cwiseProduct(w.cwiseSqrt()).squaredNorm() / n;
}

Eigen::MatrixXd
Gaussian::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return y.reshaped() - eta.reshaped();
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
