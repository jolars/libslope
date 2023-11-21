#include "gaussian.h"

namespace slope {

double
Gaussian::loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y)
{
  const int n = y.rows();
  return (0.5 / n) * (eta - y.col(0)).squaredNorm();
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

}
