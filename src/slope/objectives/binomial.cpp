#include "binomial.h"
#include "math.h"

namespace slope {

double
Binomial::loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y)
{
  double loss = eta.array().exp().log1p().sum() - y.col(0).dot(eta);
  return loss / y.rows();
}

void
Binomial::updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                          Eigen::VectorXd& z,
                                          const Eigen::VectorXd& eta,
                                          const Eigen::MatrixXd& y)
{
  for (int i = 0; i < eta.size(); ++i) {
    double p_i = sigmoid(eta(i));
    p_i = clamp(p_i, p_min, 1.0 - p_min);
    w(i) = p_i * (1.0 - p_i);
    z(i) = eta(i) + (y(i) - p_i) / w(i);
  }
}

}
