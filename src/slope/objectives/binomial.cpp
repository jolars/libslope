#include "binomial.h"
#include "../math.h"

namespace slope {

double
Binomial::loss(const Eigen::VectorXd& eta, const Eigen::MatrixXd& y)
{
  double loss = eta.array().exp().log1p().sum() - y.col(0).dot(eta);
  return loss / y.rows();
}

double
Binomial::dual(const Eigen::VectorXd& theta,
               const Eigen::VectorXd& y,
               const Eigen::VectorXd& w)
{
  using Eigen::log;

  const int n = y.rows();

  Eigen::ArrayXd eta =
    (y - theta).array().log() - (1.0 - y.array() + theta.array()).log();
  Eigen::ArrayXd x = eta + y.array();

  return (x * x.log() + (1.0 - x) * (1.0 - x).log()).sum() / n;
}

Eigen::VectorXd
Binomial::residual(const Eigen::VectorXd& eta, const Eigen::VectorXd& y)
{
  return y.array() - 1.0 / (1.0 + (-eta).array().exp());
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

} // namespace slope
