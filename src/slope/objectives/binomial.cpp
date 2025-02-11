#include "binomial.h"
#include "../math.h"

namespace slope {

double
Binomial::loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  double loss =
    eta.col(0).array().exp().log1p().sum() - y.col(0).dot(eta.col(0));
  return loss / y.rows();
}

double
Binomial::dual(const Eigen::MatrixXd& theta,
               const Eigen::MatrixXd& y,
               const Eigen::VectorXd& w)
{
  using Eigen::log;

  const int n = y.rows();

  // Clamp probabilities to [p_min, 1-p_min]
  Eigen::ArrayXd pr = (y - theta).array().min(1.0 - p_min).max(p_min);

  return -(pr * log(pr) + (1.0 - pr) * log(1.0 - pr)).mean();
}

Eigen::MatrixXd
Binomial::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return y.col(0).array() - 1.0 / (1.0 + (-eta).array().exp());
}

void
Binomial::updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                          Eigen::VectorXd& z,
                                          const Eigen::VectorXd& eta,
                                          const Eigen::VectorXd& y)
{
  const int n = y.rows();
  const int m = y.cols();

  for (int k = 0; k < m; ++k) {
    for (int i = 0; i < n; ++i) {
      double p_i = sigmoid(eta(i, k));
      p_i = clamp(p_i, p_min, 1.0 - p_min);
      w(i, k) = p_i * (1.0 - p_i);
      z(i, k) = eta(i, k) + (y(i, k) - p_i) / w(i, k);
    }
  }
}

} // namespace slope
