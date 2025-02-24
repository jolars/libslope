#include "binomial.h"
#include "../constants.h"
#include "../math.h"

namespace slope {

double
Binomial::loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  double loss =
    eta.array().exp().log1p().sum() - y.reshaped().dot(eta.reshaped());
  return loss / y.rows();
}

double
Binomial::dual(const Eigen::MatrixXd& theta,
               const Eigen::MatrixXd& y,
               const Eigen::VectorXd&)
{
  using Eigen::log;

  // Clamp probabilities to [p_min, 1-p_min]
  Eigen::ArrayXd pr = (theta + y).array().min(1.0 - p_min).max(p_min);

  return -(pr * log(pr) + (1.0 - pr) * log(1.0 - pr)).mean();
}

Eigen::MatrixXd
Binomial::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return 1.0 / (1.0 + (-eta).array().exp()) - y.array();
}

Eigen::MatrixXd
Binomial::preprocessResponse(const Eigen::MatrixXd& y)
{
  // Check if the response is in {0, 1} and convert it otherwise
  Eigen::MatrixXd y_clamped = y.array().min(1.0).max(0.0);

  // Throw an error if the response is not binary
  if ((y_clamped.array() != 0.0 && y_clamped.array() != 1.0).any()) {
    throw std::invalid_argument("Response must be binary");
  }

  return y_clamped;
}

void
Binomial::updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                          Eigen::VectorXd& z,
                                          const Eigen::VectorXd& eta,
                                          const Eigen::VectorXd& y)
{
  Eigen::ArrayXd pr =
    (1.0 / (1.0 + (-eta.array()).exp())).min(1.0 - p_min).max(p_min);
  w = pr * (1.0 - pr);
  z = eta.array() + (y.array() - pr) / w.array();
}

Eigen::MatrixXd
Binomial::link(const Eigen::MatrixXd& eta)
{
  return eta.unaryExpr([](const double& x) {
    return logit(std::clamp(x, constants::P_MIN, constants::P_MAX));
  });
}

} // namespace slope
