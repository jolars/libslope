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
               const Eigen::VectorXd& w)
{
  using Eigen::log;

  // Clamp probabilities to [p_min, 1-p_min]
  Eigen::ArrayXd pr = (y - theta).array().min(1.0 - p_min).max(p_min);

  return -(pr * log(pr) + (1.0 - pr) * log(1.0 - pr)).mean();
}

Eigen::MatrixXd
Binomial::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return y.array() - 1.0 / (1.0 + (-eta).array().exp());
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
  const int n = y.rows();

  for (int i = 0; i < n; ++i) {
    double p_i = sigmoid(eta(i, 0));
    p_i = clamp(p_i, p_min, 1.0 - p_min);
    w(i, 0) = p_i * (1.0 - p_i);
    z(i, 0) = eta(i, 0) + (y(i, 0) - p_i) / w(i, 0);
  }
}

Eigen::MatrixXd
Binomial::link(const Eigen::MatrixXd& eta)
{
  return eta.unaryExpr([](const double& x) {
    return logit(std::clamp(x, constants::P_MIN, constants::P_MAX));
  });
}

// double
// Binomial::nullDeviance(const Eigen::MatrixXd& y, const bool intercept)
// {
//   double beta0 = intercept ? logit(y.mean()) : 0.0;
//
//   Eigen::MatrixXd eta(y.rows(), y.cols());
//   eta.setConstant(beta0);
//
//   return deviance(eta, y);
// }

} // namespace slope
