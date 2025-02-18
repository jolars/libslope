#include "multinomial.h"
#include "../constants.h"
#include "../math.h"

namespace slope {

double
Multinomial::loss(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  int n = y.rows();

  double out = logSumExp(eta).mean();

  assert(out == out && "Loss is NaN");

  out -= (y.array() * eta.array()).sum() / n;

  return out;
}

double
Multinomial::dual(const Eigen::MatrixXd& theta,
                  const Eigen::MatrixXd& y,
                  const Eigen::VectorXd& w)
{
  const Eigen::MatrixXd r = y - theta;

  return -(r.array() * r.array().max(1e-9).log()).sum() / y.rows();
}

Eigen::MatrixXd
Multinomial::residual(const Eigen::MatrixXd& eta, const Eigen::MatrixXd& y)
{
  return y - softmax(eta);
}

Eigen::MatrixXd
Multinomial::preprocessResponse(const Eigen::MatrixXd& y)
{
  // Check if y is a column vector
  if (y.cols() != 1) {
    throw std::invalid_argument("Input must be a vector (n x 1)");
  }

  // Find number of observations and number of classes
  const int n = y.rows();
  const int m = y.array().maxCoeff() + 1; // Assuming classes are 0-based

  // Initialize result matrix with zeros
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, m);

  // Set 1s in appropriate positions
  for (int i = 0; i < n; i++) {
    int class_label = static_cast<int>(y(i, 0));
    if (class_label < 0) {
      throw std::invalid_argument(
        "Class labels must be consecutive integers starting from 0");
    }

    // Only set 1s for non-reference categories
    result(i, class_label) = 1.0;
  }

  return result;
}

void
Multinomial::updateWeightsAndWorkingResponse(Eigen::VectorXd& w,
                                             Eigen::VectorXd& z,
                                             const Eigen::VectorXd& eta,
                                             const Eigen::VectorXd& y)
{
  // Not implemented
  // TODO: Create not-implmented error, maybe as part of parent class.
}

Eigen::MatrixXd
Multinomial::link(const Eigen::MatrixXd& eta)
{
  return eta.unaryExpr([](const double& x) {
    return logit(std::clamp(x, constants::P_MIN, constants::P_MAX));
  });
}

// TODO: Consider adjusting the coefficients somehow.

} // namespace slope
