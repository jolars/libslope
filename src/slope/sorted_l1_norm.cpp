#include "sorted_l1_norm.h"

namespace slope {

SortedL1Norm::SortedL1Norm(const Eigen::ArrayXd& lambda)
  : lambda{ lambda }
{
}

double
SortedL1Norm::eval(const Eigen::MatrixXd& beta) const
{
  Eigen::ArrayXd beta_abs = beta.reshaped().array().abs();
  sort(beta_abs, true);
  return alpha * (beta_abs * lambda).sum();
}

Eigen::MatrixXd
SortedL1Norm::prox(const Eigen::MatrixXd& beta, const double scale) const
{
  using namespace Eigen;

  // const int p = beta.rows();
  const int m = beta.cols();

  ArrayXd beta_sign = beta.reshaped().array().sign();
  VectorXd beta_copy = beta.reshaped().array().abs();

  auto ord = sortIndex(beta_copy, true);
  permute(beta_copy, ord);

  int p = beta_copy.size();

  VectorXd s(p);
  VectorXd w(p);
  VectorXi idx_i(p);
  VectorXi idx_j(p);

  int k = 0;

  for (int i = 0; i < p; i++) {
    idx_i[k] = i;
    idx_j[k] = i;
    s[k] = beta_copy(i) - this->lambda(i) * this->alpha * scale;
    w[k] = s[k];

    while ((k > 0) && (w[k - 1] <= w[k])) {
      k--;
      idx_j[k] = i;
      s[k] += s[k + 1];
      w[k] = s[k] / (i - idx_i[k] + 1.0);
    }
    k++;
  }

  for (int j = 0; j < k; j++) {
    double d = std::max(w[j], 0.0);
    for (int i = idx_i[j]; i <= idx_j[j]; i++) {
      beta_copy(i) = d;
    }
  }

  // return order and sigsn
  inversePermute(beta_copy, ord);
  beta_copy.array() *= beta_sign;

  return beta_copy.reshaped(beta.rows(), beta.cols());
}

void
SortedL1Norm::setAlpha(const double new_alpha)
{
  alpha = new_alpha;
}

double
SortedL1Norm::getAlpha() const
{
  return alpha;
}

void
SortedL1Norm::setLambda(const Eigen::ArrayXd& new_lambda)
{
  lambda = new_lambda;
}

Eigen::ArrayXd
SortedL1Norm::getLambda() const
{
  return lambda;
}

const Eigen::ArrayXd&
SortedL1Norm::getLambdaRef() const
{
  return lambda;
}

double
SortedL1Norm::dualNorm(const Eigen::MatrixXd& gradient) const
{

  Eigen::ArrayXd abs_gradient = gradient.reshaped().array().abs();
  sort(abs_gradient, true);

  if (this->alpha == 0 || this->lambda.sum() == 0) {
    // TODO: this is a crude approach for the unregularized case.
    // We should consider something more clever and avoid
    // the division.
    return (cumSum(abs_gradient) / 1e-6).maxCoeff();
  }

  return (cumSum(abs_gradient) / (this->alpha * cumSum(this->lambda)))
    .maxCoeff();
}

} // namspace slope
