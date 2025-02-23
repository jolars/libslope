#include "math.h"

namespace slope {

Eigen::VectorXd
logSumExp(const Eigen::MatrixXd& a)
{
  // p_min prevents taking log(0) by ensuring a lower bound.
  const double p_min = 1e-9;

  // For numerical stability subtract each row's maximum.
  Eigen::VectorXd max_vals = a.rowwise().maxCoeff();

  // Compute the sum of exponentials of the shifted values.
  Eigen::ArrayXd sum_exp =
    (a.colwise() - max_vals).array().exp().rowwise().sum();

  // Return the stabilized log-sum-exp, ensuring the sum is at least p_min.
  return max_vals.array() + sum_exp.max(p_min).log();
}

Eigen::MatrixXd
softmax(const Eigen::MatrixXd& a)
{
  int m = a.cols();

  Eigen::VectorXd max_vals = a.rowwise().maxCoeff();

  Eigen::MatrixXd out = (a.colwise() - max_vals).array().exp();
  Eigen::ArrayXd row_sums = out.rowwise().sum();

  for (int k = 0; k < m; ++k) {
    out.col(k).array() /= row_sums;
  }

  return out;
}

std::vector<int>
setUnion(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> out;
  std::set_union(
    a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));

  return out;
}

std::vector<int>
setDiff(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> out;
  std::set_difference(
    a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));

  return out;
}

} // namespace slope
