#include "standardize.h"

namespace slope {

std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
rescaleCoefficients(Eigen::VectorXd beta0,
                    Eigen::MatrixXd beta,
                    const Eigen::VectorXd& x_centers,
                    const Eigen::VectorXd& x_scales,
                    const bool intercept,
                    const bool standardize)
{
  // TODO: Don't pass by value.
  const int p = beta.rows();
  const int m = beta.cols();

  if (standardize) {
    for (int k = 0; k < m; ++k) {
      double x_bar_beta_sum = 0.0;
      for (int j = 0; j < p; ++j) {
        beta(j, k) /= x_scales(j);
        x_bar_beta_sum += x_centers(j) * beta(j, k);
      }

      if (intercept) {
        beta0(0) -= x_bar_beta_sum;
      }
    }
  }

  return { beta0, beta };
}

} // namespace slope
