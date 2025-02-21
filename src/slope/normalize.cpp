#include "normalize.h"

namespace slope {

bool
normalize(Eigen::MatrixXd& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& type,
          const bool modify_x)
{
  const int p = x.cols();

  computeCentersAndScales(x, x_centers, x_scales, type);

  bool normalize_jit = type != "none" && !modify_x;

  if (modify_x && type != "none") {
    for (int j = 0; j < p; ++j) {
      x.col(j) = (x.col(j).array() - x_centers(j)) / x_scales(j);
    }
  }

  return normalize_jit;
}

bool
normalize(Eigen::SparseMatrix<double>& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& type,
          const bool)
{
  // const int p = x.cols();

  computeCentersAndScales(x, x_centers, x_scales, type);

  // TODO: Actually allow normalization in place for sparse matrices
  // Maybe we need to use separate scale_jit and center_jit, which we probably
  // need for the case when the user supplies only one or the other anyway. bool
  // normalize_jit = type != "none" && !modify_x;
  bool normalize_jit = type != "none";

  // if (modify_x) {
  //   for (int j = 0; j < p; ++j) {
  //     for (Eigen::SparseMatrix<double>::InnerIterator it(x, j); it; ++it) {
  //       it.valueRef() = it.value() / x_scales(j);
  //     }
  //   }
  // }

  return normalize_jit;
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
rescaleCoefficients(Eigen::VectorXd beta0,
                    Eigen::MatrixXd beta,
                    const Eigen::VectorXd& x_centers,
                    const Eigen::VectorXd& x_scales,
                    const bool intercept,
                    const std::string& normalization_type)
{
  const int p = beta.rows();
  const int m = beta.cols();

  if (normalization_type != "none") {
    for (int k = 0; k < m; ++k) {
      double x_bar_beta_sum = 0.0;
      for (int j = 0; j < p; ++j) {
        beta(j, k) /= x_scales(j);
        x_bar_beta_sum += x_centers(j) * beta(j, k);
      }

      if (intercept) {
        beta0(k) -= x_bar_beta_sum;
      }
    }
  }

  return { beta0, beta };
}

} // namespace slope
