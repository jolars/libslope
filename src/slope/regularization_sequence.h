#include "math.h"
#include "objectives.h"
#include "qnorm.h"
#include "sorted_l1_norm.h"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <memory>
#include <string>

namespace slope {

/**
 * Generates a BH sequence of lambda values.
 *
 * This function generates a sequence of lambda values based on the given
 * parameters based on the Benjamini-Hochberg sequence for SLOPE.
 *
 * @param p The number of lambda values to generate.
 * @param q The quantile value used in the calculation.
 * @return An Eigen::ArrayXd containing the generated lambda values.
 */
Eigen::ArrayXd
lambdaSequence(const int p, const double q, const std::string& type);

template<typename T>
Eigen::ArrayXd
regularizationPath(const T& x,
                   const Eigen::VectorXd& w,
                   const Eigen::VectorXd& z,
                   const Eigen::VectorXd& x_centers,
                   const Eigen::VectorXd& x_scales,
                   const SortedL1Norm& penalty,
                   const int path_length,
                   double alpha_min_ratio,
                   const bool intercept,
                   const bool standardize)
{
  const int n = x.rows();
  const int p = x.cols();

  if (alpha_min_ratio < 0) {
    alpha_min_ratio = n > p ? 1e-4 : 1e-2;
  }

  Eigen::VectorXd gradient(p);
  Eigen::VectorXd z_w = z.cwiseProduct(w);

  if (standardize) {
    double z_w_sum = z_w.sum();
    for (int j = 0; j < p; ++j) {
      gradient[j] = x.col(j).dot(z_w) / x_scales[j] -
                    z_w_sum * (x_centers[j] / x_scales[j]);
    }
  } else {
    gradient = x.transpose() * z_w;
  }

  double alpha_max =
    (penalty.dualNorm(gradient) / cumSum(penalty.getLambdaRef())).maxCoeff() /
    static_cast<double>(n);

  Eigen::ArrayXd alpha(path_length);

  for (int i = 0; i < path_length; ++i) {
    alpha[i] = alpha_max * std::pow(alpha_min_ratio,
                                    i / (static_cast<double>(path_length - 1)));
  }

  return alpha;
}

} // namespace slope
