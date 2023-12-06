#pragma once

#include "parameters.h"
#include <Eigen/Sparse>

namespace slope {

/**
 * Standardizes the given matrix column-wise.
 *
 * This function uses Welford's algorithm to compute the means and standard
 * deviation.
 *
 * @tparam T The type of the input matrix.
 * @param x The input matrix.
 * @param standardize Flag indicating whether to standardize the matrix.
 * @return A tuple containing the means and standard deviations of the columns.
 */
template<typename T>
std::tuple<Eigen::VectorXd, Eigen::VectorXd>
standardize(const T& x, const bool standardize)
{
  const int n = x.rows();
  const int p = x.cols();

  Eigen::VectorXd x_means(p);
  Eigen::VectorXd x_stddevs(p);

  for (int j = 0; j < p; ++j) {
    double mean = 0.0;
    double m2 = 0.0;
    int count = 0;

    for (typename T::InnerIterator it(x, j); it; ++it) {
      double delta = it.value() - mean;
      mean += delta / (++count);
      m2 += delta * (it.value() - mean);
    }

    // Account for zeros in the column
    double delta = -mean;
    while (count < n) {
      count++;
      mean += delta / count;
      m2 -= mean * delta;
    }

    x_means(j) = mean;
    x_stddevs(j) = std::sqrt(m2 / n);
  }
  return { x_means, x_stddevs };
}

std::tuple<double, Eigen::VectorXd>
rescaleCoefficients(double beta0,
                    Eigen::VectorXd beta,
                    const Eigen::VectorXd& x_centers,
                    const Eigen::VectorXd& x_scales,
                    const SlopeParameters& params);
} // namespace slope
