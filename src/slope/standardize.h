/** @file
 * @brief Functions to standardize the design matrix and rescale coefficients
 */

#pragma once

#include <Eigen/Sparse>

namespace slope {

/**
 * Compute means and standard deviations of the columns of the input matrix.
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
computeCentersAndScales(const T& x, const bool standardize)
{
  // TODO: Standadize flag has no effect; remove it.
  // TODO: Make this function more general and allow other statistics.
  const int n = x.rows();
  const int p = x.cols();

  Eigen::VectorXd x_means(p);
  Eigen::VectorXd x_stddevs(p);

  for (int j = 0; j < p; ++j) {
    double mean = 0.0;
    double m2 = 0.0;
    int count = 0;

    for (typename T::InnerIterator it(x, j); it; ++it) {
      count++;
      double delta = it.value() - mean;
      mean += delta / count;
      double delta2 = it.value() - mean;
      m2 += delta * delta2;
    }

    // Account for zeros in the column
    while (count < n) {
      count++;
      double delta = -mean;
      mean += delta / count;
      double delta2 = -mean;
      m2 += delta * delta2;
    }

    x_means(j) = mean;
    x_stddevs(j) = std::sqrt(m2 / n);
  }
  return { x_means, x_stddevs };
}

/**
 * Standardize a dense matrix by centering and scaling.
 *
 * @param x The dense input matrix.
 * @param x_centers The means of the columns.
 * @param x_scales The standard deviations of the columns.
 */
template<typename T>
void
standardizeFeatures(Eigen::DenseBase<T>& x,
                    const Eigen::VectorXd& x_centers,
                    const Eigen::VectorXd& x_scales)
{
  // TODO: Switch name to `normalize`.
  const int n = x.rows();
  const int p = x.cols();

  for (int j = 0; j < p; ++j) {
    x.col(j) = (x.col(j).array() - x_centers(j)) / x_scales(j);
  }
}

/**
 * Scale a sparse matrix (without centering to preserve sparsity).
 *
 * @param x The sparse input matrix.
 * @param x_scales The standard deviations of the columns.
 */
template<typename T>
void
standardizeFeatures(Eigen::SparseMatrixBase<T>& x,
                    const Eigen::VectorXd& x_centers,
                    const Eigen::VectorXd& x_scales)
{
  // TODO: Switch name to `normalize`.
  const int p = x.cols();

  for (int j = 0; j < p; ++j) {
    for (typename T::InnerIterator it(x.derived(), j); it; ++it) {
      it.valueRef() = it.value() / x_scales(j);
    }
  }
}

/**
 * @brief Rescales the coefficients using the given parameters.
 *
 * This function rescales the coefficients by dividing each coefficient by the
 * corresponding scale factor and subtracting the product of the center and the
 * coefficient from the intercept.
 *
 * @param beta0 The intercept coefficient.
 * @param beta The vector of coefficients.
 * @param x_centers The vector of center values.
 * @param x_scales The vector of scale factors.
 * @param intercept Should an intercept be fit?
 * @param standardize Is the design matrix standardized?
 * @return A tuple containing the rescaled intercept and coefficients.
 *
 * @note The input vectors `beta`, `x_centers`, and `x_scales` must have the
 * same size.
 * @note The output vector `beta` will be modified in-place.
 *
 * @see SlopeParameters
 */
std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
rescaleCoefficients(Eigen::VectorXd beta0,
                    Eigen::MatrixXd beta,
                    const Eigen::VectorXd& x_centers,
                    const Eigen::VectorXd& x_scales,
                    const bool intercept,
                    const bool standardize);

} // namespace slope
