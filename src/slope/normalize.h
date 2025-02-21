/** @file
 * @brief Functions to normalize the design matrix and rescale coefficients
 * in case the design was normalized
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
 * @param x_centers A vector where the computed means will be stored.
 * @param x_scales A vector where the computed standard deviations will be
 * stored.
 * @return A tuple containing the means and standard deviations of the columns.
 */
template<typename T>
void
standardize(const T& x, Eigen::VectorXd& x_centers, Eigen::VectorXd& x_scales)
{
  const int n = x.rows();
  const int p = x.cols();

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

    x_centers(j) = mean;
    x_scales(j) = std::sqrt(m2 / n);
  }
}

/**
 * Compute centers and scales for the columns of the input matrix.
 *
 * There are three supported normalization types:
 * - "none": Do not compute centers and scales.
 * - "manual": Use the provided centers and scales. If one is missing, a default
 *   of zeros for centers or ones for scales will be used.
 * - "standardization": Compute centers and scales using Welford’s algorithm.
 *
 * @tparam T The type of the input matrix.
 * @param x The input matrix.
 * @param x_centers A vector where the computed or provided centers will be
 * stored.
 * @param x_scales A vector where the computed or provided scales will be
 * stored.
 * @param type A string specifying the normalization type ("none", "manual", or
 * "standardization").
 *
 * @throws std::invalid_argument if the provided manual centers or scales have
 * invalid dimensions or contain non-finite values.
 */
template<typename T>
void
computeCentersAndScales(const T& x,
                        Eigen::VectorXd& x_centers,
                        Eigen::VectorXd& x_scales,
                        const std::string& type)
{
  int p = x.cols();

  if (type != "none" && type != "manual") {
    x_centers.resize(p);
    x_scales.resize(p);
  }

  if (type == "none") {
  } else if (type == "manual") {
    // Manual centers and scales provided, just check that they are valid

    // TODO: Right now we force both centers and scales to be present, but we
    // should allow for only specifying one or the other, which is also what
    // some types of normalization (max-abs) are defined as.
    if (x_centers.size() == 0) {
      x_centers = Eigen::VectorXd::Zero(p);
    }

    if (x_scales.size() == 0) {
      x_centers = Eigen::VectorXd::Ones(p);
    }

    if (x_centers.size() != p) {
      throw std::invalid_argument("Invalid dimensions in centers");
    }

    if (x_scales.size() != p) {
      throw std::invalid_argument("Invalid dimensions in scales");
    }

    if (!x_centers.allFinite()) {
      throw std::invalid_argument("Centers must be finite");
    }

    if (!x_scales.allFinite()) {
      throw std::invalid_argument("Scales must be finite");
    }

  } else if (type == "standardization") {
    standardize(x, x_centers, x_scales);
  } else {
    throw std::invalid_argument("Invalid normalization type");
  }
}

/**
 * Normalize a dense matrix by centering and scaling.
 *
 * The function computes column centers and scaling factors based on the
 * specified normalization type ("none", "manual", or "standardization"). If
 * modify_x is true, the normalization is applied directly to the input matrix.
 *
 * @param x         The dense input matrix to be normalized.
 * @param x_centers A vector that will hold the column centers. It will be
 *                  resized to match the number of columns.
 * @param x_scales  A vector that will hold the column scaling factors. It will
 *                  be resized to match the number of columns.
 * @param type      A string specifying the normalization type ("none",
 *                  "manual", or "standardization").
 * @param modify_x  If true, modifies x in-place; otherwise, x remains unchanged
 *                  (centers/scales are still computed).
 *
 * @return true if normalization succeeds, false otherwise.
 */
bool
normalize(Eigen::MatrixXd& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& type,
          const bool modify_x);

/**
 * Normalize a sparse matrix by scaling only.
 *
 * To preserve sparsity, centering is not applied. The scaling factors for each
 * column are computed according to the specified normalization type ("none",
 * "manual", or "standardization"). If modify_x is true, the scaling is applied
 * directly to the input matrix.
 *
 * @param x The sparse input matrix to be normalized.
 * @param x_centers A vector that will hold the column centers.
 * For sparse matrices, centering is typically skipped; this
 * parameter is maintained for consistency.
 * @param x_scales  A vector that will hold the column scaling factors. It will
 * be resized to match the number of columns.
 * @param type A string specifying the normalization type ("none",
 * "manual", or "standardization").
 * @param modify_x If true, performs in-place scaling on x; otherwise, leaves x
 * unchanged.
 *
 * @return true if normalization succeeds, false otherwise.
 */
bool
normalize(Eigen::SparseMatrix<double>& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& type,
          const bool modify_x);

/**
 * @brief Rescales the coefficients using the given parameters.
 *
 * This function rescales the coefficients by dividing each coefficient by the
 * corresponding scale factor and subtracting the product of the center and
 * the coefficient from the intercept.
 *
 * @param beta0 The intercept coefficient.
 * @param beta The vector of coefficients.
 * @param x_centers The vector of center values.
 * @param x_scales The vector of scale factors.
 * @param intercept Should an intercept be fit?
 * @param normalization_type type of normalization
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
                    const std::string& normalization_type);

} // namespace slope
