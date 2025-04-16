/** @file
 * @brief Functions to normalize the design matrix and rescale coefficients
 * in case the design was normalized
 */

#pragma once

#include "jit_normalization.h"
#include "math.h"
#include <Eigen/SparseCore>

namespace slope {

/**
 * Compute centers.
 *
 * There are two supported centering types:
 * - "none": Do not compute centers and scales.
 * - "mean": Use arithmetic means.
 *
 * @tparam T The type of the input matrix.
 * @param x_centers A vector where the computed or provided centers will be
 * stored.
 * @param x The input matrix.
 * @param type A string specifying the normalization type ("none", "manual", or
 * "standardization").
 *
 * @throws std::invalid_argument if the provided manual centers or scales have
 * invalid dimensions or contain non-finite values.
 */
template<typename T>
void
computeCenters(Eigen::VectorXd& x_centers, const T& x, const std::string& type)
{
  int p = x.cols();

  if (type == "manual") {
    if (x_centers.size() != p) {
      throw std::invalid_argument("Invalid dimensions in centers");
    }

    if (!x_centers.allFinite()) {
      throw std::invalid_argument("Centers must be finite");
    }

  } else if (type == "mean") {
    x_centers = means(x);
  } else if (type == "min") {
    x_centers = mins(x);
  } else if (type != "none") {
    throw std::invalid_argument("Invalid centering type");
  }
}

/**
 * Compute scales
 *
 * There are two supported scaling types:
 * - "none": Do not compute centers and scales.
 * - "sd": Compute centers and scales using Welford’s algorithm.
 *
 * @tparam T The type of the input matrix.
 * @param x The input matrix.
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
computeScales(Eigen::VectorXd& x_scales, const T& x, const std::string& type)
{
  int p = x.cols();

  if (type == "manual") {
    if (x_scales.size() != p) {
      throw std::invalid_argument("Invalid dimensions in scales");
    }
    if (!x_scales.allFinite()) {
      throw std::invalid_argument("Scales must be finite");
    }
  } else if (type == "sd") {
    x_scales = stdDevs(x);
  } else if (type == "l1") {
    x_scales = l1Norms(x);
  } else if (type == "l2") {
    x_scales = l2Norms(x);
  } else if (type == "max_abs") {
    x_scales = maxAbs(x);
  } else if (type == "range") {
    x_scales = ranges(x);
  } else if (type != "none") {
    throw std::invalid_argument("Invalid scaling type");
  }
}

/**
 * Normalize a mapped dense matrix by centering and scaling.
 *
 * Since Eigen::Map cannot be modified in-place (it's a view of external
 * memory), this overload ignores the modify_x parameter and always returns the
 * appropriate JitNormalization enum for deferred normalization.
 *
 * @param x The mapped dense input matrix.
 * @param x_centers A vector that will hold the column centers.
 * @param x_scales A vector that will hold the column scaling factors.
 * @param centering_type A string specifying the centering type.
 * @param scaling_type A string specifying the scaling type.
 * @param modify_x Ignored for mapped matrices (normalization is always
 * deferred).
 *
 * @return JitNormalization enum indicating what normalization should be
 * applied.
 */
template<typename Derived>
JitNormalization
normalize(Eigen::Map<Derived>& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& centering_type,
          const std::string& scaling_type,
          const bool)
{
  computeCenters(x_centers, x, centering_type);
  computeScales(x_scales, x, scaling_type);

  bool center = centering_type != "none";
  bool scale = scaling_type != "none";

  if (center && scale) {
    return JitNormalization::Both;
  } else if (center) {
    return JitNormalization::Center;
  } else if (scale) {
    return JitNormalization::Scale;
  } else {
    return JitNormalization::None;
  }
}

/**
 * Normalize a dense matrix by centering and scaling.
 *
 * The function computes column centers and scaling factors based on the
 * specified normalization type ("none", "manual", or "standardization"). If
 * modify_x is true, the normalization is applied directly to the input matrix.
 *
 * @param x The dense input matrix to be normalized.
 * @param x_centers A vector that will hold the column centers. It will be
 * resized to match the number of columns.
 * @param x_scales  A vector that will hold the column scaling factors. It will
 * be resized to match the number of columns.
 * @param centering_type A string specifying the normalization type ("none",
 * "manual", or "standardization").
 * @param scaling_type A string specifying the normalization type ("none",
 * "manual", or "standardization").
 * @param modify_x If true, modifies x in-place; otherwise, x remains unchanged
 * (centers/scales are still computed).
 *
 * @return true if normalization succeeds, false otherwise.
 */
JitNormalization
normalize(Eigen::MatrixXd& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& centering_type,
          const std::string& scaling_type,
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
 * @param centering_type A string specifying the normalization type ("none",
 * "manual", or "standardization").
 * @param scaling_type A string specifying the normalization type ("none",
 * "manual", or "standardization").
 * @param modify_x If true, performs in-place scaling on x; otherwise, leaves x
 * unchanged.
 *
 * @return true if normalization succeeds, false otherwise.
 */
JitNormalization
normalize(Eigen::SparseMatrix<double>& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& centering_type,
          const std::string& scaling_type,
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
 * @return A tuple containing the rescaled intercept and coefficients.
 *
 * @note The input vectors `beta`, `x_centers`, and `x_scales` must have the
 * same size.
 * @note The output vector `beta` will be modified in-place.
 *
 * @see SlopeParameters
 */
std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
rescaleCoefficients(const Eigen::VectorXd& beta0,
                    const Eigen::SparseMatrix<double>& beta,
                    const Eigen::VectorXd& x_centers,
                    const Eigen::VectorXd& x_scales,
                    const bool intercept);

} // namespace slope
