/**
 * @file
 * @brief Karush-Kuhn-Tucker (KKT) optimality condition checking for SLOPE
 * regression
 */

#pragma once

#include "jit_normalization.h"
#include "math.h"
#include <Eigen/Core>

namespace slope {

/**
 * @typedef ArrayXb
 * @brief Dynamic-size column vector of boolean values
 * Wrapper around Eigen::Array<bool, Eigen::Dynamic, 1>
 */
using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;

/**
 * @typedef ArrayXXb
 * @brief Dynamic-size matrix of boolean values
 * Wrapper around Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>
 */
using ArrayXXb = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Checks KKT conditions for SLOPE optimization
 *
 * @param gradient The gradient of the loss function
 * @param beta The current coefficients
 * @param lambda Vector of regularization parameters
 * @param strong_set Vector of indices in the strong set
 * @return std::vector<int> Indices where KKT conditions are violated
 *
 * Verifies if the current solution satisfies the KKT optimality conditions
 * for the SLOPE optimization problem. Returns indices where violations occur.
 */
std::vector<int>
kktCheck(const Eigen::VectorXd& gradient,
         const Eigen::VectorXd& beta,
         const Eigen::ArrayXd& lambda,
         const std::vector<int>& strong_set);

/**
 * Check for KKT violations and update the working set accordingly.
 *
 * @param gradient The current gradient
 * @param beta The current coefficients
 * @param lambda_curr Current lambda values
 * @param working_set The current working set
 * @param strong_set The strong set for screening
 * @param full_set The full set of indices
 * @param x The design matrix
 * @param residual The current residuals
 * @param x_centers The centers for normalization
 * @param x_scales The scales for normalization
 * @param jit_normalization Type of JIT normalization
 * @return True if no KKT violations found, false otherwise
 */
template<typename T>
bool
checkKktViolations(Eigen::VectorXd& gradient,
                   const Eigen::VectorXd& beta,
                   const Eigen::ArrayXd& lambda_curr,
                   std::vector<int>& working_set,
                   const std::vector<int>& strong_set,
                   const std::vector<int>& full_set,
                   const T& x,
                   const Eigen::MatrixXd& residual,
                   const Eigen::VectorXd& x_centers,
                   const Eigen::VectorXd& x_scales,
                   JitNormalization jit_normalization)
{
  // Check for violations in the strong set first
  updateGradient(gradient,
                 x,
                 residual,
                 strong_set,
                 x_centers,
                 x_scales,
                 Eigen::VectorXd::Ones(x.rows()),
                 jit_normalization);

  auto violations =
    setDiff(kktCheck(gradient, beta, lambda_curr, strong_set), working_set);

  if (violations.empty()) {
    // Now check for violations in the full set
    updateGradient(gradient,
                   x,
                   residual,
                   full_set,
                   x_centers,
                   x_scales,
                   Eigen::VectorXd::Ones(x.rows()),
                   jit_normalization);

    violations =
      setDiff(kktCheck(gradient, beta, lambda_curr, full_set), working_set);

    if (violations.empty()) {
      return true; // No violations found
    } else {
      working_set = setUnion(working_set, violations);
      return false; // Violations found and working set updated
    }
  } else {
    working_set = setUnion(working_set, violations);
    return false; // Violations found and working set updated
  }
}

} // namespace slope
