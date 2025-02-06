/**
 * @file
 * @brief Numerical solver class for SLOPE (Sorted L-One Penalized Estimation)
 */

#pragma once

#include <Eigen/Core>

namespace slope {

/**
 * @brief Base class template for SLOPE numerical solvers
 *
 * @tparam Derived CRTP derived class implementing the specific solver algorithm
 *
 * This class implements the base functionality for Sorted L-One Penalized
 * Estimation (SLOPE) solvers using the Curiously Recurring Template Pattern
 * (CRTP). SLOPE is a method that generalizes the Lasso by penalizing the sorted
 * magnitudes of the coefficients with decreasing weights.
 */
template<typename Derived>
class Solver
{
public:
  /**
   * @brief Construct a new Solver
   *
   * @param tol Convergence tolerance
   * @param max_it Maximum number of iterations
   * @param standardize_jit Whether to standardize features just-in-time
   * @param intercept Whether to fit an intercept term
   * @param update_clusters Whether to update coefficient clusters during
   * optimization
   * @param pgd_freq Frequency of proximal gradient descent updates
   */
  Solver(double tol,
         int max_it,
         bool standardize_jit,
         int print_level,
         bool intercept,
         bool update_clusters,
         int pgd_freq)
    : tol(tol)
    , max_it(max_it)
    , standardize_jit(standardize_jit)
    , print_level(print_level)
    , intercept(intercept)
    , update_clusters(update_clusters)
    , pgd_freq(pgd_freq)
  {
  }

  /**
   * @brief Run the solver
   *
   * @tparam Args Variadic template parameters for solver-specific arguments
   * @param args Solver-specific arguments forwarded to the implementation
   */
  template<typename... Args>
  void run(Args&&... args)
  {
    return static_cast<Derived*>(this)->runImpl(args...);
  }

protected:
  double tol;           ///< Convergence tolerance
  int max_it;           ///< Maximum number of iterations
  bool standardize_jit; ///< Whether to standardize features just-in-time
  int print_level;      ///< Verbosity level for solver output
  bool intercept;       ///< Whether to fit an intercept term
  bool update_clusters; ///< Whether to update coefficient clusters during
                        ///< optimization
  int pgd_freq;         ///< Frequency of proximal gradient descent updates
};

} // namespace slope
