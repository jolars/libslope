/**
 * @file
 * @brief Thread management for parallel computations
 */

#pragma once

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace slope {

/**
 * @brief Manages OpenMP thread settings across libslope
 *
 * This class provides a centralized way to control thread settings for parallel
 * computations. It handles both OpenMP and non-OpenMP builds, defaulting to
 * sequential execution when OpenMP is not available.
 *
 * Thread count defaults to half of the available threads (typically the number
 * of physical CPU cores) to avoid performance degradation from hyperthreading
 * when using Eigen.
 *
 * Usage:
 * @code
 * Threads::set(4);  // Set to use 4 threads
 * int n = Threads::get();  // Get current thread count
 * @endcode
 */
class Threads
{
public:
  /**
   * @brief Set the number of threads to use for parallel computations
   *
   * @param n Number of threads. Must be positive.
   */
  static void set(const int n)
  {
    if (n > 0) {
      num_threads = n;
#ifdef _OPENMP
      omp_set_num_threads(n);
#endif
    } else {
      throw std::invalid_argument("Number of threads must be positive");
    }
  }

  /**
   * @brief Get the current number of threads
   *
   * @return Current thread count
   */
  static int get() { return num_threads; }

private:
#ifdef _OPENMP
  /// Number of threads to use. Defaults to half of max threads (physical cores)
  inline static int num_threads = std::max(1, omp_get_max_threads() / 2);
#else
  /// Default to single thread when OpenMP is not available
  inline static int num_threads = 1;
#endif
};

} // namespace slope
