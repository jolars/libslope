#include "solver.h"
#include <memory>
#include <string>

namespace slope {

/**
 * @brief Factory function to create and configure a SLOPE solver
 *
 * @details Creates a solver object based on the specified type and parameters.
 * The solver implements the Sorted L1 Penalized Estimation (SLOPE) algorithm
 * with various configurations possible.
 *
 * @param solver_type Type of solver to use (e.g., "pgd", "admm")
 * @param objective Type of objective function ("gaussian", "binomial",
 * "poisson", "multinomial")
 * @param tol Convergence tolerance for the solver
 * @param max_it_inner Maximum number of inner iterations
 * @param standardize_jit Whether to standardize features just-in-time
 * @param print_level Level of verbosity for solver output (0: none, 1: some, 2:
 * verbose)
 * @param intercept Whether to fit an intercept term
 * @param update_clusters Whether to update cluster assignments during
 * optimization
 * @param pgd_freq Frequency of proximal gradient descent updates
 *
 * @return std::unique_ptr<solvers::SolverBase> A unique pointer to the
 * configured solver
 */
std::unique_ptr<solvers::SolverBase>
setupSolver(const std::string& solver_type,
            const std::string& objective,
            double tol,
            int max_it_inner,
            bool standardize_jit,
            int print_level,
            bool intercept,
            bool update_clusters,
            int pgd_freq);

} // namespace slope
