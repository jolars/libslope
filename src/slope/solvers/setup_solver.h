#include "solver.h"
#include <memory>
#include <string>

namespace slope {

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
