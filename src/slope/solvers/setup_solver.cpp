#include "hybrid.h"
#include "pgd.h"
#include <memory>
#include <stdexcept>
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
            int pgd_freq)
{
  std::string solver_choice = solver_type;

  if (solver_type == "auto") {
    // TODO: Make this more sophisticated, e.g. define in solver class
    // and check if compatible with objective.
    solver_choice = objective == "multinomial" ? "pgd" : "hybrid";
  }

  if (objective == "multinomial" && solver_choice != "pgd") {
    throw std::invalid_argument(
      "multinomial objective is only supported with pgd solver");
  }

  if (solver_choice == "pgd") {
    return std::make_unique<solvers::PGD>(tol,
                                          max_it_inner,
                                          standardize_jit,
                                          print_level,
                                          intercept,
                                          update_clusters,
                                          pgd_freq);
  } else if (solver_choice == "hybrid") {
    return std::make_unique<solvers::Hybrid>(tol,
                                             max_it_inner,
                                             standardize_jit,
                                             print_level,
                                             intercept,
                                             update_clusters,
                                             pgd_freq);
  } else {
    throw std::invalid_argument("solver type not recognized");
  }
}
}
