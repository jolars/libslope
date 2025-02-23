#include "hybrid.h"
#include "pgd.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace slope {

std::unique_ptr<solvers::SolverBase>
setupSolver(const std::string& solver_type,
            const std::string& loss,
            double tol,
            int max_it_inner,
            JitNormalization jit_normalization,
            bool intercept,
            bool update_clusters,
            int pgd_freq)
{
  std::string solver_choice = solver_type;

  if (solver_type == "auto") {
    // TODO: Make this more sophisticated, e.g. define in solver class
    // and check if compatible with the loss function.
    solver_choice = loss == "multinomial" ? "fista" : "hybrid";
  }

  if (loss == "multinomial" && solver_choice == "hybrid") {
    throw std::invalid_argument("multinomial loss is currently not "
                                "supported with the hybrid solver");
  }

  if (solver_choice == "pgd") {
    return std::make_unique<solvers::PGD>(tol,
                                          max_it_inner,
                                          jit_normalization,
                                          intercept,
                                          update_clusters,
                                          pgd_freq,
                                          "pgd");
  } else if (solver_choice == "fista") {
    return std::make_unique<solvers::PGD>(tol,
                                          max_it_inner,
                                          jit_normalization,
                                          intercept,
                                          update_clusters,
                                          pgd_freq,
                                          "fista");
  } else if (solver_choice == "hybrid") {
    return std::make_unique<solvers::Hybrid>(tol,
                                             max_it_inner,
                                             jit_normalization,
                                             intercept,
                                             update_clusters,
                                             pgd_freq);
  } else {
    throw std::invalid_argument("solver type not recognized");
  }
}
}
