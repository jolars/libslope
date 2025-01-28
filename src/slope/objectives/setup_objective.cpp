#include "setup_objective.h"
#include "binomial.h"
#include "gaussian.h"

namespace slope {

std::unique_ptr<Objective>
setupObjective(const std::string family)
{
  if (family == "binomial")
    return std::make_unique<Binomial>();

  // else Gaussian
  return std::make_unique<Gaussian>();
}

}
