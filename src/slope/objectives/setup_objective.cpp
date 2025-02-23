#include "setup_objective.h"
#include "binomial.h"
#include "gaussian.h"
#include "multinomial.h"
#include "poisson.h"

namespace slope {

std::unique_ptr<Objective>
setupObjective(const std::string family)
{
  if (family == "binomial")
    return std::make_unique<Binomial>();
  else if (family == "poisson")
    return std::make_unique<Poisson>();
  else if (family == "multinomial")
    return std::make_unique<Multinomial>();

  // else Gaussian
  return std::make_unique<Gaussian>();
}

}
