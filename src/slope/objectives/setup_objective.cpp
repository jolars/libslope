#include "setup_objective.h"
#include "binomial.h"
#include "gaussian.h"
#include "multinomial.h"
#include "poisson.h"

namespace slope {

/**
 * @brief Factory function to create the appropriate objective function based on
 * the distribution family.
 *
 * @details This function creates and returns an objective function object based
 * on the specified statistical distribution family. The supported families are:
 * - "binomial": For binary classification problems (logistic regression)
 * - "poisson": For count data modeling (Poisson regression)
 * - "multinomial": For multi-class classification problems
 * - "gaussian": For continuous response variables (linear regression, default
 *   if unspecified)
 *
 * @param family A string specifying the distribution family ("binomial",
 * "poisson", "multinomial", or "gaussian")
 * @return std::unique_ptr<Objective> A unique pointer to the appropriate
 * objective function object
 */
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
