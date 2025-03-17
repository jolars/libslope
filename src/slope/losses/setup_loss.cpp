#include "setup_loss.h"
#include "logistic.h"
#include "multinomial.h"
#include "poisson.h"
#include "quadratic.h"

namespace slope {

std::unique_ptr<losses::Loss>
setupLoss(const std::string& loss)
{
  if (loss == "logistic")
    return std::make_unique<losses::Logistic>();
  else if (loss == "poisson")
    return std::make_unique<losses::Poisson>();
  else if (loss == "multinomial")
    return std::make_unique<losses::Multinomial>();

  // else Quadratic
  return std::make_unique<losses::Quadratic>();
}

}
