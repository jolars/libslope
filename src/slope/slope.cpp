#include "slope.h"
#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <set>

namespace slope {

void
Slope::setIntercept(bool intercept)
{
  this->intercept = intercept;
}
void
Slope::setStandardize(bool standardize)
{
  this->standardize = standardize;
}
void
Slope::setUpdateClusters(bool update_clusters)
{
  this->update_clusters = update_clusters;
}
void
Slope::setAlphaMinRatio(double alpha_min_ratio)
{
  this->alpha_min_ratio = alpha_min_ratio;
}
void
Slope::setLearningRateDecr(double learning_rate_decr)
{
  this->learning_rate_decr = learning_rate_decr;
}
void
Slope::setQ(double q)
{
  this->q = q;
}
void
Slope::setTol(double tol)
{
  this->tol = tol;
}
void
Slope::setMaxIt(int max_it)
{
  this->max_it = max_it;
}
void
Slope::setMaxItOuter(int maxItOuter)
{
  this->max_it_outer = maxItOuter;
}
void
Slope::setPathLength(int path_length)
{
  this->path_length = path_length;
}
void
Slope::setPgdFreq(int pgd_freq)
{
  this->pgd_freq = pgd_freq;
}
void
Slope::setPrintLevel(int print_level)
{
  this->print_level = print_level;
}
void
Slope::setLambdaType(const std::string& lambda_type)
{
  validateOption(lambda_type, { "bh" }, "lambda_type");

  this->lambda_type = lambda_type;
}
void
Slope::setObjective(const std::string& objective)
{
  static const std::set<std::string> valid_objectives = { "gaussian",
                                                          "binomial",
                                                          "poisson" };

  if (valid_objectives.find(objective) == valid_objectives.end()) {
    throw std::invalid_argument(
      "Invalid objective function: " + objective +
      ". Must be one of: 'gaussian', 'binomial', or 'poisson'");
  }
  this->objective = objective;
}
void
Slope::setModifyX(const bool modify_x)
{
  this->modify_x = modify_x;
}

const Eigen::ArrayXd&
Slope::getAlpha() const
{
  return alpha_out;
}
const Eigen::ArrayXd&
Slope::getLambda() const
{
  return lambda_out;
}

const Eigen::SparseMatrix<double>&
Slope::getCoefs() const
{
  return betas;
}

const Eigen::VectorXd&
Slope::getIntercepts() const
{
  return beta0s;
}

int
Slope::getTotalIterations() const
{
  return it_total;
}

const std::vector<std::vector<double>>&
Slope::getDualGaps() const
{
  return dual_gaps_path;
}

const std::vector<std::vector<double>>&
Slope::getPrimals() const
{
  return primals_path;
}

} // namespace slope
