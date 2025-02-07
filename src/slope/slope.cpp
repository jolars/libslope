#include "slope.h"
#include "solvers/pgd.h"
#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <set>

namespace slope {

// Explicit instantiations for common matrix/solver combinations
template void
Slope::fit<solvers::Hybrid, Eigen::MatrixXd>(Eigen::MatrixXd&,
                                             const Eigen::MatrixXd&,
                                             Eigen::ArrayXd,
                                             Eigen::ArrayXd);

template void
Slope::fit<solvers::Hybrid, Eigen::SparseMatrix<double>>(
  Eigen::SparseMatrix<double>&,
  const Eigen::MatrixXd&,
  Eigen::ArrayXd,
  Eigen::ArrayXd);

template void
Slope::fit<solvers::PGD, Eigen::MatrixXd>(Eigen::MatrixXd&,
                                          const Eigen::MatrixXd&,
                                          Eigen::ArrayXd,
                                          Eigen::ArrayXd);

template void
Slope::fit<solvers::PGD, Eigen::SparseMatrix<double>>(
  Eigen::SparseMatrix<double>&,
  const Eigen::MatrixXd&,
  Eigen::ArrayXd,
  Eigen::ArrayXd);

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
  if (alpha_min_ratio <= 0 || alpha_min_ratio >= 1) {
    throw std::invalid_argument("alpha_min_ratio must be in (0, 1)");
  }
  this->alpha_min_ratio = alpha_min_ratio;
}

void
Slope::setLearningRateDecr(double learning_rate_decr)
{
  if (learning_rate_decr <= 0 || learning_rate_decr >= 1) {
    throw std::invalid_argument("learning_rate_decr must be in (0, 1)");
  }
  this->learning_rate_decr = learning_rate_decr;
}

void
Slope::setQ(double q)
{
  if (q < 0 || q > 1) {
    throw std::invalid_argument("q must be between 0 and 1");
  }
  this->q = q;
}

void
Slope::setTol(double tol)
{
  if (tol < 0) {
    throw std::invalid_argument("tol must be non-negative");
  }
  this->tol = tol;
}

void
Slope::setMaxIt(int max_it)
{
  if (max_it < 1) {
    throw std::invalid_argument("max_it_outer must be >= 1");
  }
  this->max_it = max_it;
}

void
Slope::setMaxItInner(int max_it_inner)
{
  if (max_it_inner < 1) {
    throw std::invalid_argument("max_it_inner must be >= 1");
  }
  this->max_it_inner = max_it_inner;
}

void
Slope::setPathLength(int path_length)
{
  if (path_length < 1) {
    throw std::invalid_argument("path_length must be >= 1");
  }
  this->path_length = path_length;
}

void
Slope::setPgdFreq(int pgd_freq)
{
  if (pgd_freq < 1) {
    throw std::invalid_argument("pgd_freq must be > 1");
  }
  this->pgd_freq = pgd_freq;
}

void
Slope::setPrintLevel(int print_level)
{
  if (print_level < 0) {
    throw std::invalid_argument("print_level must be >= 0");
  }
  this->print_level = print_level;
}

void
Slope::setLambdaType(const std::string& lambda_type)
{
  validateOption(
    lambda_type, { "bh", "gaussian", "oscar", "lasso" }, "lambda_type");

  this->lambda_type = lambda_type;
}

void
Slope::setObjective(const std::string& objective)
{
  validateOption(objective, { "gaussian", "binomial", "poisson" }, "objective");
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

const std::vector<Eigen::SparseMatrix<double>>
Slope::getCoefs() const
{
  return betas;
}

const std::vector<Eigen::VectorXd>
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

void
Slope::reset()
{
  this->dual_gaps_path.clear();
  this->primals_path.clear();
  this->betas.clear();
  this->beta0s.clear();
  this->it_total = 0;
}

// slope.cpp

} // namespace slope
