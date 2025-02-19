#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace slope {

class SlopeFit
{
private:
  Eigen::VectorXd intercepts;
  Eigen::SparseMatrix<double> coefs;
  double alpha;
  Eigen::ArrayXd lambda;
  double deviance;
  double null_deviance;
  std::vector<double> primals;
  std::vector<double> duals;

public:
  SlopeFit() = default;

  SlopeFit(const Eigen::VectorXd& intercepts,
           const Eigen::SparseMatrix<double>& coefs,
           const double alpha,
           const Eigen::ArrayXd& lambda,
           double deviance,
           double null_deviance,
           const std::vector<double>& primals,
           const std::vector<double>& duals)
    : intercepts{ intercepts }
    , coefs{ coefs }
    , alpha{ alpha }
    , lambda{ lambda }
    , deviance{ deviance }
    , null_deviance{ null_deviance }
    , primals{ primals }
    , duals{ duals }
  {
  }

  const Eigen::VectorXd& getIntercepts() const { return intercepts; }
  const Eigen::SparseMatrix<double>& getCoefs() const { return coefs; }
  const Eigen::ArrayXd& getLambda() const { return lambda; }
  double getAlpha() const { return alpha; }
  double getDeviance() const { return deviance; }
  double getNullDeviance() const { return null_deviance; }
  const std::vector<double>& getPrimals() const { return primals; }
  const std::vector<double>& getDuals() const { return duals; }

  double getDevianceRatios() const { return 1.0 - deviance / null_deviance; }

  std::vector<double> getGaps() const
  {
    std::vector<double> gaps(primals.size());

    for (size_t i = 0; i < primals.size(); i++) {
      gaps[i] = primals[i] - duals[i];
    }

    return gaps;
  }
};

} // namespace slope
