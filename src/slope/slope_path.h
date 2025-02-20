#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace slope {

class SlopePath
{
private:
  std::vector<Eigen::VectorXd> intercepts;
  std::vector<Eigen::SparseMatrix<double>> coefs;
  Eigen::ArrayXd alpha;
  Eigen::ArrayXd lambda;
  std::vector<double> deviance;
  double null_deviance;
  std::vector<std::vector<double>> primals;
  std::vector<std::vector<double>> duals;
  std::vector<std::vector<double>> time;
  std::vector<int> passes;

public:
  SlopePath() = default;

  SlopePath(const std::vector<Eigen::VectorXd>& intercepts,
            const std::vector<Eigen::SparseMatrix<double>>& coefs,
            const Eigen::ArrayXd& alpha,
            const Eigen::ArrayXd& lambda,
            const std::vector<double>& deviance,
            double null_deviance,
            const std::vector<std::vector<double>>& primals,
            const std::vector<std::vector<double>>& duals,
            const std::vector<std::vector<double>>& time,
            const std::vector<int>& passes)
    : intercepts{ intercepts }
    , coefs{ coefs }
    , alpha{ alpha }
    , lambda{ lambda }
    , deviance{ deviance }
    , null_deviance{ null_deviance }
    , primals{ primals }
    , duals{ duals }
    , time{ time }
    , passes{ passes }
  {
  }

  const std::vector<Eigen::VectorXd>& getIntercepts() const
  {
    return intercepts;
  }

  const std::vector<Eigen::SparseMatrix<double>>& getCoefs() const
  {
    return coefs;
  }

  const Eigen::SparseMatrix<double>& getCoefs(const std::size_t i) const
  {
    assert(i >= 0 && i < coefs.size() && "Index out of bounds");
    return coefs[i];
  }

  const Eigen::ArrayXd& getAlpha() const { return alpha; }
  const Eigen::ArrayXd& getLambda() const { return lambda; }
  const std::vector<double>& getDeviance() const { return deviance; }
  double getNullDeviance() const { return null_deviance; }
  const std::vector<std::vector<double>>& getPrimals() const { return primals; }
  const std::vector<std::vector<double>>& getDuals() const { return duals; }
  const std::vector<std::vector<double>>& getTime() const { return time; }
  const std::vector<int>& getPasses() const { return passes; }

  const std::vector<double> getDevianceRatios() const
  {
    std::vector<double> ratios(deviance.size());

    for (size_t i = 0; i < deviance.size(); i++) {
      ratios[i] = 1.0 - deviance[i] / null_deviance;
    }

    return ratios;
  }

  const std::vector<std::vector<double>> getGaps() const
  {
    std::vector<std::vector<double>> gaps(primals.size());

    for (size_t i = 0; i < primals.size(); i++) {
      gaps[i].resize(primals[i].size());

      for (size_t j = 0; j < primals[i].size(); j++) {
        gaps[i][j] = primals[i][j] - duals[i][j];
      }
    }

    return gaps;
  }
};

} // namespace slope
