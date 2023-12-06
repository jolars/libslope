#pragma once

#include "math.h"
#include "utils.h"
#include <Eigen/Core>

namespace slope {

class SortedL1Norm
{
private:
  double alpha = 1.0;
  Eigen::ArrayXd lambda;

public:
  SortedL1Norm(const Eigen::ArrayXd& lambda);

  double eval(const Eigen::VectorXd& beta) const;

  // template<typename T>
  // double dual(const Eigen::VectorXd& gradient) const
  // {
  //   sort(xt_theta_abs, true);
  //   return (cumSum(xt_theta_abs) / cumSum((lambda *
  //   alpha).eval())).maxCoeff();
  // }

  Eigen::VectorXd prox(const Eigen::VectorXd& beta, const double scale) const;

  double dualNorm(const Eigen::VectorXd& a) const;

  void setLambda(const Eigen::ArrayXd& new_lambda);

  void setAlpha(const double new_alpha);

  Eigen::ArrayXd getLambda() const;

  const Eigen::ArrayXd& getLambdaRef() const;

  double getAlpha() const;
};

}
