#pragma once

#include <Eigen/Core>
#include <numeric>
#include <vector>

namespace slope {

template<typename T>
int
sign(T val)
{
  return (T(0) < val) - (val < T(0));
}

template<typename T>
Eigen::ArrayXd
cumSum(const T& x)
{
  std::vector<double> cum_sum(x.size());
  std::partial_sum(
    x.data(), x.data() + x.size(), cum_sum.begin(), std::plus<double>());

  Eigen::Map<Eigen::ArrayXd> out(cum_sum.data(), cum_sum.size());

  return out;
}

} // namespace slope
