#pragma once

#include "clusters.h"
#include <Eigen/Core>
#include <tuple>

namespace slope {

std::tuple<double, int>
slopeThreshold(const double x,
               const int j,
               const Eigen::ArrayXd lambdas,
               const Clusters& clusters);

}
