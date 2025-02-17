#pragma once

#include <Eigen/Core>
#include <vector>

namespace slope {

std::vector<int>
previouslyActiveSet(const Eigen::MatrixXd& beta);

std::vector<int>
strongSet(const Eigen::MatrixXd& gradient_prev,
          const Eigen::ArrayXd& lambda,
          const Eigen::ArrayXd& lambda_prev);

}
