#pragma once

#include <Eigen/Core>

namespace slope {

std::vector<int>
kktCheck(const Eigen::MatrixXd& gradient,
         const Eigen::MatrixXd& beta,
         const Eigen::ArrayXd& lambda,
         const std::vector<int>& strong_set);

}
