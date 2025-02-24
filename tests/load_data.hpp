#pragma once

#include <Eigen/Dense>
#include <string>

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
load_dataset(const std::string& filename);
