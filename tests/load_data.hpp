#pragma once

#include <Eigen/Dense>
#include <string>

std::string
findProjectRoot();

std::string
getProjectRelpath(const std::string& relative_path);

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
loadData(const std::string& filename);
