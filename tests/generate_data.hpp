#pragma once

#include <Eigen/Dense>
#include <string>

struct SimulatedData
{
  SimulatedData(const int n, const int p, const int m);
  Eigen::MatrixXd x;
  Eigen::VectorXd y;
  Eigen::MatrixXd beta;
};

SimulatedData
generateData(int n = 200,
             int p = 20,
             const std::string& type = "quadratic",
             int m = 1,
             double x_sparsity = 0.3,
             double coef_sparsity = 0.2,
             unsigned seed = 1234);
