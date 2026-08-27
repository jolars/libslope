#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <string>

struct SparseData
{
  Eigen::SparseMatrix<double> x;
  Eigen::VectorXd y;
};

SparseData
loadLibsvm(const std::string& filename, int n_features, bool binary_response);
