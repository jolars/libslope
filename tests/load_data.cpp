#include "load_data.hpp"
#include <fstream>
#include <sstream>
#include <vector>

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
load_dataset(const std::string& filename)
{
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::string line;
  // Skip header
  std::getline(file, line);

  std::vector<double> y_values;
  std::vector<std::vector<double>> x_values;

  // Read data
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;

    // First value is y
    std::getline(ss, value, ',');
    y_values.push_back(std::stod(value));

    // Remaining values are x
    std::vector<double> row;
    while (std::getline(ss, value, ',')) {
      row.push_back(std::stod(value));
    }
    x_values.push_back(row);
  }

  // Convert to Eigen matrices
  Eigen::VectorXd y(y_values.size());
  Eigen::MatrixXd x(x_values.size(), x_values[0].size());

  for (size_t i = 0; i < y_values.size(); ++i) {
    y(i) = y_values[i];
    for (size_t j = 0; j < x_values[i].size(); ++j) {
      x(i, j) = x_values[i][j];
    }
  }

  return { x, y };
}
