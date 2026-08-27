#include "load_libsvm.hpp"

#include <Eigen/SparseCore>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

SparseData
loadLibsvm(const std::string& filename,
           const int n_features,
           const bool binary_response)
{
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open LIBSVM data: " + filename);
  }

  std::vector<double> responses;
  std::vector<Eigen::Triplet<double>> triplets;
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream row_stream(line);
    double response;

    if (!(row_stream >> response)) {
      throw std::runtime_error("Invalid response in LIBSVM data");
    }

    responses.push_back(binary_response ? response > 0.0 : response);

    std::string field;
    while (row_stream >> field) {
      const std::size_t separator = field.find(':');
      if (separator == std::string::npos) {
        throw std::runtime_error("Invalid feature in LIBSVM data");
      }

      const int column = std::stoi(field.substr(0, separator)) - 1;
      const double value = std::stod(field.substr(separator + 1));

      if (column < 0 || column >= n_features) {
        throw std::runtime_error("LIBSVM feature index is out of range");
      }

      triplets.emplace_back(responses.size() - 1, column, value);
    }
  }

  Eigen::SparseMatrix<double> x(responses.size(), n_features);
  x.setFromTriplets(triplets.begin(), triplets.end());
  x.makeCompressed();

  Eigen::VectorXd y(responses.size());
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    y(i) = responses[i];
  }

  return { std::move(x), std::move(y) };
}
