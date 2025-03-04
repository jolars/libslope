
#pragma once

#include "slope/utils.h"
#include <Eigen/src/Core/Matrix.h>
#include <cstdint>
#include <vector>

namespace slope {

class Folds
{
public:
  // Constructor for generating folds
  Folds(int n_samples, int n_folds, uint64_t seed = 42)
    : folds(createFolds(n_samples, n_folds, seed))
    , n_folds(n_folds)
  {
  }

  // Constructor for user-provided folds
  explicit Folds(std::vector<std::vector<int>> folds)
    : folds(std::move(folds))
    , n_folds(folds.size())
  {
  }

  // Get test indices for a specific fold
  const std::vector<int>& getTestIndices(size_t fold_idx) const
  {
    return folds[fold_idx];
  }

  // Get training indices (all except the specified fold)
  std::vector<int> getTrainingIndices(size_t fold_idx) const
  {
    std::vector<int> train_indices;
    for (size_t i = 0; i < folds.size(); ++i) {
      if (i != fold_idx) {
        train_indices.insert(
          train_indices.end(), folds[i].begin(), folds[i].end());
      }
    }
    return train_indices;
  }

  template<typename MatrixType>
  std::tuple<MatrixType, Eigen::MatrixXd, MatrixType, Eigen::MatrixXd>
  split(MatrixType& x, const Eigen::MatrixXd& y, size_t fold_idx) const
  {
    auto test_idx = getTestIndices(fold_idx);
    auto train_idx = getTrainingIndices(fold_idx);

    MatrixType x_test = subset(x, test_idx);
    Eigen::MatrixXd y_test = y(test_idx, Eigen::all);

    MatrixType x_train = subset(x, train_idx);
    Eigen::MatrixXd y_train = y(train_idx, Eigen::all);

    return { x_train, y_train, x_test, y_test };
  }

  size_t numFolds() const { return n_folds; }

private:
  std::vector<std::vector<int>> folds;
  std::size_t n_folds;

  static std::vector<std::vector<int>> createFolds(int n_samples,
                                                   int n_folds,
                                                   uint64_t seed);
};

} // namespace slope
