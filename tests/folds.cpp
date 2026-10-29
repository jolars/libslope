#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <numeric>
#include <slope/folds.h>
#include <vector>

TEST_CASE("Fold assignments are platform-independent", "[cv][folds]")
{
  struct TestCase
  {
    uint64_t seed;
    std::vector<std::vector<std::vector<int>>> expected;
  };

  const std::vector<TestCase> cases = {
    { 0,
      { { { 7, 2, 0, 8 }, { 3, 9, 6 }, { 1, 5, 4 } },
        { { 1, 7, 3, 9 }, { 4, 0, 5 }, { 2, 6, 8 } } } },
    { 42,
      { { { 1, 7, 9, 0 }, { 3, 8, 4 }, { 2, 5, 6 } },
        { { 2, 4, 3, 8 }, { 1, 7, 5 }, { 9, 0, 6 } } } },
    { (uint64_t{ 1 } << 32) + 42,
      { { { 3, 4, 7, 0 }, { 5, 1, 2 }, { 8, 6, 9 } },
        { { 4, 5, 8, 6 }, { 3, 2, 0 }, { 9, 7, 1 } } } },
    { std::numeric_limits<uint64_t>::max(),
      { { { 1, 3, 6, 9 }, { 8, 4, 2 }, { 7, 5, 0 } },
        { { 7, 2, 0, 8 }, { 3, 9, 6 }, { 1, 5, 4 } } } }
  };

  for (const auto& test_case : cases) {
    DYNAMIC_SECTION("seed = " << test_case.seed)
    {
      const slope::Folds folds(10, 3, 2, test_case.seed);
      REQUIRE(folds.folds == test_case.expected);
    }
  }
}

TEST_CASE("Random folds partition samples evenly", "[cv][folds]")
{
  for (const int n_samples : { 0, 1, 2, 10, 101 }) {
    for (const int n_folds : { 1, 2, 3, 10 }) {
      DYNAMIC_SECTION(n_samples << " samples, " << n_folds << " folds")
      {
        const slope::Folds folds(n_samples, n_folds, 2);
        REQUIRE(folds.numFolds() == n_folds);
        REQUIRE(folds.numRepetitions() == 2);
        REQUIRE(folds.numEvals() == 2 * n_folds);

        std::vector<int> expected(n_samples);
        std::iota(expected.begin(), expected.end(), 0);

        for (int rep = 0; rep < 2; ++rep) {
          std::vector<int> indices;
          for (int fold = 0; fold < n_folds; ++fold) {
            const auto& test = folds.getTestIndices(fold, rep);
            const auto train = folds.getTrainingIndices(fold, rep);
            const int fold_size =
              n_samples / n_folds + (fold < n_samples % n_folds);
            REQUIRE(test.size() == fold_size);
            REQUIRE(train.size() == n_samples - fold_size);

            auto combined = train;
            combined.insert(combined.end(), test.begin(), test.end());
            std::sort(combined.begin(), combined.end());
            REQUIRE(combined == expected);

            indices.insert(indices.end(), test.begin(), test.end());
          }
          std::sort(indices.begin(), indices.end());
          REQUIRE(indices == expected);
        }
      }
    }
  }
}
