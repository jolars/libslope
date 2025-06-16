#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/timer.h>
#include <slope/utils.h>
#include <thread>

TEST_CASE("Timer", "[utils]")
{
  using namespace slope;

  Timer timer;

  timer.start();

  double t0 = timer.elapsed();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  timer.pause();

  double t1 = timer.elapsed();
  double t1b = timer.elapsed();

  REQUIRE(t1 == t1b);

  timer.resume();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  double t2 = timer.elapsed();

  REQUIRE(t1 > t0);
  REQUIRE(t2 > t1);
}

TEST_CASE("Subsetting", "[utils]")
{
  auto data = generateData(20, 3, "quadratic", 1, 0.25);

  Eigen::SparseMatrix<double> x_sparse = data.x.sparseView();

  std::vector<int> indices = { 0, 4, 5, 7, 9, 19 };

  auto x_subset = slope::subset(data.x, indices);
  auto x_subset_sparse = slope::subset(x_sparse, indices);
  Eigen::MatrixXd x_subset_sparse_densified = x_subset_sparse;

  REQUIRE(x_subset.rows() == static_cast<int>(indices.size()));
  REQUIRE(x_subset_sparse.rows() == x_subset.rows());
  REQUIRE(x_subset_sparse.size() == x_subset.size());
  REQUIRE_THAT(x_subset.reshaped(),
               VectorApproxEqual(x_subset_sparse_densified.reshaped()));
}

TEST_CASE("move_elements", "[utils]")
{
  using namespace slope;

  SECTION("Move elements from higher to lower index (from > to)")
  {
    std::vector<int> v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<int> expected = { 0, 1, 5, 6, 7, 2, 3, 4, 8, 9 };

    // Move elements [5,6,7] to position 2
    move_elements(v, 5, 2, 3);

    REQUIRE(v == expected);
  }

  SECTION("Move elements from lower to higher index (from < to)")
  {
    std::vector<int> v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<int> expected = { 0, 1, 5, 6, 7, 8, 2, 3, 4, 9 };

    // Move elements [2,3,4] to position 6
    move_elements(v, 2, 6, 3);

    REQUIRE(v == expected);
  }

  SECTION("Move elements to adjacent position (higher to lower)")
  {
    std::vector<int> v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<int> expected = { 0, 1, 5, 2, 3, 4, 6, 7, 8, 9 };

    // Move element [5] to position 2
    move_elements(v, 5, 2, 1);

    REQUIRE(v == expected);
  }

  SECTION("Move elements to adjacent position (lower to higher)")
  {
    std::vector<int> v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<int> expected = { 0, 1, 3, 4, 5, 2, 6, 7, 8, 9 };

    // Move element [2] to position 5
    move_elements(v, 2, 5, 1);

    REQUIRE(v == expected);
  }

  SECTION("Move multiple elements with size > 1")
  {
    std::vector<int> v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<int> expected = { 0, 3, 4, 5, 6, 1, 2, 7, 8, 9 };

    // Move elements [1,2] to position 5
    move_elements(v, 1, 5, 2);

    REQUIRE(v == expected);
  }

  SECTION("Move elements to end of vector")
  {
    std::vector<int> v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<int> expected = { 0, 1, 2, 3, 4, 5, 6, 9, 7, 8 };

    // Move elements [1,2] to position 5
    move_elements(v, 7, 8, 2);

    REQUIRE(v == expected);
  }
}

TEST_CASE("Permutations", "[utils]")
{
  using namespace slope;

  SECTION("permute function")
  {
    std::vector<int> values = { 10, 20, 30, 40, 50 };
    std::vector<int> indices = { 3, 0, 4, 1, 2 };
    std::vector<int> expected = { 40, 10, 50, 20, 30 };

    permute(values, indices);
    REQUIRE(values == expected);

    // Second permutation to test chaining
    std::vector<int> indices2 = { 1, 3, 4, 0, 2 };
    std::vector<int> expected2 = {
      10, 20, 30, 40, 50
    }; // Should return to original

    permute(values, indices2);
    REQUIRE(values == expected2);
  }

  SECTION("inversePermute function")
  {
    std::vector<int> values = { 10, 20, 30, 40, 50 };
    std::vector<int> indices = { 3, 0, 4, 1, 2 };
    std::vector<int> expected = { 20, 40, 50, 10, 30 };

    inversePermute(values, indices);
    REQUIRE(values == expected);
  }

  SECTION("permute and inversePermute are inverse operations")
  {
    std::vector<int> original = { 10, 20, 30, 40, 50 };
    std::vector<int> values = original;
    std::vector<int> indices = { 3, 0, 4, 1, 2 };

    // Apply permutation
    permute(values, indices);

    // Apply inverse permutation
    inversePermute(values, indices);

    // Should get back the original values
    REQUIRE(values == original);
  }

  SECTION("permute with different container types")
  {
    using Catch::Matchers::WithinAbs;
    // Test with std::vector
    std::vector<double> vec = { 1.1, 2.2, 3.3, 4.4 };
    std::vector<int> indices = { 2, 0, 3, 1 };
    permute(vec, indices);
    REQUIRE_THAT(vec[0], WithinAbs(3.3, 1e-8));
    REQUIRE_THAT(vec[1], WithinAbs(1.1, 1e-8));
    REQUIRE_THAT(vec[2], WithinAbs(4.4, 1e-8));
    REQUIRE_THAT(vec[3], WithinAbs(2.2, 1e-8));

    // Test with Eigen::VectorXd
    Eigen::VectorXd eigen_vec(4);
    eigen_vec << 1.1, 2.2, 3.3, 4.4;
    permute(eigen_vec, indices);
    REQUIRE_THAT(eigen_vec[0], WithinAbs(3.3, 1e-8));
    REQUIRE_THAT(eigen_vec[1], WithinAbs(1.1, 1e-8));
    REQUIRE_THAT(eigen_vec[2], WithinAbs(4.4, 1e-8));
    REQUIRE_THAT(eigen_vec[3], WithinAbs(2.2, 1e-8));
  }
}

TEST_CASE("Sorting", "[utils]")
{
  using namespace slope;
  using Catch::Matchers::WithinAbs;

  SECTION("Sorting integers in ascending order")
  {
    std::vector<int> values = { 30, 10, 50, 20, 40 };
    std::vector<int> expected_indices = {
      1, 3, 0, 4, 2
    }; // Indices of sorted values: 10,20,30,40,50

    auto indices = sortIndex(values);

    REQUIRE(indices == expected_indices);

    // Verify the indices point to the correct sorted order
    std::vector<int> sorted_values;
    for (auto idx : indices) {
      sorted_values.push_back(values[idx]);
    }

    std::vector<int> expected_sorted = { 10, 20, 30, 40, 50 };
    REQUIRE(sorted_values == expected_sorted);
  }

  SECTION("Sorting integers in descending order")
  {
    std::vector<int> values = { 30, 10, 50, 20, 40 };
    std::vector<int> expected_indices = {
      2, 4, 0, 3, 1
    }; // Indices of sorted values: 50,40,30,20,10

    auto indices = sortIndex(values, true); // descending=true

    REQUIRE(indices == expected_indices);

    // Verify the indices point to the correct sorted order
    std::vector<int> sorted_values;
    for (auto idx : indices) {
      sorted_values.push_back(values[idx]);
    }

    std::vector<int> expected_sorted = { 50, 40, 30, 20, 10 };
    REQUIRE(sorted_values == expected_sorted);
  }

  SECTION("Sorting floating point numbers")
  {
    std::vector<double> values = { 3.14, 1.41, 2.71, 1.62 };
    std::vector<int> expected_indices = {
      1, 3, 2, 0
    }; // Indices of sorted values: 1.41,1.62,2.71,3.14

    auto indices = sortIndex(values);

    REQUIRE(indices == expected_indices);
  }

  SECTION("Sorting with duplicate values")
  {
    std::vector<int> values = { 5, 2, 8, 2, 1, 5 };

    auto indices = sortIndex(values);

    // Expected indices for sorted values: 1,2,3,4,5,8
    // Note: The specific indices for duplicate values might vary by
    // implementation So we'll check the resulting sorted order instead

    std::vector<int> sorted_values;
    for (auto idx : indices) {
      sorted_values.push_back(values[idx]);
    }

    std::vector<int> expected_sorted = { 1, 2, 2, 5, 5, 8 };
    REQUIRE(sorted_values == expected_sorted);
  }

  SECTION("Sorting with Eigen::VectorXd")
  {
    Eigen::VectorXd eigen_vec(5);
    eigen_vec << 3.5, 1.2, 4.8, 2.3, 0.9;

    auto indices = sortIndex(eigen_vec);

    std::vector<double> sorted_values;
    for (auto idx : indices) {
      sorted_values.push_back(eigen_vec[idx]);
    }

    // Check if values are in ascending order
    for (size_t i = 1; i < sorted_values.size(); ++i) {
      REQUIRE(sorted_values[i - 1] <= sorted_values[i]);
    }

    // Check first and last values specifically
    REQUIRE_THAT(sorted_values.front(), WithinAbs(0.9, 1e-8));
    REQUIRE_THAT(sorted_values.back(), WithinAbs(4.8, 1e-8));
  }
}
