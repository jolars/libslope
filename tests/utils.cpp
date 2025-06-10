#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
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
}
