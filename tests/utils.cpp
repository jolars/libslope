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

  std::this_thread::sleep_for(std::chrono::microseconds(10));

  timer.pause();

  double t1 = timer.elapsed();
  double t1b = timer.elapsed();

  REQUIRE(t1 == t1b);

  timer.resume();

  std::this_thread::sleep_for(std::chrono::microseconds(10));

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
