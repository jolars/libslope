#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/standardize.h>

std::tuple<Eigen::VectorXd, Eigen::VectorXd>
computeMeanAndStdDev(const Eigen::MatrixXd& x)
{
  const int n = x.rows();
  const int p = x.cols();

  Eigen::VectorXd x_means = x.colwise().mean();
  Eigen::VectorXd x_stddevs(p);

  for (int j = 0; j < p; ++j) {
    x_stddevs(j) =
      std::sqrt((x.col(j).array() - x_means(j)).square().sum() / n);
  }

  return { x_means, x_stddevs };
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd>
computeMeanAndStdDev(const Eigen::SparseMatrix<double>& x)
{
  const int n = x.rows();
  const int p = x.cols();

  Eigen::VectorXd x_means(p);
  Eigen::VectorXd x_stddevs(p);

  for (int j = 0; j < p; ++j) {
    x_means(j) = x.col(j).sum() / n;
    // TODO: Reconsider this implementation since it might overflow.
    x_stddevs(j) =
      std::sqrt(x.col(j).squaredNorm() / n - std::pow(x_means(j), 2));
  }

  return { x_means, x_stddevs };
}

TEST_CASE("Check that standardization algorithm works",
          "[utils][standardization]")
{
  using Catch::Matchers::WithinAbs;

  Eigen::SparseMatrix<double> x(3, 3);

  x.coeffRef(0, 0) = 1;
  x.coeffRef(1, 0) = 98.2;
  x.coeffRef(2, 0) = -1007;
  x.coeffRef(0, 2) = 1000;
  x.coeffRef(1, 2) = 34;

  Eigen::MatrixXd x_dense = x;

  auto [x_centers_ref, x_scales_ref] = computeMeanAndStdDev(x);

  auto [x_centers, x_scales] = slope::standardize(x, true);
  auto [x_centers_dense, x_scales_dense] = slope::standardize(x_dense, true);

  REQUIRE_THAT(x_centers, VectorApproxEqual(x_centers_ref));
  REQUIRE_THAT(x_scales, VectorApproxEqual(x_scales_ref));

  REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_ref));
  REQUIRE_THAT(x_scales, VectorApproxEqual(x_scales_ref));
}
