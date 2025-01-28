#include "../src/slope/standardize.h"
#include "../src/slope/math.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
  Eigen::MatrixXd x_dense = x;

  return computeMeanAndStdDev(x_dense);
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

  auto [x_centers, x_scales] = slope::computeCentersAndScales(x, true);
  auto [x_centers_dense, x_scales_dense] =
    slope::computeCentersAndScales(x_dense, true);

  REQUIRE_THAT(x_centers, VectorApproxEqual(x_centers_ref));
  REQUIRE_THAT(x_scales, VectorApproxEqual(x_scales_ref));

  REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_ref));
  REQUIRE_THAT(x_scales, VectorApproxEqual(x_scales_ref));
}

TEST_CASE("Check that in-place standardization works",
          "[utils][standardization]")
{
  using Catch::Matchers::WithinAbs;

  Eigen::MatrixXd x_dense(3, 3);
  x_dense << 1, 0, 1000, 98.2, 0, 34, -1007, 0, 0;
  Eigen::SparseMatrix<double> x_sparse = x_dense.sparseView();
  Eigen::VectorXd beta(3);
  beta << 1, 0, -1.8;
  Eigen::VectorXd y = x_dense * beta;
  Eigen::VectorXd w = Eigen::VectorXd::Ones(3); // weights

  Eigen::VectorXd residual = y;
  residual(0) += 1;
  residual(1) += -0.2;
  residual(2) += 0.9;

  auto [x_centers_dense, x_scales_dense] = computeMeanAndStdDev(x_dense);
  auto [x_centers_sparse, x_scales_sparse] = computeMeanAndStdDev(x_sparse);

  REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_sparse));
  REQUIRE_THAT(x_scales_dense, VectorApproxEqual(x_scales_sparse));

  Eigen::VectorXd residual_dense =
    y - x_dense * beta.cwiseQuotient(x_scales_dense);
  residual_dense.array() +=
    x_centers_dense.cwiseQuotient(x_scales_dense).dot(beta);

  Eigen::VectorXd residual_sparse =
    y - x_sparse * beta.cwiseQuotient(x_scales_sparse);
  residual_sparse.array() +=
    x_centers_sparse.cwiseQuotient(x_scales_sparse).dot(beta);

  REQUIRE_THAT(residual_dense, VectorApproxEqual(residual_sparse));

  auto gradient_dense = slope::computeGradient(
    x_dense, residual_dense, w, x_centers_dense, x_scales_dense, true);
  auto gradient_sparse = slope::computeGradient(
    x_sparse, residual_sparse, w, x_centers_sparse, x_scales_sparse, true);

  REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));
}
