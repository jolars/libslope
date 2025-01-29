#include "../src/slope/slope.h"
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

TEST_CASE("JIT standardization and modify-X standardization",
          "[standardization]")
{
  using Catch::Matchers::WithinAbs;

  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
  Eigen::VectorXd beta(p);

  // clang-format off
  x <<  0.288,   0,      0.880,
        0.788,   0.576,  0,
        0,       0.390,  -0.621,
        -2.210,  0,      0,
        0,       0.944,   0.821,
        0.594,   0,       0.782,
        0,      -1.990,   0,
        -0.0561, 0,      -1.470,
        0,       0.418,   1.360,
        -0.103,  0.388,   0;
  // clang-format on

  Eigen::SparseMatrix<double> x_sparse = x.sparseView();

  beta << 1, 0, -1.8;
  Eigen::VectorXd y = x * beta;
  Eigen::VectorXd w = Eigen::VectorXd::Ones(n); // weights

  Eigen::VectorXd residual = y;
  residual(0) += 1;
  residual(1) -= 0.2;
  residual(2) += 0.9;
  residual(4) -= 1.1;
  residual(8) += 0.1;
  residual(9) -= 0.3;

  slope::Slope model;

  model.setTol(1e-4);
  model.setObjective("gaussian");
  model.setStandardize(true);
  model.setModifyX(true);
  model.setIntercept(false);

  Eigen::MatrixXd x_copy = x;

  SECTION("Gradient computations for JIT standardization")
  {
    auto [x_centers, x_scales] = slope::computeCentersAndScales(x, true);
    slope::standardizeFeatures(x, x_centers, x_scales);

    auto gradient =
      slope::computeGradient(x, residual, x_centers, x_scales, w, false);
    auto gradient_jit =
      slope::computeGradient(x_copy, residual, x_centers, x_scales, w, true);
    auto gradient_sparse_jit =
      slope::computeGradient(x_sparse, residual, x_centers, x_scales, w, true);

    REQUIRE_THAT(gradient_jit, VectorApproxEqual(gradient, 1e-6));
    REQUIRE_THAT(gradient_sparse_jit, VectorApproxEqual(gradient, 1e-6));
  }

  model.fit(x_copy, y);

  Eigen::VectorXd coefs_ref = model.getCoefs().col(0);

  SECTION("JIT standardization for dense X")
  {
    model.setModifyX(false);

    model.fit(x, y);

    Eigen::VectorXd coefs_mod = model.getCoefs().col(0);

    REQUIRE_THAT(coefs_mod, VectorApproxEqual(coefs_ref, 1e-6));
  }

  SECTION("JIT standardization for sparse X")
  {
    // Never modify sparse X
    model.setModifyX(false);

    model.fit(x_sparse, y);

    Eigen::VectorXd coefs_sparse = model.getCoefs().col(0);

    REQUIRE_THAT(coefs_sparse, VectorApproxEqual(coefs_ref, 1e-6));
  }
}
