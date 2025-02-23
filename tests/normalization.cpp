#include "slope/math.h"
#include "slope/normalize.h"
#include "slope/slope.h"
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

  int n = 3;
  int p = 3;

  Eigen::SparseMatrix<double> x(n, p);

  x.coeffRef(0, 0) = 1;
  x.coeffRef(1, 0) = 98.2;
  x.coeffRef(2, 0) = -1007;
  x.coeffRef(0, 2) = 1000;
  x.coeffRef(1, 2) = 34;

  Eigen::MatrixXd x_dense = x;

  Eigen::VectorXd x_centers_sparse(p);
  Eigen::VectorXd x_scales_sparse(p);
  Eigen::VectorXd x_centers_dense(p);
  Eigen::VectorXd x_scales_dense(p);

  auto [x_centers_ref, x_scales_ref] = computeMeanAndStdDev(x);

  slope::computeCenters(x_centers_sparse, x, "mean");
  slope::computeScales(x_scales_sparse, x, "sd");
  slope::computeCenters(x_centers_dense, x_dense, "mean");
  slope::computeScales(x_scales_dense, x_dense, "sd");

  REQUIRE_THAT(x_centers_sparse, VectorApproxEqual(x_centers_ref));
  REQUIRE_THAT(x_scales_sparse, VectorApproxEqual(x_scales_ref));

  REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_ref));
  REQUIRE_THAT(x_scales_dense, VectorApproxEqual(x_scales_ref));
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

  Eigen::VectorXd residual = -y;
  residual(0) += 1;
  residual(1) += -0.2;
  residual(2) += 0.9;

  auto [x_centers_dense, x_scales_dense] = computeMeanAndStdDev(x_dense);
  auto [x_centers_sparse, x_scales_sparse] = computeMeanAndStdDev(x_sparse);

  REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_sparse));
  REQUIRE_THAT(x_scales_dense, VectorApproxEqual(x_scales_sparse));

  Eigen::VectorXd residual_dense =
    x_dense * beta.cwiseQuotient(x_scales_dense) - y;
  residual_dense.array() -=
    x_centers_dense.cwiseQuotient(x_scales_dense).dot(beta);

  Eigen::VectorXd residual_sparse =
    x_sparse * beta.cwiseQuotient(x_scales_sparse) - y;
  residual_sparse.array() -=
    x_centers_sparse.cwiseQuotient(x_scales_sparse).dot(beta);

  REQUIRE_THAT(residual_dense, VectorApproxEqual(residual_sparse));

  std::vector<int> active_set = { 0, 1, 2 };

  Eigen::MatrixXd gradient_dense(3, 1);
  Eigen::MatrixXd gradient_sparse = gradient_dense;

  slope::JitNormalization jit_normalization = slope::JitNormalization::Both;

  slope::updateGradient(gradient_dense,
                        x_dense,
                        residual_dense,
                        active_set,
                        w,
                        x_centers_dense,
                        x_scales_dense,
                        jit_normalization);
  slope::updateGradient(gradient_sparse,
                        x_sparse,
                        residual_sparse,
                        active_set,
                        w,
                        x_centers_sparse,
                        x_scales_sparse,
                        jit_normalization);

  REQUIRE_THAT(gradient_dense.reshaped(),
               VectorApproxEqual(gradient_sparse.reshaped()));
}

TEST_CASE("JIT standardization and modify-X standardization",
          "[standardization]")
{
  using Catch::Matchers::WithinAbs;

  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
  Eigen::VectorXd beta(p);

  std::vector<int> active_set = { 0, 1, 2 };

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

  Eigen::VectorXd residual = -y;
  residual(0) += 1;
  residual(1) -= 0.2;
  residual(2) += 0.9;
  residual(4) -= 1.1;
  residual(8) += 0.1;
  residual(9) -= 0.3;

  slope::Slope model;

  model.setTol(1e-4);
  model.setObjective("gaussian");
  model.setNormalization("standardization");
  model.setModifyX(true);
  model.setIntercept(false);

  Eigen::MatrixXd x_copy = x;

  SECTION("Gradient computations for JIT standardization")
  {
    Eigen::VectorXd x_centers(p);
    Eigen::VectorXd x_scales(p);
    slope::normalize(x, x_centers, x_scales, "mean", "sd", true);

    Eigen::MatrixXd gradient(3, 1);
    Eigen::MatrixXd gradient_jit = gradient;
    Eigen::MatrixXd gradient_sparse_jit = gradient;

    slope::updateGradient(gradient,
                          x,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::None);
    slope::updateGradient(gradient_jit,
                          x_copy,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Both);
    slope::updateGradient(gradient_sparse_jit,
                          x_sparse,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Both);

    REQUIRE_THAT(gradient_jit.reshaped(),
                 VectorApproxEqual(gradient.reshaped(), 1e-6));
    REQUIRE_THAT(gradient_sparse_jit.reshaped(),
                 VectorApproxEqual(gradient.reshaped(), 1e-6));
  }

  auto fit = model.path(x_copy, y);

  Eigen::VectorXd coefs_ref = fit.getCoefs().back();

  SECTION("JIT standardization for dense X")
  {
    model.setModifyX(false);

    fit = model.path(x, y);

    Eigen::VectorXd coefs_mod = fit.getCoefs().back();

    REQUIRE_THAT(coefs_mod, VectorApproxEqual(coefs_ref, 1e-6));
  }

  SECTION("JIT standardization for sparse X")
  {
    // Never modify sparse X
    model.setModifyX(false);

    fit = model.path(x_sparse, y);

    Eigen::VectorXd coefs_sparse = fit.getCoefs().back();

    REQUIRE_THAT(coefs_sparse, VectorApproxEqual(coefs_ref, 1e-6));
  }

  SECTION("Assertions")
  {
    Eigen::VectorXd centers = Eigen::VectorXd::Ones(p);
    Eigen::VectorXd scales = Eigen::VectorXd::Ones(p);
    Eigen::VectorXd centers_wrong = Eigen::VectorXd::Ones(p - 1);
    Eigen::VectorXd scales_wrong = Eigen::VectorXd::Ones(p + 5);

    REQUIRE_THROWS_AS(model.setScaling("minabs"), std::invalid_argument);
    REQUIRE_THROWS_AS(model.setCentering("quantile"), std::invalid_argument);

    model.setCentering(centers);
    model.setScaling(scales);

    REQUIRE_NOTHROW(model.fit(x_sparse, y));

    model.setCentering(centers_wrong);

    REQUIRE_THROWS_AS(model.fit(x_sparse, y), std::invalid_argument);

    model.setCentering(centers);
    model.setScaling(scales_wrong);

    REQUIRE_THROWS_AS(model.path(x_sparse, y), std::invalid_argument);

    Eigen::VectorXd centers_nan = centers;
    Eigen::VectorXd scales_nan = scales;

    scales_nan(0) = std::log(-1);

    model.setScaling(scales_nan);

    REQUIRE_THROWS_AS(model.path(x_sparse, y), std::invalid_argument);

    centers_nan(0) = std::log(-1);

    model.setScaling(centers);
    model.setCentering(centers_nan);

    REQUIRE_THROWS_AS(model.path(x_sparse, y), std::invalid_argument);
  }

  SECTION("Loop over normalization types")
  {
    slope::Slope model;

    model.setObjective("gaussian");
    model.setDiagnostics(true);

    Eigen::MatrixXd x(3, 2);
    Eigen::Vector2d beta;
    Eigen::VectorXd y(3);

    // clang-format off
    x << 1.1, 2.3,
         0.2, 1.5,
         0.5, 0.2;
    // clang-format on
    beta << 1, 2;

    y = x * beta;

    std::vector<std::string> centering_types = { "none", "mean", "min" };
    std::vector<std::string> scaling_types = { "none", "sd",      "l1",
                                               "l2",   "max_abs", "range" };

    model.setModifyX(false);

    for (const auto& centering_type : centering_types) {
      for (const auto& scaling_type : scaling_types) {
        model.setCentering(centering_type);
        model.setScaling(scaling_type);

        REQUIRE_NOTHROW(model.fit(x, y));
      }
    }
  }
}
