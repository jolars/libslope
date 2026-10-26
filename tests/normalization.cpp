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

TEST_CASE("Weighted feature normalization", "[normalization][weights]")
{
  using Catch::Matchers::WithinAbs;

  Eigen::MatrixXd x(4, 3);
  x << 2, 0, 4, 4, 3, 4, -100, 100, -100, 0, 6, 4;
  Eigen::SparseMatrix<double> sparse = x.sparseView();
  Eigen::VectorXd weights(4);
  weights << 1, 2, 0, 1;
  Eigen::MatrixXd retained(3, 3);
  retained << 2, 0, 4, 4, 3, 4, 0, 6, 4;
  Eigen::VectorXd retained_weights(3);
  retained_weights << 1, 2, 1;

  SECTION("Weighted moments and historical norm scales")
  {
    Eigen::VectorXd centers, scales;
    slope::computeCenters(centers, x, "mean", weights);
    Eigen::VectorXd expected_centers(3);
    expected_centers << 2.5, 3, 4;
    REQUIRE_THAT(centers, VectorApproxEqual(expected_centers));

    slope::computeScales(scales, x, "sd", weights);
    Eigen::VectorXd expected_scales(3);
    expected_scales << std::sqrt(2.75), std::sqrt(4.5), 0;
    REQUIRE_THAT(scales, VectorApproxEqual(expected_scales));

    slope::computeScales(scales, x, "l1", weights);
    expected_scales << 7.5, 9, 12;
    REQUIRE_THAT(scales, VectorApproxEqual(expected_scales));

    slope::computeScales(scales, x, "l2", weights);
    expected_scales << std::sqrt(27), std::sqrt(40.5), std::sqrt(48);
    REQUIRE_THAT(scales, VectorApproxEqual(expected_scales));
  }

  SECTION("Sparse parity, weight rescaling, and zero-row deletion")
  {
    for (const std::string type : { "mean", "min" }) {
      Eigen::VectorXd dense_result, sparse_result, rescaled, deleted;
      slope::computeCenters(dense_result, x, type, weights);
      slope::computeCenters(sparse_result, sparse, type, weights);
      slope::computeCenters(rescaled, x, type, 7 * weights);
      slope::computeCenters(deleted, retained, type, retained_weights);
      REQUIRE_THAT(dense_result, VectorApproxEqual(sparse_result));
      REQUIRE_THAT(dense_result, VectorApproxEqual(rescaled));
      REQUIRE_THAT(dense_result, VectorApproxEqual(deleted));
    }

    for (const std::string type : { "sd", "l1", "l2", "max_abs", "range" }) {
      Eigen::VectorXd dense_result, sparse_result, rescaled, deleted;
      slope::computeScales(dense_result, x, type, weights);
      slope::computeScales(sparse_result, sparse, type, weights);
      slope::computeScales(rescaled, x, type, 7 * weights);
      slope::computeScales(deleted, retained, type, retained_weights);
      REQUIRE_THAT(dense_result, VectorApproxEqual(sparse_result));
      REQUIRE_THAT(dense_result, VectorApproxEqual(rescaled));
      REQUIRE_THAT(dense_result, VectorApproxEqual(deleted));
    }
  }

  SECTION("Uniform weights preserve existing normalization")
  {
    for (const std::string type : { "sd", "l1", "l2", "max_abs", "range" }) {
      Eigen::VectorXd unweighted, weighted;
      slope::computeScales(unweighted, x, type);
      slope::computeScales(weighted, x, type, Eigen::VectorXd::Constant(4, 5));
      REQUIRE_THAT(weighted, VectorApproxEqual(unweighted));
    }
  }

  SECTION("Manual values and constant columns")
  {
    Eigen::VectorXd centers = Eigen::VectorXd::Constant(3, 2);
    Eigen::VectorXd scales = Eigen::VectorXd::Constant(3, 3);
    slope::computeCenters(centers, x, "manual", weights);
    slope::computeScales(scales, x, "manual", weights);
    REQUIRE_THAT(centers(0), WithinAbs(2, 0));
    REQUIRE_THAT(scales(0), WithinAbs(3, 0));

    auto normalized = x;
    slope::normalize(normalized, centers, scales, "mean", "sd", true, weights);
    REQUIRE_THAT(scales(2), WithinAbs(1, 0));
    REQUIRE_THAT(normalized(0, 2), WithinAbs(0, 0));
  }

  SECTION("In-place normalization masks ignored rows before scaling")
  {
    Eigen::MatrixXd tiny(4, 1);
    tiny << -1e-150, 0, 1e-150, 1e200;
    Eigen::VectorXd support(4);
    support << 1, 1, 1, 0;
    Eigen::VectorXd centers, scales;

    slope::normalize(tiny, centers, scales, "mean", "sd", true, support);

    REQUIRE(tiny.allFinite());
    REQUIRE_THAT(centers(0), WithinAbs(0, 0));
    REQUIRE_THAT(scales(0) / 1e-150, WithinAbs(std::sqrt(2.0 / 3.0), 1e-14));
    REQUIRE_THAT(tiny(0, 0), WithinAbs(-std::sqrt(1.5), 1e-14));
    REQUIRE_THAT(tiny(1, 0), WithinAbs(0, 0));
    REQUIRE_THAT(tiny(2, 0), WithinAbs(std::sqrt(1.5), 1e-14));
    REQUIRE_THAT(tiny(3, 0), WithinAbs(0, 0));
  }

  SECTION("Sparse extrema ignore implicit zeros outside positive support")
  {
    Eigen::MatrixXd positive(3, 1);
    positive << 2, 4, 0;
    Eigen::SparseMatrix<double> positive_sparse = positive.sparseView();
    Eigen::VectorXd support(3);
    support << 1, 1, 0;
    Eigen::VectorXd centers, scales;
    slope::computeCenters(centers, positive_sparse, "min", support);
    slope::computeScales(scales, positive_sparse, "range", support);
    REQUIRE_THAT(centers(0), WithinAbs(2, 0));
    REQUIRE_THAT(scales(0), WithinAbs(2, 0));
  }

  SECTION("Invalid observation weights")
  {
    Eigen::VectorXd result;
    REQUIRE_THROWS_AS(
      slope::computeCenters(result, x, "mean", Eigen::VectorXd::Ones(3)),
      std::invalid_argument);
    REQUIRE_THROWS_AS(
      slope::computeScales(result, x, "sd", Eigen::VectorXd::Zero(4)),
      std::invalid_argument);
  }
}

TEST_CASE("Weighted constant columns have exactly zero variance",
          "[normalization][weights]")
{
  Eigen::MatrixXd x = Eigen::MatrixXd::Constant(5, 1, 0.1);
  Eigen::VectorXd weights(5);
  weights << 1, 2, 3, 4, 5;

  SECTION("Constant on every row") {}

  SECTION("Constant only on positive-weight rows")
  {
    x.conservativeResize(7, 1);
    weights.conservativeResize(7);
    x(5, 0) = 0.0;
    x(6, 0) = 1e100;
    weights(5) = 0.0;
    weights(6) = 0.0;
  }

  Eigen::SparseMatrix<double> sparse = x.sparseView();
  Eigen::VectorXd centers, dense_scales, sparse_scales;
  slope::computeScales(dense_scales, x, "sd", weights);
  slope::computeScales(sparse_scales, sparse, "sd", weights);
  REQUIRE(dense_scales(0) == 0.0);
  REQUIRE(sparse_scales(0) == 0.0);

  slope::normalize(x, centers, dense_scales, "mean", "sd", true, weights);
  slope::normalize(
    sparse, centers, sparse_scales, "mean", "sd", false, weights);
  REQUIRE(dense_scales(0) == 1.0);
  REQUIRE(sparse_scales(0) == 1.0);
  REQUIRE(x.isZero(1e-16));
}

TEST_CASE("Weighted sparse variance retains small implicit-zero weights",
          "[normalization][weights][sparse]")
{
  using Catch::Matchers::WithinRel;

  Eigen::MatrixXd x(2, 1);
  x << 1, 0;
  Eigen::VectorXd weights(2);
  weights << 1, 1e-16;

  SECTION("Trailing implicit zero") {}

  SECTION("Leading implicit zero")
  {
    x.col(0).reverseInPlace();
    weights.reverseInPlace();
  }

  SECTION("Implicit zeros between stored entries")
  {
    x.resize(5, 1);
    x << 0, 1, 0, 1, 0;
    weights.resize(5);
    weights << 1e-16, 0.25, 2e-16, 0.75, 1e-16;
  }

  SECTION("Explicit zero and implicit zero")
  {
    x.resize(3, 1);
    x << 1, 0, 0;
    weights.resize(3);
    weights << 1, 1e-16, 1e-16;
  }

  SECTION("Long gaps with tiny weights between dominant stored rows")
  {
    x = Eigen::MatrixXd::Zero(257, 1);
    weights = Eigen::VectorXd::Constant(257, 1e-200);
    for (const int row : { 0, 64, 127, 256 }) {
      x(row, 0) = 1.0;
      weights(row) = 0.25;
    }
  }

  Eigen::SparseMatrix<double> sparse = x.sparseView();
  if (x.rows() == 3) {
    sparse.coeffRef(1, 0) = 0.0;
  }

  const double zero_weight =
    (weights.array() * (x.col(0).array() == 0.0).cast<double>()).sum();
  const double expected = std::sqrt(zero_weight) / (1.0 + zero_weight);
  Eigen::VectorXd centers, dense_scales, sparse_scales;
  slope::computeScales(dense_scales, x, "sd", weights);
  slope::computeScales(sparse_scales, sparse, "sd", weights);
  REQUIRE_THAT(dense_scales(0), WithinRel(expected, 1e-12));
  REQUIRE_THAT(sparse_scales(0), WithinRel(expected, 1e-12));

  slope::normalize(x, centers, dense_scales, "none", "sd", false, weights);
  slope::normalize(
    sparse, centers, sparse_scales, "none", "sd", false, weights);
  REQUIRE_THAT(sparse_scales(0), WithinRel(dense_scales(0), 1e-12));
}

TEST_CASE("Weighted sparse variance across different column supports",
          "[normalization][weights][sparse]")
{
  using Catch::Matchers::WithinAbs;
  using Catch::Matchers::WithinRel;

  const int n = 257;
  Eigen::MatrixXd x = Eigen::MatrixXd::Ones(n, 5);
  Eigen::VectorXd weights = Eigen::VectorXd::Constant(n, 1e-200);
  weights(n / 2) = 1.0;
  x(127, 0) = 0.0;
  x.col(1).setZero();
  x(n / 2, 1) = 1.0;
  for (int i = 1; i < n; i += 2) {
    x(i, 2) = 0.0;
  }
  x.col(4).setZero();

  Eigen::SparseMatrix<double> sparse = x.sparseView();
  sparse.coeffRef(64, 1) = 0.0;
  Eigen::VectorXd dense_scales, sparse_scales;
  slope::computeScales(dense_scales, x, "sd", weights);
  slope::computeScales(sparse_scales, sparse, "sd", weights);

  for (int j = 0; j < 3; ++j) {
    const double zero_weight =
      (weights.array() * (x.col(j).array() == 0.0).cast<double>()).sum();
    REQUIRE_THAT(sparse_scales(j), WithinRel(std::sqrt(zero_weight), 1e-12));
    REQUIRE_THAT(sparse_scales(j), WithinRel(dense_scales(j), 1e-12));
  }
  REQUIRE_THAT(sparse_scales(3), WithinAbs(0.0, 0.0));
  REQUIRE_THAT(sparse_scales(4), WithinAbs(0.0, 0.0));
}

TEST_CASE("Normalization, check solutions", "[normalization]")
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

TEST_CASE("Normalization, JIT", "[normalization]")
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

  Eigen::VectorXd gradient_dense(3);
  Eigen::VectorXd gradient_sparse = gradient_dense;

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

  REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));
}

TEST_CASE("Normalization, JIT modify-X", "[normalization]")
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
  model.setLoss("quadratic");
  model.setNormalization("standardization");
  model.setModifyX(true);
  model.setIntercept(false);

  Eigen::MatrixXd x_copy = x;

  SECTION("Gradient computations for JIT standardization")
  {
    Eigen::VectorXd x_centers(p);
    Eigen::VectorXd x_scales(p);
    slope::normalize(x, x_centers, x_scales, "mean", "sd", true);

    Eigen::VectorXd gradient(3);
    Eigen::VectorXd gradient_jit = gradient;
    Eigen::VectorXd gradient_sparse_jit = gradient;

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

    REQUIRE_THAT(gradient_jit, VectorApproxEqual(gradient, 1e-6));
    REQUIRE_THAT(gradient_sparse_jit, VectorApproxEqual(gradient, 1e-6));
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
}

TEST_CASE("Normalization, loop combinations", "[normalization]")
{
  slope::Slope model;

  model.setLoss("quadratic");
  model.setDiagnostics(true);

  int n = 7;
  int p = 2;

  Eigen::MatrixXd x(n, p);
  Eigen::VectorXd beta(p);
  Eigen::VectorXd y(n);

  // clang-format off
  x <<  1.1,  2.3,
        0.2,  0.0,
        0.1, -5.1,
        8.5, -0.9,
        0.0,  0.0,
        9.1,  0.0,
        0.5, -0.2;
  // clang-format on
  beta << 1, 2;

  y = x * beta;

  std::vector<std::string> centering_types = { "mean", "none", "min" };
  std::vector<std::string> scaling_types = { "none", "sd",      "l1",
                                             "l2",   "max_abs", "range" };

  Eigen::VectorXd x_centers_dense;
  Eigen::VectorXd x_centers_sparse;
  Eigen::VectorXd x_scales_dense;
  Eigen::VectorXd x_scales_sparse;

  for (const auto& centering_type : centering_types) {
    for (const auto& scaling_type : scaling_types) {
      model.setCentering(centering_type);
      model.setScaling(scaling_type);

      // clang-format off
      x << 1.1,  2.3,
           0.2,  0.0,
           0.0, -5.1,
           8.5, -0.9,
           0.0,  0.0,
           9.1,  0.0,
           0.5, -0.2;
      // clang-format on
      Eigen::SparseMatrix<double> x_sparse = x.sparseView();

      slope::computeCenters(x_centers_dense, x, centering_type);
      slope::computeCenters(x_centers_sparse, x_sparse, centering_type);
      REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_sparse, 1e-5));

      slope::computeScales(x_scales_dense, x, scaling_type);
      slope::computeScales(x_scales_sparse, x, scaling_type);
      REQUIRE_THAT(x_scales_dense, VectorApproxEqual(x_scales_sparse, 1e-5));

      double alpha = 1e-5;

      model.setModifyX(false);
      auto fit_mod = model.fit(x, y, alpha);
      double intercept_mod = fit_mod.getIntercepts()[0];
      Eigen::VectorXd coefs_mod = fit_mod.getCoefs();

      model.setModifyX(true);
      auto fit_nomod = model.fit(x, y, alpha);
      Eigen::VectorXd coefs_nomod = fit_nomod.getCoefs();
      double intercept_nomod = fit_mod.getIntercepts()[0];

      model.setModifyX(false);
      auto fit_sparse_mod = model.fit(x_sparse, y, alpha);
      double intercept_sparse_mod = fit_sparse_mod.getIntercepts()[0];
      Eigen::VectorXd coefs_sparse_mod = fit_sparse_mod.getCoefs();

      model.setModifyX(true);
      auto fit_sparse_nomod = model.fit(x_sparse, y, alpha);
      Eigen::VectorXd coefs_sparse_nomod = fit_sparse_nomod.getCoefs();
      double intercept_sparse_nomod = fit_sparse_nomod.getIntercepts()[0];

      REQUIRE_THAT(coefs_mod, VectorApproxEqual(coefs_nomod, 1e-4));
      REQUIRE_THAT(coefs_sparse_mod, VectorApproxEqual(coefs_nomod, 1e-4));
      REQUIRE_THAT(coefs_sparse_nomod, VectorApproxEqual(coefs_nomod, 1e-4));
      REQUIRE_THAT(intercept_mod,
                   Catch::Matchers::WithinAbs(intercept_nomod, 1e-4));
      REQUIRE_THAT(intercept_sparse_mod,
                   Catch::Matchers::WithinAbs(intercept_nomod, 1e-4));
      REQUIRE_THAT(intercept_sparse_nomod,
                   Catch::Matchers::WithinAbs(intercept_nomod, 1e-4));
    }
  }
}
