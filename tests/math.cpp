#include "slope/math.h"
#include "generate_data.hpp"
#include "slope/normalize.h"
#include "test_helpers.hpp"
#include <slope/threads.h>
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Linear predictor computations", "[math][linearPredictor]")
{
  using namespace Catch::Matchers;

  // Create a simple matrix
  int n = 6;
  int p = 2;
  int m = 2;
  Eigen::MatrixXd x(n, p);
  // clang-format off
  x <<  1.0,  2.0,
        3.0,  3.0,
        0.0,  4.0,
        5.0, -6.0,
       -1.0,  0.0,
        0.0,  0.0;
  // clang-format on

  Eigen::SparseMatrix<double> x_sparse = x.sparseView();

  Eigen::VectorXd beta0(m);
  Eigen::VectorXd beta(p * m);
  beta0 << 0.5, -0.5;
  beta << 1.0, -1.0, -0.5, 0.5;

  // Centers and scales for the features
  Eigen::VectorXd x_centers(p);
  Eigen::VectorXd x_scales(p);

  x_centers << -1.0, 2.0;
  x_scales << 2.0, 2.0;

  // Create an active set with all indices (0, 1, 2, 3)
  std::vector<int> active_set = { 0, 1, 2, 3 };

  SECTION("No normalization and no intercept")
  {
    Eigen::MatrixXd eta = slope::linearPredictor(x,
                                                 active_set,
                                                 beta0,
                                                 beta,
                                                 x_centers,
                                                 x_scales,
                                                 slope::JitNormalization::None,
                                                 false);

    Eigen::MatrixXd eta_sparse =
      slope::linearPredictor(x_sparse,
                             active_set,
                             beta0,
                             beta,
                             x_centers,
                             x_scales,
                             slope::JitNormalization::None,
                             false);

    REQUIRE_THAT(eta.reshaped(), VectorApproxEqual(eta_sparse.reshaped()));
  }

  SECTION("Both center and scale and no intercept")
  {
    Eigen::MatrixXd eta = slope::linearPredictor(x,
                                                 active_set,
                                                 beta0,
                                                 beta,
                                                 x_centers,
                                                 x_scales,
                                                 slope::JitNormalization::Both,
                                                 false);

    Eigen::MatrixXd eta_sparse =
      slope::linearPredictor(x_sparse,
                             active_set,
                             beta0,
                             beta,
                             x_centers,
                             x_scales,
                             slope::JitNormalization::Both,
                             false);

    REQUIRE_THAT(eta.reshaped(), VectorApproxEqual(eta_sparse.reshaped()));
  }

  SECTION("Center only and no intercept")
  {
    Eigen::MatrixXd eta =
      slope::linearPredictor(x,
                             active_set,
                             beta0,
                             beta,
                             x_centers,
                             x_scales,
                             slope::JitNormalization::Center,
                             false);

    Eigen::MatrixXd eta_sparse =
      slope::linearPredictor(x_sparse,
                             active_set,
                             beta0,
                             beta,
                             x_centers,
                             x_scales,
                             slope::JitNormalization::Center,
                             false);

    REQUIRE_THAT(eta.reshaped(), VectorApproxEqual(eta_sparse.reshaped()));
  }

  SECTION("Scale only and no intercept")
  {
    Eigen::MatrixXd eta = slope::linearPredictor(x,
                                                 active_set,
                                                 beta0,
                                                 beta,
                                                 x_centers,
                                                 x_scales,
                                                 slope::JitNormalization::Scale,
                                                 false);

    Eigen::MatrixXd eta_sparse =
      slope::linearPredictor(x_sparse,
                             active_set,
                             beta0,
                             beta,
                             x_centers,
                             x_scales,
                             slope::JitNormalization::Scale,
                             false);

    REQUIRE_THAT(eta.reshaped(), VectorApproxEqual(eta_sparse.reshaped()));
  }

  SECTION("With intercept")
  {
    Eigen::MatrixXd eta = slope::linearPredictor(x,
                                                 active_set,
                                                 beta0,
                                                 beta,
                                                 x_centers,
                                                 x_scales,
                                                 slope::JitNormalization::None,
                                                 true);

    Eigen::MatrixXd eta_sparse =
      slope::linearPredictor(x_sparse,
                             active_set,
                             beta0,
                             beta,
                             x_centers,
                             x_scales,
                             slope::JitNormalization::None,
                             true);

    REQUIRE_THAT(eta.reshaped(), VectorApproxEqual(eta_sparse.reshaped()));
  }

  SECTION("Parallel execution with large problem")
  {
    // Create a larger problem to trigger parallel execution
    int n = 1e4;
    int p = 1000;
    int m = 2;

    auto data = generateData(n, p, "quadratic", m, 0.3, 1);

    Eigen::MatrixXd x = data.x;
    Eigen::SparseMatrix<double> x_sparse = x.sparseView();

    Eigen::VectorXd beta0(m);
    beta0 << 0.1, -0.1;

    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p * m);
    // Make enough coefficients non-zero to trigger parallel execution
    for (int i = 0; i < p * m; i += 10) {
      beta(i) = 0.5;
    }

    Eigen::VectorXd centers(p);
    Eigen::VectorXd scales(p);

    slope::computeCenters(centers, x, "min");
    slope::computeScales(scales, x, "max_abs");

    // Create an active set with 150 indices to trigger parallel execution
    std::vector<int> active_set;
    for (int i = 0; i < 1000; i++) {
      active_set.push_back(i);
    }

    slope::Threads::set(1);
    Eigen::MatrixXd eta_seq =
      slope::linearPredictor(x,
                             active_set,
                             beta0,
                             beta,
                             centers,
                             scales,
                             slope::JitNormalization::Both,
                             true);

    slope::Threads::set(2);
    Eigen::MatrixXd eta_par =
      slope::linearPredictor(x,
                             active_set,
                             beta0,
                             beta,
                             centers,
                             scales,
                             slope::JitNormalization::Both,
                             true);

    Eigen::MatrixXd eta_par_sparse =
      slope::linearPredictor(x_sparse,
                             active_set,
                             beta0,
                             beta,
                             centers,
                             scales,
                             slope::JitNormalization::Both,
                             true);

    REQUIRE_THAT(eta_seq.reshaped(), VectorApproxEqual(eta_par.reshaped()));
    REQUIRE_THAT(eta_seq.reshaped(),
                 VectorApproxEqual(eta_par_sparse.reshaped()));
  }
}

TEST_CASE("Gradient computations", "[math][updateGradient]")
{
  using namespace Catch::Matchers;

  // Create a simple matrix
  int n = 6;
  int p = 2;
  int m = 2;
  Eigen::MatrixXd x(n, p);
  // clang-format off
  x <<  1.0,  2.0,
        3.0,  3.0,
        0.0,  4.0,
        5.0, -6.0,
       -1.0,  0.0,
        0.0,  0.0;
  // clang-format on

  Eigen::SparseMatrix<double> x_sparse = x.sparseView();

  // Create residuals (e.g., from a model prediction)
  Eigen::MatrixXd residual(n, m);
  // clang-format off
  residual << 0.5, -0.2,
              0.1,  0.3,
             -0.4,  0.5,
              0.7, -0.1,
             -0.2,  0.0,
              0.3,  0.2;
  // clang-format on

  // Centers and scales for the features
  Eigen::VectorXd x_centers(p);
  Eigen::VectorXd x_scales(p);

  x_centers << -1.0, 2.0;
  x_scales << 2.0, 2.0;

  // Create an active set with all indices (0, 1, 2, 3)
  std::vector<int> active_set = { 0, 1, 2, 3 };

  // Working weights - typically 1.0 for linear models, different for GLMs
  Eigen::VectorXd w = Eigen::VectorXd::Ones(n);

  SECTION("No normalization")
  {
    // Test with dense matrix
    Eigen::VectorXd gradient_dense = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_dense,
                          x,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::None);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_sparse,
                          x_sparse,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::None);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));

    // Manually compute expected values for verification
    double expected_0_0 = x.col(0).dot(residual.col(0)) / n;
    double expected_1_0 = x.col(1).dot(residual.col(0)) / n;
    double expected_0_1 = x.col(0).dot(residual.col(1)) / n;
    double expected_1_1 = x.col(1).dot(residual.col(1)) / n;

    REQUIRE_THAT(gradient_dense(0), WithinAbs(expected_0_0, 1e-10));
    REQUIRE_THAT(gradient_dense(p), WithinAbs(expected_0_1, 1e-10));
    REQUIRE_THAT(gradient_dense(1), WithinAbs(expected_1_0, 1e-10));
    REQUIRE_THAT(gradient_dense(p + 1), WithinAbs(expected_1_1, 1e-10));
  }

  SECTION("With centering")
  {
    // Test with dense matrix
    Eigen::VectorXd gradient_dense = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_dense,
                          x,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Center);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_sparse,
                          x_sparse,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Center);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));
  }

  SECTION("With scaling")
  {
    // Test with dense matrix
    Eigen::VectorXd gradient_dense = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_dense,
                          x,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Scale);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_sparse,
                          x_sparse,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Scale);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));
  }

  SECTION("With both centering and scaling")
  {
    // Test with dense matrix
    Eigen::VectorXd gradient_dense = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_dense,
                          x,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Both);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_sparse,
                          x_sparse,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Both);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));
  }

  SECTION("With non-uniform weights")
  {
    // Set up non-uniform weights
    w << 0.5, 1.2, 0.8, 1.0, 0.7, 0.9;

    // Test with dense matrix
    Eigen::VectorXd gradient_dense = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_dense,
                          x,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Both);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse = Eigen::VectorXd::Zero(p * m);
    slope::updateGradient(gradient_sparse,
                          x_sparse,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::Both);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));
  }
}

TEST_CASE("Gradient offset calculations", "[math][offsetGradient]")
{
  using namespace Catch::Matchers;

  // Create a simple matrix
  int n = 6;
  int p = 2;
  int m = 2;
  Eigen::MatrixXd x(n, p);
  // clang-format off
  x <<  1.0,  2.0,
        3.0,  3.0,
        0.0,  4.0,
        5.0, -6.0,
       -1.0,  0.0,
        0.0,  0.0;
  // clang-format on

  Eigen::SparseMatrix<double> x_sparse = x.sparseView();

  // Centers and scales for the features
  Eigen::VectorXd x_centers(p);
  Eigen::VectorXd x_scales(p);

  x_centers << -1.0, 2.0;
  x_scales << 2.0, 2.0;

  // Create an active set with all indices (0, 1, 2, 3)
  std::vector<int> active_set = { 0, 1, 2, 3 };

  // Create offset vector for each outcome
  Eigen::VectorXd offset(m);
  offset << 0.5, -0.3;

  // Compute column sums for manual verification
  Eigen::VectorXd col_sums(p);
  for (int j = 0; j < p; ++j) {
    col_sums(j) = x.col(j).sum();
  }

  SECTION("No normalization")
  {
    // Test with dense matrix
    Eigen::VectorXd gradient_dense = Eigen::VectorXd::Zero(p * m);
    Eigen::VectorXd gradient_dense_copy = gradient_dense;
    slope::offsetGradient(gradient_dense,
                          x,
                          offset,
                          active_set,
                          x_centers,
                          x_scales,
                          slope::JitNormalization::None);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse = Eigen::VectorXd::Zero(p * m);
    Eigen::VectorXd gradient_sparse_copy = gradient_sparse;
    slope::offsetGradient(gradient_sparse,
                          x_sparse,
                          offset,
                          active_set,
                          x_centers,
                          x_scales,
                          slope::JitNormalization::None);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));

    // Manual verification for the first outcome
    double expected_offset_0_0 = -offset(0) * col_sums(0) / n;
    double expected_offset_1_0 = -offset(0) * col_sums(1) / n;

    // Manual verification for the second outcome
    double expected_offset_0_1 = -offset(1) * col_sums(0) / n;
    double expected_offset_1_1 = -offset(1) * col_sums(1) / n;

    REQUIRE_THAT(gradient_dense(0) - gradient_dense_copy(0),
                 WithinAbs(expected_offset_0_0, 1e-10));
    REQUIRE_THAT(gradient_dense(1) - gradient_dense_copy(1),
                 WithinAbs(expected_offset_1_0, 1e-10));
    REQUIRE_THAT(gradient_dense(p) - gradient_dense_copy(p),
                 WithinAbs(expected_offset_0_1, 1e-10));
    REQUIRE_THAT(gradient_dense(p + 1) - gradient_dense_copy(p + 1),
                 WithinAbs(expected_offset_1_1, 1e-10));
  }

  SECTION("With centering")
  {
    // Test with dense matrix
    Eigen::VectorXd gradient_dense =
      Eigen::VectorXd::Ones(p * m); // Start with non-zero values
    Eigen::VectorXd gradient_dense_copy = gradient_dense;
    slope::offsetGradient(gradient_dense,
                          x,
                          offset,
                          active_set,
                          x_centers,
                          x_scales,
                          slope::JitNormalization::Center);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse =
      Eigen::VectorXd::Ones(p * m); // Start with non-zero values
    Eigen::VectorXd gradient_sparse_copy = gradient_sparse;
    slope::offsetGradient(gradient_sparse,
                          x_sparse,
                          offset,
                          active_set,
                          x_centers,
                          x_scales,
                          slope::JitNormalization::Center);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));

    // Manual verification for first feature with first outcome
    double expected_offset_0_0 = -offset(0) * (col_sums(0) / n - x_centers(0));
    REQUIRE_THAT(gradient_dense(0) - gradient_dense_copy(0),
                 WithinAbs(expected_offset_0_0, 1e-10));
  }

  SECTION("With scaling")
  {
    // Test with dense matrix
    Eigen::VectorXd gradient_dense = Eigen::VectorXd::Zero(p * m);
    Eigen::VectorXd gradient_dense_copy = gradient_dense;
    slope::offsetGradient(gradient_dense,
                          x,
                          offset,
                          active_set,
                          x_centers,
                          x_scales,
                          slope::JitNormalization::Scale);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse = Eigen::VectorXd::Zero(p * m);
    slope::offsetGradient(gradient_sparse,
                          x_sparse,
                          offset,
                          active_set,
                          x_centers,
                          x_scales,
                          slope::JitNormalization::Scale);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));

    // Manual verification for first feature with first outcome
    double expected_offset_0_0 = -offset(0) * col_sums(0) / (n * x_scales(0));
    REQUIRE_THAT(gradient_dense(0) - gradient_dense_copy(0),
                 WithinAbs(expected_offset_0_0, 1e-10));
  }

  SECTION("With both centering and scaling")
  {
    // Test with dense matrix
    Eigen::VectorXd gradient_dense = Eigen::VectorXd::Zero(p * m);
    Eigen::VectorXd gradient_dense_copy = gradient_dense;
    slope::offsetGradient(gradient_dense,
                          x,
                          offset,
                          active_set,
                          x_centers,
                          x_scales,
                          slope::JitNormalization::Both);

    // Test with sparse matrix
    Eigen::VectorXd gradient_sparse = Eigen::VectorXd::Zero(p * m);
    slope::offsetGradient(gradient_sparse,
                          x_sparse,
                          offset,
                          active_set,
                          x_centers,
                          x_scales,
                          slope::JitNormalization::Both);

    // Check that dense and sparse implementations produce the same result
    REQUIRE_THAT(gradient_dense, VectorApproxEqual(gradient_sparse));

    // Manual verification for first feature with first outcome
    double expected_offset_0_0 =
      -offset(0) * (col_sums(0) / n - x_centers(0)) / x_scales(0);
    REQUIRE_THAT(gradient_dense(0) - gradient_dense_copy(0),
                 WithinAbs(expected_offset_0_0, 1e-10));
  }
}
