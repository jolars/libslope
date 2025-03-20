#include "slope/math.h"
#include "generate_data.hpp"
#include "slope/normalize.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE(
  "linearPredictor correctly computes predictions with different normalization",
  "[math][linearPredictor]")
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
