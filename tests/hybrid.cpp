#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>
#include <slope/clusters.h>
#include <slope/regularization_sequence.h>
#include <slope/slope.h>
#include <slope/solvers/hybrid_cd.h>

TEST_CASE("Hybrid derivatives ignore overflow on zero-weight rows",
          "[hybrid][weights]")
{
  using namespace Catch::Matchers;
  Eigen::MatrixXd x(3, 2), residual(3, 1), weights(3, 1);
  x << 1.0, -2.0, 3.0, 1.0, 1e200, -1e200;
  residual << 0.5, -0.1, std::numeric_limits<double>::infinity();
  weights << 1.0, 2.0, 0.0;
  Eigen::MatrixXd safe_x = x, safe_residual = residual;
  safe_x.row(2).setZero();
  safe_residual.row(2).setZero();
  Eigen::SparseMatrix<double> sparse_x = x.sparseView();
  const Eigen::Vector2d centers(0.5, -0.5), scales(2.0, 0.5);
  const Eigen::Vector2d beta(1.0, -1.0);
  slope::Clusters clusters(beta);

  for (const auto normalization : { slope::JitNormalization::None,
                                    slope::JitNormalization::Center,
                                    slope::JitNormalization::Scale,
                                    slope::JitNormalization::Both }) {
    const auto expected = slope::computeGradientAndHessian(safe_x,
                                                           0,
                                                           weights,
                                                           safe_residual,
                                                           centers,
                                                           scales,
                                                           1.0,
                                                           normalization,
                                                           3);
    const auto dense = slope::computeGradientAndHessian(
      x, 0, weights, residual, centers, scales, 1.0, normalization, 3);
    const auto sparse = slope::computeGradientAndHessian(
      sparse_x, 0, weights, residual, centers, scales, 1.0, normalization, 3);
    REQUIRE_THAT(dense.first, WithinAbs(expected.first, 1e-12));
    REQUIRE_THAT(dense.second, WithinAbs(expected.second, 1e-12));
    REQUIRE_THAT(sparse.first, WithinAbs(expected.first, 1e-12));
    REQUIRE_THAT(sparse.second, WithinAbs(expected.second, 1e-12));

    const auto expected_cluster =
      slope::computeClusterGradientAndHessian(safe_x,
                                              0,
                                              { 1, -1 },
                                              clusters,
                                              weights,
                                              safe_residual,
                                              centers,
                                              scales,
                                              normalization);
    const auto dense_cluster =
      slope::computeClusterGradientAndHessian(x,
                                              0,
                                              { 1, -1 },
                                              clusters,
                                              weights,
                                              residual,
                                              centers,
                                              scales,
                                              normalization);
    const auto sparse_cluster =
      slope::computeClusterGradientAndHessian(sparse_x,
                                              0,
                                              { 1, -1 },
                                              clusters,
                                              weights,
                                              residual,
                                              centers,
                                              scales,
                                              normalization);
    REQUIRE_THAT(dense_cluster.first, WithinAbs(expected_cluster.first, 1e-12));
    REQUIRE_THAT(dense_cluster.second,
                 WithinAbs(expected_cluster.second, 1e-12));
    REQUIRE_THAT(sparse_cluster.first,
                 WithinAbs(expected_cluster.first, 1e-12));
    REQUIRE_THAT(sparse_cluster.second,
                 WithinAbs(expected_cluster.second, 1e-12));
  }

  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd zero_beta = Eigen::VectorXd::Zero(2);
  slope::Clusters zero_clusters(zero_beta);
  std::mt19937 rng(42);
  slope::coordinateDescent(beta0,
                           zero_beta,
                           residual,
                           zero_clusters,
                           Eigen::ArrayXd::Zero(3),
                           x,
                           weights,
                           weights.colwise().sum().transpose(),
                           centers,
                           scales,
                           true,
                           slope::JitNormalization::None,
                           false,
                           rng);
  REQUIRE_THAT(beta0(0), WithinAbs(-0.1, 1e-12));
}

TEST_CASE("Cluster gradient and Hessian computation", "[hybrid]")
{
  using namespace Catch::Matchers;
  using namespace slope;

  const int n = 3;
  const int p = 4;

  Eigen::MatrixXd x(n, p);
  // clang-format off
  x << 1.0, 2.0, 3.0, 0.0,
       2.0, 0.0, 0.0, 5.0,
       3.0, 4.0, 0.0, 6.0;
  // clang-format on

  Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd residual(n);
  residual << 0.1, -0.2, 0.3;

  Eigen::VectorXd x_centers(p);
  x_centers << 2.0, 3.0, 4.0, 5.0;

  Eigen::VectorXd x_scales(p);
  x_scales << 1.0, 0.9, 0.4, 0.05;

  Eigen::VectorXd beta(p);
  beta << 1.0, -1.0, 0.5, -0.5;

  Clusters clusters(beta);

  std::vector<int> s;

  int j = 0;

  for (auto c_it = clusters.cbegin(j); c_it != clusters.cend(j); ++c_it) {
    double s_k = sign(beta(*c_it));
    s.emplace_back(s_k);
  }

  Eigen::SparseMatrix<double> x_sparse = x.sparseView();

  SECTION("No normalization")
  {
    auto [hessian, gradient] =
      computeClusterGradientAndHessian(x,
                                       0,
                                       s,
                                       clusters,
                                       w,
                                       residual,
                                       x_centers,
                                       x_scales,
                                       JitNormalization::None);

    REQUIRE_THAT(hessian, WithinAbs(2, 1e-6));
    REQUIRE_THAT(gradient, WithinAbs(-0.26667, 1e-4));

    auto [hessian2, gradient2] =
      computeClusterGradientAndHessian(x_sparse,
                                       0,
                                       s,
                                       clusters,
                                       w,
                                       residual,
                                       x_centers,
                                       x_scales,
                                       JitNormalization::None);

    REQUIRE(hessian == hessian2);
    REQUIRE(gradient == gradient2);
  }

  SECTION("With centering")
  {
    auto [hessian, gradient] =
      computeClusterGradientAndHessian(x,
                                       0,
                                       s,
                                       clusters,
                                       w,
                                       residual,
                                       x_centers,
                                       x_scales,
                                       JitNormalization::Center);

    REQUIRE_THAT(hessian, WithinAbs(3, 1e-6));
    REQUIRE_THAT(gradient, WithinAbs(-0.2, 1e-6));

    auto [hessian2, gradient2] =
      computeClusterGradientAndHessian(x_sparse,
                                       0,
                                       s,
                                       clusters,
                                       w,
                                       residual,
                                       x_centers,
                                       x_scales,
                                       JitNormalization::Center);

    REQUIRE_THAT(gradient, WithinAbs(gradient2, 1e-9));
    REQUIRE_THAT(hessian, WithinAbs(hessian2, 1e-9));
  }

  SECTION("With both centering and scaling")
  {
    auto [hessian, gradient] =
      computeClusterGradientAndHessian(x,
                                       0,
                                       s,
                                       clusters,
                                       w,
                                       residual,
                                       x_centers,
                                       x_scales,
                                       JitNormalization::Both);

    REQUIRE_THAT(hessian, WithinAbs(3.7119, 1e-4));
    REQUIRE_THAT(gradient, WithinAbs(-0.2296, 1e-4));

    auto [hessian2, gradient2] =
      computeClusterGradientAndHessian(x_sparse,
                                       0,
                                       s,
                                       clusters,
                                       w,
                                       residual,
                                       x_centers,
                                       x_scales,
                                       JitNormalization::Both);

    REQUIRE_THAT(hessian, WithinAbs(hessian2, 1e-9));
    REQUIRE_THAT(gradient, WithinAbs(gradient2, 1e-9));
  }
}

TEST_CASE("Sparse cluster derivatives agree for multiple responses", "[hybrid]")
{
  using namespace Catch::Matchers;
  using namespace slope;

  constexpr int n = 5;
  constexpr int p = 4;
  constexpr int m = 3;

  Eigen::MatrixXd x(n, p);
  // clang-format off
  x << 1.0,  1.0, 0.0,  2.0,
       0.0,  0.0, 3.0,  0.0,
       2.0,  2.0, 0.0, -1.0,
       0.0,  0.0, 4.0,  0.0,
      -1.0, -1.0, 0.0,  3.0;
  // clang-format on

  Eigen::MatrixXd weights(n, m);
  weights << 1.0, 0.5, 1.5, 0.8, 1.2, 0.7, 1.1, 0.9, 1.3, 0.6, 1.4, 0.4, 1.7,
    0.3, 1.0;

  Eigen::MatrixXd residual(n, m);
  residual << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
    1.3, -1.4, 1.5;

  Eigen::VectorXd x_centers(p);
  x_centers << 0.4, -0.2, 0.7, -0.5;

  Eigen::VectorXd x_scales(p);
  x_scales << 0.8, 1.2, 0.5, 1.7;

  Eigen::VectorXd beta = Eigen::VectorXd::Zero(p * m);
  beta(0) = 2.0;
  beta(1) = -2.0;
  beta(p + 2) = 2.0;
  beta(2 * p + 3) = -2.0;

  Clusters clusters(beta);
  std::vector<int> signs;
  for (auto it = clusters.cbegin(0); it != clusters.cend(0); ++it) {
    signs.emplace_back(sign(beta(*it)));
  }

  const Eigen::SparseMatrix<double> x_sparse = x.sparseView();
  const std::array normalizations{ JitNormalization::None,
                                   JitNormalization::Center,
                                   JitNormalization::Scale,
                                   JitNormalization::Both };

  for (const JitNormalization normalization : normalizations) {
    const auto [dense_hessian, dense_gradient] =
      computeClusterGradientAndHessian(x,
                                       0,
                                       signs,
                                       clusters,
                                       weights,
                                       residual,
                                       x_centers,
                                       x_scales,
                                       normalization);
    const auto [sparse_hessian, sparse_gradient] =
      computeClusterGradientAndHessian(x_sparse,
                                       0,
                                       signs,
                                       clusters,
                                       weights,
                                       residual,
                                       x_centers,
                                       x_scales,
                                       normalization);

    REQUIRE_THAT(sparse_hessian, WithinAbs(dense_hessian, 1e-12));
    REQUIRE_THAT(sparse_gradient, WithinAbs(dense_gradient, 1e-12));
  }
}

TEST_CASE("Sparse singleton derivatives agree with dense derivatives",
          "[hybrid]")
{
  using namespace Catch::Matchers;
  using namespace slope;

  constexpr int n = 5;
  constexpr int p = 4;
  constexpr int m = 3;

  Eigen::MatrixXd x(n, p);
  // clang-format off
  x << 1.0,  0.0, 2.0,  0.0,
       0.0, -1.0, 0.0,  3.0,
       4.0,  0.0, 0.0, -2.0,
       0.0,  5.0, 1.0,  0.0,
      -3.0,  0.0, 0.0,  6.0;
  // clang-format on

  Eigen::MatrixXd weights(n, m);
  weights << 1.0, 0.5, 1.5, 0.8, 1.2, 0.7, 1.1, 0.9, 1.3, 0.6, 1.4, 0.4, 1.7,
    0.3, 1.0;

  Eigen::MatrixXd residual(n, m);
  residual << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
    1.3, -1.4, 1.5;

  Eigen::VectorXd x_centers(p);
  x_centers << 0.4, -0.2, 0.7, -0.5;

  Eigen::VectorXd x_scales(p);
  x_scales << 0.8, 1.2, 0.5, 1.7;

  const Eigen::SparseMatrix<double> x_sparse = x.sparseView();
  const Eigen::VectorXd weight_sums = weights.colwise().sum().transpose();
  const std::array normalizations{ JitNormalization::None,
                                   JitNormalization::Center,
                                   JitNormalization::Scale,
                                   JitNormalization::Both };

  for (const JitNormalization normalization : normalizations) {
    for (int ind = 0; ind < p * m; ++ind) {
      const double coefficient_sign = ind % 2 == 0 ? 1.0 : -1.0;
      const auto [dense_gradient, dense_hessian] =
        computeGradientAndHessian(x,
                                  ind,
                                  weights,
                                  residual,
                                  x_centers,
                                  x_scales,
                                  coefficient_sign,
                                  normalization,
                                  n);
      const auto [sparse_gradient, sparse_hessian] =
        computeGradientAndHessian(x_sparse,
                                  ind,
                                  weights,
                                  residual,
                                  x_centers,
                                  x_scales,
                                  coefficient_sign,
                                  normalization,
                                  n);
      const auto [cached_dense_gradient, cached_dense_hessian] =
        computeGradientAndHessian(x,
                                  ind,
                                  weights,
                                  residual,
                                  x_centers,
                                  x_scales,
                                  coefficient_sign,
                                  normalization,
                                  n,
                                  weight_sums);
      const auto [cached_gradient, cached_hessian] =
        computeGradientAndHessian(x_sparse,
                                  ind,
                                  weights,
                                  residual,
                                  x_centers,
                                  x_scales,
                                  coefficient_sign,
                                  normalization,
                                  n,
                                  weight_sums);

      REQUIRE_THAT(sparse_gradient, WithinAbs(dense_gradient, 1e-12));
      REQUIRE_THAT(sparse_hessian, WithinAbs(dense_hessian, 1e-12));
      REQUIRE_THAT(cached_dense_gradient, WithinAbs(dense_gradient, 1e-12));
      REQUIRE_THAT(cached_dense_hessian, WithinAbs(dense_hessian, 1e-12));
      REQUIRE_THAT(cached_gradient, WithinAbs(dense_gradient, 1e-12));
      REQUIRE_THAT(cached_hessian, WithinAbs(dense_hessian, 1e-12));
    }
  }
}

TEST_CASE("Randomized CD", "[quadratic][hybrid]")
{
  using namespace Catch::Matchers;

  auto data = generateData(100, 2000);

  slope::Slope model;
  model.setSolver("hybrid");

  model.setHybridCdType("cyclical");
  auto fit_cyclical = model.fit(data.x, data.y);

  model.setHybridCdType("permuted");
  model.setRandomSeed(40);
  auto fit_permuted = model.fit(data.x, data.y);

  Eigen::VectorXd coefs_cyclical = fit_cyclical.getCoefs();
  Eigen::VectorXd coefs_permuted = fit_permuted.getCoefs();

  REQUIRE_THAT(coefs_cyclical, VectorApproxEqual(coefs_permuted));
}
