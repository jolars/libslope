#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/clusters.h>
#include <slope/regularization_sequence.h>
#include <slope/slope.h>
#include <slope/solvers/hybrid_cd.h>

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

      REQUIRE_THAT(sparse_gradient, WithinAbs(dense_gradient, 1e-12));
      REQUIRE_THAT(sparse_hessian, WithinAbs(dense_hessian, 1e-12));
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
