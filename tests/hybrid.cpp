#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
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

TEST_CASE("Randomized CD", "[quadratic][hybrid]")
{
  using namespace Catch::Matchers;

  auto data = generateData(100, 2000);

  slope::Slope model;
  model.setSolver("hybrid");

  model.setHybridCdType("cyclical");
  auto fit_cyclical = model.fit(data.x, data.y);

  model.setHybridCdType("permuted");
  auto fit_permuted = model.fit(data.x, data.y);

  Eigen::VectorXd coefs_cyclical = fit_cyclical.getCoefs();
  Eigen::VectorXd coefs_permuted = fit_permuted.getCoefs();

  REQUIRE_THAT(coefs_cyclical, VectorApproxEqual(coefs_permuted));
}
