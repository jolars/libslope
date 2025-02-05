#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>

TEST_CASE("Poisson models", "[models][poisson]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  const int p = 3;

  Eigen::Matrix<double, n, p> x;
  Eigen::Vector3d beta;
  Eigen::VectorXd y(n);

  // clang-format off
    x << 0.288,  -0.0452,  0.880,
         0.788,   0.576,  -0.305,
         1.510,   0.390,  -0.621,
        -2.210,  -1.120,  -0.0449,
        -0.0162,  0.944,   0.821,
         0.594,   0.919,   0.782,
         0.0746, -1.990,   0.620,
        -0.0561, -0.156,  -1.470,
        -0.478,   0.418,   1.360,
        -0.103,   0.388,  -0.0538;
  // clang-format on

  beta << 0.5, -0.5, 0.0;

  y << 2, 0, 1, 0, 0, 0, 1, 0, 1, 2;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  alpha[0] = 0.01;
  lambda << 2.0, 1.8, 1.0;

  slope::Slope model;

  model.setTol(1e-6);
  model.setObjective("poisson");

  Eigen::Vector3d coefs_ref;

  SECTION("No intercept, no standardization")
  {
    model.setStandardize(false);
    model.setIntercept(false);

    model.fit(x, y, alpha, lambda);

    auto dual_gaps = model.getDualGaps().front();

    REQUIRE(dual_gaps.back() >= 0);
    REQUIRE(dual_gaps.back() <= 1e-4);

    Eigen::VectorXd coefs = model.getCoefs().front();

    coefs_ref << 0.1957634, -0.1612890, 0.1612890;

    REQUIRE_THAT(coefs, VectorApproxEqual(coefs_ref, 1e-6));
  }

  SECTION("With intercept, with standardization")
  {
    model.setStandardize(true);
    model.setIntercept(true);

    model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs = model.getCoefs().front();
    double intercept = model.getIntercepts()[0][0];

    coefs_ref << 0.4017805, -0.2396130, 0.4600816;

    REQUIRE_THAT(coefs, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept, Catch::Matchers::WithinRel(-0.5482493, 1e-4));
  }
}
