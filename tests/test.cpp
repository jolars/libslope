#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <slope/slope.h>

TEST_CASE("Simple low-dimensional design", "[gaussian, dense]")
{
  using namespace Catch::Matchers;

  Eigen::Matrix<double, 3, 2> x;
  Eigen::Vector2d beta;
  Eigen::Vector3d y;

  // clang-format off
  x << 1, 2,
       0, 1,
       1, 0;
  // clang-format on
  beta << 1, 1;

  y = x * beta;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(2);

  auto no_intercept_no_std =
    slope::slope(x, y, alpha, lambda, "gaussian", false, false);

  REQUIRE_THAT(no_intercept_no_std.betas.coeff(0, 0),
               WithinAbsMatcher(1, 0.001));
  REQUIRE_THAT(no_intercept_no_std.betas.coeff(1, 0),
               WithinAbsMatcher(1, 0.001));
}

TEST_CASE("X is identity", "[gaussian, dense]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(4, 4);
  Eigen::Vector4d y;
  y << 8, 6, 4, 2;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Constant(1, 1);
  Eigen::Array4d lambda;
  lambda << 1, 0.75, 0.5, 0.25;

  auto res = slope::slope(x, y, alpha, lambda, "gaussian", false, false);

  Eigen::VectorXd betas = res.betas.col(0);

  REQUIRE_THAT(betas(0), WithinAbsMatcher(4.0, 0.001));
  REQUIRE_THAT(betas(1), WithinAbsMatcher(3.0, 0.001));
  REQUIRE_THAT(betas(2), WithinAbsMatcher(2.0, 0.001));
  REQUIRE_THAT(betas(3), WithinAbsMatcher(1.0, 0.001));
}
