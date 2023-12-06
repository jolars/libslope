#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <slope/slope.h>

TEST_CASE("Simple low-dimensional design", "[gaussian][basic]")
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

  slope::SlopeParameters params;
  params.objective = "gaussian";
  params.intercept = false;
  params.standardize = false;

  auto res = slope::slope(x, y, alpha, lambda, params);

  Eigen::VectorXd coef = res.betas.col(0);

  REQUIRE_THAT(coef, VectorApproxEqual(beta, 1e-4));

  double gap = res.dual_gaps[0].back();
  double primal = res.primals[0].back();

  REQUIRE(gap <= primal * 1e-4);
}

TEST_CASE("X is identity", "[gaussian][identity]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(4, 4);
  Eigen::Vector4d y;
  y << 8, 6, 4, 2;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Constant(1, 1);
  Eigen::Array4d lambda;
  lambda << 1, 0.75, 0.5, 0.25;

  slope::SlopeParameters params;

  params.objective = "gaussian";
  params.intercept = false;
  params.standardize = false;

  auto res = slope::slope(x, y, alpha, lambda, params);

  double gap = res.dual_gaps[0].back();
  double primal = res.primals[0].back();

  Eigen::VectorXd betas = res.betas.col(0);

  std::array<double, 4> expected = { 4.0, 3.0, 2.0, 1.0 };

  REQUIRE_THAT(betas, VectorApproxEqual(expected));
  REQUIRE(gap < primal * 1e-4);
}
