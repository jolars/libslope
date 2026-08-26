#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <slope/sorted_l1_norm.h>

TEST_CASE("The SLOPE dual norm handles a zero penalty", "[prox][dual]")
{
  slope::SortedL1Norm norm;
  Eigen::VectorXd gradient(2);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(2);

  gradient << 1.0, 0.0;
  REQUIRE(std::isinf(norm.dualNorm(gradient, lambda)));

  gradient.setZero();
  REQUIRE(norm.dualNorm(gradient, lambda) == 0.0);
}

TEST_CASE("Check that proximal operator works", "[prox]")
{
  Eigen::Vector2d beta;
  Eigen::Array2d lambda;

  beta << 5, 2;
  lambda << 4, 2;

  slope::SortedL1Norm norm;
  Eigen::VectorXd res = norm.prox(beta, lambda);

  std::array<double, 2> expected = { 1.0, 0.0 };

  REQUIRE_THAT(res, VectorApproxEqual(expected, 1e-4));

  beta << 3, 3;
  lambda << 3, 3;

  res = norm.prox(beta, lambda);
  expected = { 0.0, 0.0 };

  REQUIRE_THAT(res, VectorApproxEqual(expected, 1e-6));

  beta << 2, 1;
  lambda << 3, 0;

  res = norm.prox(beta, lambda);
  expected = { 0.0, 0.0 };

  REQUIRE_THAT(res, VectorApproxEqual(expected, 1e-6));
}
