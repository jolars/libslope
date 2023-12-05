#include "../src/slope/sorted_l1_norm.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

TEST_CASE("Check that proximal operator works", "[prox]")
{
  Eigen::Vector2d beta;
  Eigen::Array2d lambda;

  beta << 5, 2;
  lambda << 4, 2;

  slope::SortedL1Norm norm(lambda);
  auto res = norm.prox(beta, 1.0);

  std::array<double, 2> expected = { 1.0, 0.0 };

  REQUIRE_THAT(res, VectorApproxEqual(expected, 1e-4));

  beta << 3, 3;
  lambda << 3, 3;

  norm.setLambda(lambda);
  res = norm.prox(beta, 1.0);
  expected = { 0.0, 0.0 };

  REQUIRE_THAT(res, VectorApproxEqual(expected, 1e-6));

  beta << 2, 1;
  lambda << 3, 0;

  norm.setLambda(lambda);
  res = norm.prox(beta, 1.0);
  expected = { 0.0, 0.0 };

  REQUIRE_THAT(res, VectorApproxEqual(expected, 1e-6));
}
