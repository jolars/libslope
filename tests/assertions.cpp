#include "../src/slope/slope.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

TEST_CASE("Assertions", "[assertions]")
{
  const int n = 10;
  const int p = 3;

  Eigen::Matrix<double, n, p> x;
  Eigen::VectorXd y(n);

  slope::Slope model;

  SECTION("Invalid family")
  {
    REQUIRE_THROWS(model.setObjective("ols"));
  }

  SECTION("Invalid X, y dimensions")
  {
    Eigen::VectorXd y(n - 1);

    REQUIRE_THROWS(model.fit(x, y));
  }
}
