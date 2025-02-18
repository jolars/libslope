#include "../src/slope/slope.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

TEST_CASE("Assertions", "[assertions]")
{
  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
  Eigen::MatrixXd y(n, 1);

  slope::Slope model;

  SECTION("Invalid family")
  {
    REQUIRE_THROWS(model.setObjective("ols"));
  }

  SECTION("Invalid X, y dimensions")
  {
    Eigen::MatrixXd y(n - 1, 1);

    REQUIRE_THROWS(model.fit(x, y));
  }

  SECTION("Invalid lambda type")
  {
    Eigen::VectorXd lambda(p + 1);
    REQUIRE_THROWS(model.setLambdaType("l1"));
  }

  SECTION("Invalid max iterations")
  {
    REQUIRE_THROWS(model.setMaxIt(0));
    REQUIRE_THROWS(model.setMaxItInner(-1));
  }

  SECTION("Invalid early stopping criteria")
  {
    REQUIRE_THROWS(model.setDevChangeTol(1.1));
    REQUIRE_THROWS(model.setMaxClusters(0));
    REQUIRE_THROWS(model.setDevRatioTol(-1));
  }
}
