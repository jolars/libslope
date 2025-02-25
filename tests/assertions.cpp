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
    REQUIRE_THROWS(model.setLoss("ols"));
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
  }

  SECTION("Invalid early stopping criteria")
  {
    REQUIRE_THROWS(model.setDevChangeTol(1.1));
    REQUIRE_THROWS(model.setMaxClusters(0));
    REQUIRE_THROWS(model.setDevRatioTol(-1));
  }

  SECTION("Invalid solver combinations")
  {
    model.setSolver("hybrid");
    model.setLoss("multinomial");
    y << 1, 0, 1, 2, 1, 0, 2, 0, 3, 0;
    REQUIRE_THROWS_AS(model.fit(x, y), std::invalid_argument);
  }

  SECTION("Invalid OSCAR parameters")
  {
    model.setLambdaType("oscar");
    REQUIRE_THROWS_AS(model.setOscarParameters(-0.1, 2.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(model.setOscarParameters(1.0, -2.0),
                      std::invalid_argument);
  }
}
