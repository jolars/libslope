#include "../src/slope/qnorm.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>

TEST_CASE("Check that normal quantile algorithm works", "[stats]")
{
  using Catch::Matchers::WithinAbs;
  using slope::normalQuantile;

  const double tol = 1e-6;

  SECTION("Incorrect input throws an exception")
  {
    REQUIRE_THROWS(normalQuantile(-1));
    REQUIRE_THROWS(normalQuantile(1.01));
  }

  SECTION("0 and 1 return negative an positive infinity")
  {
    double inf = std::numeric_limits<double>::infinity();
    REQUIRE(normalQuantile(0) == -inf);
    REQUIRE(normalQuantile(1) == inf);
  }

  SECTION("Output conforms with R's qnorm()")
  {
    REQUIRE(normalQuantile(0.5) == 0);
    REQUIRE_THAT(normalQuantile(0.01), WithinAbs(-2.326348, tol));
    REQUIRE_THAT(normalQuantile(0.9995), WithinAbs(3.290527, tol));
    REQUIRE_THAT(normalQuantile(1e-6), WithinAbs(-4.753424, tol));
  }
}
