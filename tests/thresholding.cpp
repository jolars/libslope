#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/math.h>
#include <slope/regularization_sequence.h>
#include <slope/solvers/slope_threshold.h>

TEST_CASE("SlopeThreshold function", "[slope][solvers]")
{
  using namespace Catch::Matchers;

  // Create a sample Clusters object
  // Assuming Clusters has a constructor that can take these parameters
  int p = 5;
  Eigen::VectorXd beta(p);
  Eigen::ArrayXd lambdas(p);

  beta << 4, -1, 4, 0.5, 0;
  lambdas << 4, 3.0, 2.0, 1.0, 0.5;

  Eigen::ArrayXd lambda_cumsum = slope::cumSum(lambdas, true);

  // Three clusters: {0, 2}, {1}, {3}
  // Coefficients:     4,     1,  0.5

  slope::Clusters clusters(beta);

  SECTION("Direction up")
  {
    // Test case for direction_up = true
    double x = 10.0; // Large enough to trigger direction_up
    int j = 1;

    auto [y, idx] = slopeThreshold(x, j, lambda_cumsum, clusters);

    // Should be new top cluster
    REQUIRE(y == 6.0);
    REQUIRE(idx == 0);
  }

  SECTION("Direction down")
  {
    // Test case for direction_up = false
    double x = 3.5; // Small enough to not trigger direction_up
    int j = 1;

    auto [y, idx] = slopeThreshold(x, j, lambda_cumsum, clusters);

    REQUIRE(y == 1.5);
    REQUIRE(idx == 1);

    x = 2.9; // Should not move
    j = 1;

    std::tie(y, idx) = slopeThreshold(x, j, lambda_cumsum, clusters);

    REQUIRE(idx == 1);
    REQUIRE_THAT(y, WithinAbs(0.9, 1e-4));

    x = 1; // Should merge with zero cluster
    j = 1;

    std::tie(y, idx) = slopeThreshold(x, j, lambda_cumsum, clusters);

    REQUIRE(idx == 3);
    REQUIRE(y == 0);

    x = 2.9; // Should merge with second cluster
    j = 2;

    std::tie(y, idx) = slopeThreshold(x, j, lambda_cumsum, clusters);

    REQUIRE(idx == 1);
    REQUIRE(y == 1);
  }

  SECTION("Negative input")
  {
    // Test a boundary case
    double x = -9.0;
    int j = 2;

    auto [y, idx] = slopeThreshold(x, j, lambda_cumsum, clusters);

    REQUIRE(y == -5);
    REQUIRE(idx == 0);

    x = -5.0;
    j = 2;

    std::tie(y, idx) = slopeThreshold(x, j, lambda_cumsum, clusters);

    CHECK(y <= 0); // Output should maintain sign of input
    CHECK(idx >= 0);
    CHECK(idx < clusters.size());
  }
}
