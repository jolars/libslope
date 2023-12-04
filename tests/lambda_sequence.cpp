#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <slope/regularization_sequence.h>

TEST_CASE("Test that regularization sequence generation works",
          "[regularization sequence]")
{
  SECTION("Test that BH sequence works")
  {
    Eigen::ArrayXd lambda1 = slope::lambdaSequence(4, 0.1, "bh");

    std::vector<double> lambda1_expected = {
      2.24140272760495, 1.95996398454005, 1.78046434169203, 1.64485362695147
    };

    REQUIRE_THAT(lambda1, VectorApproxEqual(lambda1_expected, 1e-6));

    Eigen::ArrayXd lambda2 = slope::lambdaSequence(6, 0.5, "bh");

    std::vector<double> lambda2_expected = {
      1.73166439612225,  1.38299412710064,  1.15034938037601,
      0.967421566101701, 0.812217801499913, 0.674489750196082
    };

    REQUIRE_THAT(lambda2, VectorApproxEqual(lambda2_expected, 1e-6));
  }
}
