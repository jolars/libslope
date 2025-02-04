#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <slope/regularization_sequence.h>

TEST_CASE("Test that regularization sequence generation works",
          "[regularization sequence]")
{
  double tol = 1e-6;
  int n = 10;
  int p = 4;
  double q = 0.3;

  SECTION("BH sequence")
  {
    Eigen::ArrayXd lambda1 = slope::lambdaSequence(p, 0.1, "bh");

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

  SECTION("Gaussian sequence")
  {

    Eigen::ArrayXd l1 = slope::lambdaSequence(p, q, "gaussian", n);
    std::vector<double> l1_ref = { 1.780464, 1.700998, 1.657533, 1.628381 };

    REQUIRE_THAT(l1, VectorApproxEqual(l1_ref, tol));

    n = 3;
    q = 0.5;

    Eigen::ArrayXd l2 = slope::lambdaSequence(p, q, "gaussian", n);
    std::vector<double> l2_ref = { 1.5341205, 1.5341205, 1.5341205, 1.5341205 };

    REQUIRE_THAT(l2, VectorApproxEqual(l2_ref, tol));
  }

  SECTION("OSCAR sequence")
  {
    Eigen::ArrayXd l3 = slope::lambdaSequence(p, q, "oscar", n, 2.0, 0.1);
    std::vector<double> l3_ref = {
      2.3,
      2.2,
      2.1,
      2.0,
    };

    REQUIRE_THAT(l3, VectorApproxEqual(l3_ref, tol));
  }

  SECTION("Lasso sequence")
  {
    Eigen::ArrayXd l3 = slope::lambdaSequence(p, q, "lasso", n);
    REQUIRE_THAT(l3, VectorApproxEqual(std::vector<double>(4, 1.0), tol));
  }

  SECTION("Assertions")
  {
    REQUIRE_THROWS(slope::lambdaSequence(p, q, "gaussian", -5));
    REQUIRE_THROWS(slope::lambdaSequence(p, 0.0, "bh"));
    REQUIRE_THROWS(slope::lambdaSequence(p, 1.0, "bh"));
    REQUIRE_THROWS(slope::lambdaSequence(p, 1.0, "oscar", 0, -1.0, 1.0));
    REQUIRE_THROWS(slope::lambdaSequence(p, 1.0, "oscar", 0, 1.0, -1.0));
  }
}
