#include "../src/slope/slope.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

TEST_CASE("Binomial, simple fixed design", "[binomial][basic]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  const int p = 3;

  Eigen::Matrix<double, n, p> x;
  Eigen::VectorXd beta(p);

  // clang-format off
    x <<  0.288,  -0.0452,  0.880,
          0.788,   0.576,  -0.305,
          1.510,   0.390,  -0.621,
         -2.210,  -1.120,  -0.0449,
         -0.0162,  0.944,   0.821,
          0.594,   0.919,   0.782,
          0.0746, -1.990,   0.620,
         -0.0561, -0.156,  -1.470,
         -0.478,   0.418,   1.360,
         -0.103,   0.388,  -0.0538;
  // clang-format on

  // Fixed coefficients beta
  beta << 0.5, -0.1, 0.2;

  // Compute linear predictor
  Eigen::VectorXd linear_predictor = x * beta;

  // Compute probabilities using logistic function
  Eigen::VectorXd prob =
    linear_predictor.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });

  // Generate deterministic response variable y
  Eigen::VectorXd y =
    prob.unaryExpr([](double p) { return p > 0.5 ? 1.0 : 0.0; });

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  alpha[0] = 0.05;
  // lambda << 2.128045 / n, 1.833915 / n, 1.644854 / n;
  lambda << 2.128045, 1.833915, 1.644854;

  slope::Slope model;

  model.setTol(1e-9);
  model.setObjective("binomial");

  SECTION("No intercept, no standardization")
  {
    model.setStandardize(false);
    model.setIntercept(false);

    Eigen::Vector3d coef_target;
    coef_target << 1.3808558, 0.0000000, 0.3205496;

    // PGD
    model.setPgdFreq(1);
    model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_pgd = model.getCoefs().col(0);

    auto dual_gaps = model.getDualGaps().front();

    REQUIRE(dual_gaps.front() >= 0);
    REQUIRE(dual_gaps.back() >= 0);
    REQUIRE(dual_gaps.back() <= 1e-4);

    // Hybrid
    model.setPgdFreq(10);
    model.fit(x, y, alpha, lambda);
    model.setTol(1e-12);

    Eigen::VectorXd coefs_hybrid = model.getCoefs().col(0);

    // auto dual_gaps = model.getDualGaps();
    // auto primals = model.getPrimals();

    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-6));
    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-6));
  }

  // SECTION("No intercept, standardization")
  // {
  //   model.setIntercept(false);
  //   model.setStandardize(true);
  //
  //   Eigen::Vector3d coef_target;
  //   coef_target << 1.4144889, 0.0000000, 0.3285132;
  //
  //   SECTION("PGD, standardized")
  //   {
  //     model.setPgdFreq(1);
  //     model.fit(x, y, alpha, lambda);
  //
  //     auto coefs = model.getCoefs();
  //     Eigen::VectorXd coef = coefs.col(0);
  //
  //     REQUIRE_THAT(coef, VectorApproxEqual(coef_target, 1e-6));
  //   }
  //
  //   SECTION("Hybrid, standardized")
  //   {
  //     // Hybrid
  //     model.setPgdFreq(10);
  //     model.fit(x, y, alpha, lambda);
  //
  //     Eigen::VectorXd coefs_hybrid = model.getCoefs().col(0);
  //
  //     REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-6));
  //   }
  // }
}
