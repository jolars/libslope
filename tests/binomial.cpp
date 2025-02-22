#include "../src/slope/slope.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Binomial, simple fixed design", "[binomial][basic]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
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
  Eigen::VectorXd prob = linear_predictor.unaryExpr(
    [](double x) { return 1.0 / (1.0 + std::exp(-x)); });

  // Generate deterministic response variable y
  Eigen::VectorXd y =
    prob.unaryExpr([](double p) { return p > 0.5 ? 1.0 : 0.0; });

  double alpha = 0.05;
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  lambda << 2.128045, 1.833915, 1.644854;

  slope::Slope model;

  model.setTol(1e-7);
  model.setObjective("binomial");
  model.setDiagnostics(true);

  slope::SlopeFit fit;

  SECTION("No intercept, no standardization")
  {
    model.setNormalization("none");
    model.setIntercept(false);

    Eigen::Vector3d coef_target;
    coef_target << 1.3808558, 0.0000000, 0.3205496;

    model.setSolver("pgd");

    fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_pgd = fit.getCoefs();

    auto dual_gaps = fit.getGaps();

    REQUIRE(dual_gaps.front() >= 0);
    REQUIRE(dual_gaps.back() >= 0);
    REQUIRE(dual_gaps.back() <= 1e-4);

    model.setSolver("hybrid");

    fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_hybrid = fit.getCoefs();

    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("Intercept, no standardization")
  {
    model.setIntercept(true);
    model.setNormalization("none");

    std::vector<double> coef_target = { 1.2748806, 0.0, 0.2062611 };
    double intercept_target = 0.3184528;

    model.setSolver("pgd");
    model.setMaxIt(1e7);
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coef_pgd = fit.getCoefs();
    double intercept_pgd = fit.getIntercepts()[0];

    REQUIRE_THAT(coef_pgd, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(intercept_pgd, WithinAbs(intercept_target, 1e-4));

    model.setSolver("hybrid");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_hybrid = fit.getCoefs();
    double intercept_hybrid = fit.getIntercepts()[0];

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(intercept_hybrid, WithinAbs(intercept_target, 1e-4));
  }
}
