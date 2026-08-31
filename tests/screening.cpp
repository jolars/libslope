#include "../src/slope/kkt_check.h"
#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <slope/screening.h>
#include <slope/slope.h>

TEST_CASE("Strong screening rule", "[screening]")
{
  using namespace Catch::Matchers;
  using namespace slope;

  const int p = 3;
  const int n = 4;

  Eigen::VectorXd beta(p);
  Eigen::VectorXd beta_hat(p);
  Eigen::MatrixXd x(n, p);
  Eigen::VectorXd gradient(p);
  Eigen::VectorXd residual(n);
  Eigen::ArrayXd lambda(p);

  // clang-format off
  x <<  1.23924512, -0.5494198, -1.6060947,
        0.36553273,  1.7317157,  0.1894368,
       -1.52453897, -0.5772386,  0.2718006,
       -0.08023889, -0.6050573,  1.1448572;
  // clang-format on

  beta << 3, 3, 2;
  lambda << 0.3, 0.2, 0.1;

  // Calculate y
  Eigen::VectorXd y = x * beta;

  // Correct solution
  beta_hat << 2.675938, 2.796458, 1.721710;

  SECTION("KKT checks")
  {
    // Modified, incorrect solution, which should now have a KKT violation
    beta_hat(0) = 0.0;

    residual = x * beta_hat - y;
    gradient = x.transpose() * residual;

    auto violations = kktCheck(gradient, beta_hat, lambda, { 0, 1, 2 });
    auto full_violations = kktCheck(gradient, beta_hat, lambda);

    REQUIRE(!violations.empty());
    REQUIRE(full_violations == violations);
  }

  SECTION("Strong screening")
  {
    lambda *= 10 * 1.25 * n;
    beta_hat << 0.0, 0.09096501, 0.0;

    residual = x * beta_hat - y;
    gradient = x.transpose() * residual;

    Eigen::ArrayXd lambda_prev = lambda;
    lambda *= 0.99;

    auto strong_set = strongSet(gradient, lambda, lambda_prev);

    REQUIRE(strong_set.size() == 1);
  }

  SECTION("Random data")
  {
    auto data = generateData(500, 50);

    slope::Slope model;

    model.setScreening("none");
    auto fit = model.path(data.x, data.y);
    Eigen::VectorXd coefs = fit.getCoefs().back();

    model.setScreening("strong");
    fit = model.path(data.x, data.y);
    Eigen::VectorXd coefs_screen = fit.getCoefs().back();

    REQUIRE_THAT(coefs, VectorApproxEqual(coefs_screen, 1e-4));
  }
}

TEST_CASE("Gaps on screened path", "[screening][gaps]")
{
  slope::Slope model;
  model.setPathLength(100);
  model.setDiagnostics(true);

  double tol = 1e-5;

  model.setTol(tol);

  auto data = generateData(100, 5, "quadratic", 1, 1, 0.2);

  auto path = model.path(data.x, data.y);

  for (int step = 0; step < path.size(); step++) {
    auto fit = path(step);
    auto gaps = fit.getGaps();
    auto primals = fit.getPrimals();

    DYNAMIC_SECTION("Step: " << step)
    {
      REQUIRE_FALSE(slope::WarningLogger::hasWarnings());
      REQUIRE(gaps.back() / primals.back() <= tol);
    }
  }
}

TEST_CASE("Screening rules manage their working sets", "[screening]")
{
  constexpr int feature_count = 4;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(feature_count);
  Eigen::VectorXd beta = Eigen::VectorXd::Zero(feature_count);
  Eigen::ArrayXd lambda_curr = Eigen::ArrayXd::Ones(feature_count);
  Eigen::ArrayXd lambda_prev = Eigen::ArrayXd::Ones(feature_count);
  const std::vector<int> expected = { 0, 1, 2, 3 };

  SECTION("No screening retains all coefficients")
  {
    auto rule = slope::createScreeningRule("none");
    auto working_set = rule->initialize(feature_count, 0);

    rule->screen(working_set, gradient, lambda_curr, lambda_prev, beta);

    REQUIRE(working_set == expected);
  }

  SECTION("Zero regularization disables strong screening")
  {
    auto rule = slope::createScreeningRule("strong");
    auto working_set = rule->initialize(feature_count, 0);
    lambda_curr.setZero();

    rule->screen(working_set, gradient, lambda_curr, lambda_prev, beta);

    REQUIRE(working_set == expected);
  }
}

TEST_CASE("Strong screening respects the iteration limit", "[screening]")
{
  constexpr int max_iterations = 5;
  auto data = generateData(40, 30, "quadratic", 1, 1.0, 0.2, 5);

  slope::Slope model;
  model.setMaxIterations(max_iterations);
  model.setTol(1e-3);
  model.setScreening("strong");

  slope::WarningLogger::clearWarnings();
  auto fit = model.fit(data.x, data.y, 0.05);
  slope::WarningLogger::clearWarnings();

  REQUIRE(fit.getPasses() <= max_iterations);
}
