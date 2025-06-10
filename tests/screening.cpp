#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <slope/kkt_check.h>
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

    REQUIRE(!violations.empty());
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

TEST_CASE("Gaps on screened path", "[screening][gaps][fail]")
{
  slope::Slope model;
  model.setPathLength(100);
  model.setDiagnostics(true);
  model.setMaxIterations(100000);
  // model.setScreening("none");

  double tol = 1e-5;

  model.setTol(tol);

  auto data = generateData(10, 15, "quadratic", 1, 1, 0.2);

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
