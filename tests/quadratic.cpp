#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>
#include <slope/threads.h>

TEST_CASE("Guassian, simple design", "[quadratic]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(3, 2);
  Eigen::Vector2d beta;
  Eigen::VectorXd y(3);

  // clang-format off
  x << 1.1, 2.3,
       0.2, 1.5,
       0.5, 0.2;
  // clang-format on
  beta << 1, 2;

  y = x * beta;

  double alpha = 1e-12;
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Ones(2);

  slope::Slope model;
  slope::SlopeFit fit;

  model.setIntercept(false);
  model.setNormalization("none");
  model.setDiagnostics(true);

  fit = model.fit(x, y, alpha, lambda);

  Eigen::VectorXd coef = fit.getCoefs();
  auto dual_gaps = fit.getGaps();
  auto primals = fit.getPrimals();

  REQUIRE_THAT(coef, VectorApproxEqual(beta, 1e-4));

  double gap = dual_gaps.back();
  double primal = primals.back();

  REQUIRE(gap <= (primal + 1e-10) * 1e-4);
}

TEST_CASE("Quadratic, X is identity", "[quadratic]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(4, 4);
  Eigen::VectorXd y(4);
  y << 8, 6, 4, 2;

  double alpha = 1.0;
  Eigen::ArrayXd lambda(4);
  lambda << 1, 0.75, 0.5, 0.25;

  slope::Slope model;
  model.setIntercept(false);
  model.setNormalization("none");
  model.setDiagnostics(true);
  auto fit = model.fit(x, y, alpha, lambda);

  double gap = fit.getGaps().back();
  double primal = fit.getPrimals().back();

  Eigen::VectorXd coefs = fit.getCoefs();

  std::array<double, 4> expected = { 4.0, 3.0, 2.0, 1.0 };

  REQUIRE_THAT(coefs, VectorApproxEqual(expected));
  REQUIRE(gap < primal * 1e-4);
}

TEST_CASE("Quadratic, automatic lambda and alpha", "[quadratic]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(3, 2);
  // clang-format off
  x << 1, 2,
       0, 1,
       1, 0;
  // clang-format on
  Eigen::Vector3d y;
  y << -1, 6, 2;

  slope::Slope model;

  REQUIRE_NOTHROW(model.path(x, y));
}

TEST_CASE("Quadratic, various models", "[quadratic]")
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
  beta << 1, -1, 0.2;

  // Compute linear predictor
  Eigen::VectorXd y = x * beta;

  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  double alpha = 0.05;

  lambda << 3.0, 2.0, 2.0;

  Eigen::Vector3d coef_target;

  slope::Slope model;

  model.setTol(1e-8);
  model.setLoss("quadratic");
  model.setDiagnostics(true);

  SECTION("No intercept, no standardization")
  {
    model.setNormalization("none");
    model.setIntercept(false);

    coef_target << 0.6864545, -0.6864545, 0.0000000;

    model.setSolver("fista");
    auto fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();

    model.setSolver("hybrid");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_hybrid = fit.getCoefs();

    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));

    auto dual_gaps = fit.getGaps();

    REQUIRE(dual_gaps.back() >= 0);
    REQUIRE(dual_gaps.back() <= 1e-4);
  }

  SECTION("No intercept, with standardization")
  {
    model.setNormalization("standardization");
    model.setIntercept(false);
    // model.setModifyX(true);

    coef_target << 0.700657772, -0.730587233, 0.008997323;

    model.setSolver("pgd");
    auto fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();

    model.setSolver("hybrid");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_hybrid = fit.getCoefs();

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("With intercept, with standardization")
  {
    model.setNormalization("standardization");
    model.setIntercept(true);

    coef_target << 0.700657772, -0.730587234, 0.008997323;
    std::vector<double> intercept_target = { 0.040584733 };

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs = fit.getCoefs();
    Eigen::VectorXd intercept = fit.getIntercepts();

    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();
    Eigen::VectorXd intercept_pgd = fit.getIntercepts();

    REQUIRE_THAT(intercept, VectorApproxEqual(intercept_target, 1e-3));
    REQUIRE_THAT(intercept_pgd, VectorApproxEqual(intercept_target, 1e-3));

    REQUIRE_THAT(coefs, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("With intercept, no standardization")
  {
    model.setNormalization("none");
    model.setIntercept(true);
    model.setMaxIterations(1e5);

    coef_target << 0.68614138, -0.68614138, 0.00000000;

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs = fit.getCoefs();
    double intercept = fit.getIntercepts()[0];

    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();
    double intercept_pgd = fit.getIntercepts()[0];

    REQUIRE_THAT(intercept, WithinAbs(0.04148455, 1e-3));
    REQUIRE_THAT(intercept_pgd, WithinAbs(0.04148455, 1e-3));

    REQUIRE_THAT(coefs, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));

    auto time = fit.getTime();

    REQUIRE_THAT(time, VectorMonotonic(true, false));

    auto passes = fit.getPasses();
    REQUIRE(passes > 0);
    REQUIRE(passes < 1e5);
  }

  SECTION("mtcars data")
  {
    Eigen::MatrixXd x(32, 1);
    Eigen::MatrixXd y(32, 1);

    x << 21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4,
      17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3,
      19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4;
    y << 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
      0, 1, 0, 1, 0, 0, 0, 1;

    slope::Slope model;

    model.setPathLength(1);

    REQUIRE_NOTHROW(model.path(x, y));
  }
}

TEST_CASE("Guassian parallel", "[quadratic]")
{
  using namespace Catch::Matchers;

  auto data = generateData(1000, 20);

  slope::Slope model;

  slope::Threads::set(1);
  auto fit_par = model.fit(data.x, data.y);

  slope::Threads::set(2);
  auto fit_seq = model.fit(data.x, data.y);

  Eigen::VectorXd coefs_par = fit_par.getCoefs();
  Eigen::VectorXd coefs_seq = fit_seq.getCoefs();

  REQUIRE_THAT(coefs_par, VectorApproxEqual(coefs_seq));
}
