#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>

TEST_CASE("Simple low-dimensional design", "[gaussian][basic]")
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

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Ones(2);

  alpha(0) = 1e-12;

  slope::Slope model;

  model.setIntercept(false);
  model.setStandardize(false);

  model.fit(x, y, alpha, lambda);

  std::vector<Eigen::SparseMatrix<double>> coefs = model.getCoefs();
  auto dual_gaps = model.getDualGaps();
  auto primals = model.getPrimals();

  Eigen::VectorXd coef = coefs.front();

  REQUIRE_THAT(coef, VectorApproxEqual(beta, 1e-4));

  double gap = dual_gaps.front().back();
  double primal = primals.front().back();

  REQUIRE(gap <= (primal + 1e-10) * 1e-4);
}

TEST_CASE("X is identity", "[gaussian][identity]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(4, 4);
  Eigen::VectorXd y(4);
  y << 8, 6, 4, 2;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Ones(1);
  Eigen::ArrayXd lambda(4);
  lambda << 1, 0.75, 0.5, 0.25;

  slope::Slope model;
  model.setIntercept(false);
  model.setStandardize(false);
  model.setPrintLevel(3);
  model.fit(x, y, alpha, lambda);

  double gap = model.getDualGaps()[0].back();
  double primal = model.getPrimals()[0].back();

  Eigen::VectorXd coefs = model.getCoefs().front();

  std::array<double, 4> expected = { 4.0, 3.0, 2.0, 1.0 };

  REQUIRE_THAT(coefs, VectorApproxEqual(expected));
  REQUIRE(gap < primal * 1e-4);
}

TEST_CASE("Automatic lambda and alpha", "[gaussian]")
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

  REQUIRE_NOTHROW(model.fit(x, y));
}

TEST_CASE("Gaussian models", "[gaussian]")
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

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  alpha[0] = 0.05;

  lambda << 3.0, 2.0, 2.0;

  Eigen::Vector3d coef_target;

  slope::Slope model;

  model.setTol(1e-8);
  model.setObjective("gaussian");

  SECTION("No intercept, no standardization")
  {
    model.setStandardize(false);
    model.setIntercept(false);

    coef_target << 0.6864545, -0.6864545, 0.0000000;

    model.setSolver("fista");
    model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();

    model.setSolver("hybrid");
    model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_hybrid = model.getCoefs().front();

    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));

    auto dual_gaps = model.getDualGaps().front();

    REQUIRE(dual_gaps.back() >= 0);
    REQUIRE(dual_gaps.back() <= 1e-4);
  }

  SECTION("No intercept, with standardization")
  {
    model.setStandardize(true);
    model.setIntercept(false);

    coef_target << 0.700657772, -0.730587233, 0.008997323;

    model.setSolver("pgd");
    model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();

    model.setSolver("hybrid");
    model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_hybrid = model.getCoefs().front();

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("With intercept, with standardization")
  {
    model.setStandardize(true);
    model.setIntercept(true);

    coef_target << 0.700657772, -0.730587234, 0.008997323;
    std::vector<double> intercept_target = { 0.040584733 };

    model.setSolver("hybrid");
    model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs = model.getCoefs().front();
    Eigen::VectorXd intercept = model.getIntercepts().front();

    model.setSolver("pgd");
    model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();
    Eigen::VectorXd intercept_pgd = model.getIntercepts().front();

    REQUIRE_THAT(intercept, VectorApproxEqual(intercept_target, 1e-3));
    REQUIRE_THAT(intercept_pgd, VectorApproxEqual(intercept_target, 1e-3));

    REQUIRE_THAT(coefs, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("With intercept, no standardization")
  {
    model.setStandardize(false);
    model.setIntercept(true);

    coef_target << 0.68614138, -0.68614138, 0.00000000;

    model.setSolver("hybrid");
    model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs = model.getCoefs().front();
    double intercept = model.getIntercepts().front()[0];

    model.setSolver("pgd");
    model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();
    double intercept_pgd = model.getIntercepts().front()[0];

    REQUIRE_THAT(intercept, WithinAbs(0.04148455, 1e-3));
    REQUIRE_THAT(intercept_pgd, WithinAbs(0.04148455, 1e-3));

    REQUIRE_THAT(coefs, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
  }
}
