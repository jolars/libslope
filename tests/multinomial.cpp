#include "slope/losses/multinomial.h"
#include "generate_data.hpp"
#include "load_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <slope/math.h>
#include <slope/slope.h>
#include <slope/sorted_l1_norm.h>

TEST_CASE("Multinomial, unpenalized", "[multinomial]")
{

  int n = 20;
  int p = 2;
  int m = 2;

  Eigen::MatrixXd x(n, p);

  // clang-format off
  x <<  1.2, -0.3,
       -0.5,  0.7,
        0.8, -1.2,
       -1.1,  0.4,
        0.3, -0.8,
        1.5,  0.2,
       -0.2, -0.5,
        0.7,  1.1,
       -0.9, -0.9,
        0.4,  0.6,
        0.1, -1.0,
       -1.3,  0.3,
        0.6, -0.7,
       -0.7,  0.8,
        1.1, -0.4,
       -0.4,  1.3,
        0.9, -0.6,
       -1.0,  0.5,
        0.5, -1.1,
       -0.8,  0.9;
  // clang-format on

  Eigen::VectorXd y(n);
  y << 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1;

  Eigen::MatrixXd expected_coef(p, m);
  Eigen::VectorXd expected_intercept(m);
  Eigen::VectorXd lambda(p * m);

  slope::Slope model;

  model.setLoss("multinomial");
  model.setSolver("hybrid");
  model.setNormalization("none");
  model.setMaxIterations(2000);
  model.setUpdateClusters(true);
  model.setTol(1e-8);

  SECTION("No regularization, no intercept")
  {
    expected_coef << 0.3329687, 0.2235199, 0.4369607, 0.7467499;

    // Fit the model
    model.setIntercept(false);
    model.setDiagnostics(true);

    double alpha = 0;
    lambda << 6.0, 5.0, 4.0, 3.0;

    auto fit = model.fit(x, y, alpha, lambda);

    // Get coefficients
    Eigen::MatrixXd coef = fit.getCoefs();

    REQUIRE(coef.rows() == p);
    REQUIRE(coef.cols() == m);

    REQUIRE(!slope::WarningLogger::hasWarnings());

    // Compare coefficients with expected values
    REQUIRE_THAT(coef.reshaped(),
                 VectorApproxEqual(expected_coef.reshaped(), 1e-4));

    REQUIRE(fit.getGaps().back() <= 1e-6);
  }

  SECTION("Regularization, no intercept")
  {
    expected_coef << 0.09260631, 0.0000000, 0.09260631, 0.3757873;

    // Fit the model
    model.setIntercept(false);

    double alpha = 0.005;
    lambda << 4, 3, 2, 1;

    auto fit = model.fit(x, y, alpha, lambda);

    // Get coefficients
    Eigen::MatrixXd coef = fit.getCoefs();

    REQUIRE(!slope::WarningLogger::hasWarnings());

    // Compare coefficients with expected values
    REQUIRE_THAT(coef.reshaped(),
                 VectorApproxEqual(expected_coef.reshaped(), 1e-4));
  }

  SECTION("Regularization, intercept")
  {
    expected_coef << 0.2310035, 0.1333503, 0.3364695, 0.6209708;
    expected_intercept << 0.1775123, 0.1730744;

    model.setIntercept(true);
    model.setDiagnostics(true);

    double alpha = 0.002;
    lambda << 4, 3, 2, 1;

    auto fit = model.fit(x, y, alpha, lambda);

    // Get coefficients
    Eigen::MatrixXd coef = fit.getCoefs();
    Eigen::MatrixXd intercept = fit.getIntercepts();
    auto gaps = fit.getGaps();

    REQUIRE(!slope::WarningLogger::hasWarnings());
    REQUIRE(std::all_of(gaps.begin(), gaps.end(), [](const double gap) {
      return std::isfinite(gap) && gap >= -1e-12;
    }));
    REQUIRE(gaps.back() <= 1e-6);

    // Compare coefficients with expected values
    REQUIRE_THAT(intercept.reshaped(),
                 VectorApproxEqual(expected_intercept.reshaped(), 1e-4));
  }

  SECTION("Path")
  {
    auto data = generateData(200, 20, "multinomial", 3, 0.4, 0.5, 93);

    model.setTol(1e-4);
    auto fit = model.path(data.x, data.y);

    auto null_deviance = fit.getNullDeviance();
    auto deviances = fit.getDeviance();

    REQUIRE(null_deviance >= 0);
    REQUIRE(deviances.size() > 0);
    REQUIRE(deviances.size() < 100);
    REQUIRE_THAT(deviances, VectorMonotonic(false, true));

    REQUIRE(!slope::WarningLogger::hasWarnings());
  }
}

TEST_CASE("Multinomial wine data", "[multinomial]")
{
  auto [x, y] = loadData("tests/data/wine.csv");

  slope::Slope model;

  model.setLoss("multinomial");

  // TODO: Hybrid solver has convergence issues with this data
  model.setSolver("pgd");

  auto path = model.path(x, y);

  REQUIRE(path.getDeviance().back() > 0);
  REQUIRE(path.getDeviance().size() > 5);
}

TEST_CASE("Multinomial predictions", "[multinomial][predict]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  int p = 3;
  int m = 3;

  Eigen::MatrixXd x(n, p);
  Eigen::MatrixXd beta(p, m - 1);
  Eigen::MatrixXd eta(n, m - 1);

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

  beta <<  1,   2,
          -1,   2,
          -0.1, 0;
  // clang-format on

  eta = x * beta;

  slope::Multinomial loss;

  auto pred = loss.predict(eta);

  std::array<double, n> expected = { 1, 1, 1, 2, 1, 1, 0, 0, 2, 1 };

  REQUIRE_THAT(pred.reshaped(), VectorApproxEqual(expected));
  REQUIRE(!slope::WarningLogger::hasWarnings());
}

TEST_CASE("Multinomial dual points remain feasible with an intercept",
          "[multinomial][dual]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd eta(6, 3);
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(6, 3);
  eta << 31.485260345944724, -0.94707176117017333, 0.70905200876054053, -2.0,
    4.0, 0.5, 1.5, -3.0, 2.5, 0.0, 0.0, 0.0, -8.0, 1.0, 6.0, 2.0, -5.0, -1.0;
  y(0, 0) = 1.0;
  y(1, 1) = 1.0;
  y(2, 2) = 1.0;
  y(4, 0) = 1.0;

  slope::Multinomial loss;
  Eigen::MatrixXd theta = loss.dualPoint(eta, y, true);
  Eigen::ArrayXXd means = theta.array() + y.array();

  for (Eigen::Index k = 0; k < theta.cols(); ++k) {
    REQUIRE_THAT(theta.col(k).sum(), WithinAbs(0.0, 1e-12));
  }
  REQUIRE((means >= 0.0).all());
  REQUIRE((means.rowwise().sum() <= 1.0).all());
  REQUIRE(std::isfinite(loss.dual(theta, y, Eigen::VectorXd::Ones(y.rows()))));
}

TEST_CASE(
  "Multinomial dual points handle absent classes and simplex boundaries",
  "[multinomial][dual]")
{
  using namespace Catch::Matchers;

  slope::Multinomial loss;
  Eigen::MatrixXd eta(4, 3);
  eta << 20.0, -20.0, 3.0, -4.0, 12.0, -8.0, 2.0, 1.0, -6.0, -10.0, -3.0, 15.0;

  SECTION("A non-reference class is absent")
  {
    Eigen::MatrixXd y = Eigen::MatrixXd::Zero(4, 3);
    y(0, 0) = 1.0;
    y(1, 1) = 1.0;
    y(2, 0) = 1.0;

    Eigen::MatrixXd theta = loss.dualPoint(eta, y, true);
    Eigen::ArrayXXd means = theta.array() + y.array();

    REQUIRE_THAT(theta.col(2).sum(), WithinAbs(0.0, 1e-12));
    REQUIRE((means.col(2) == 0.0).all());
    REQUIRE((means >= 0.0).all());
    REQUIRE((means.rowwise().sum() <= 1.0).all());
    REQUIRE(
      std::isfinite(loss.dual(theta, y, Eigen::VectorXd::Ones(y.rows()))));
  }

  SECTION("The reference class is absent")
  {
    Eigen::MatrixXd y = Eigen::MatrixXd::Zero(4, 3);
    y(0, 0) = 1.0;
    y(1, 1) = 1.0;
    y(2, 2) = 1.0;
    y(3, 0) = 1.0;

    Eigen::MatrixXd theta = loss.dualPoint(eta, y, true);
    Eigen::ArrayXXd means = theta.array() + y.array();

    for (Eigen::Index k = 0; k < theta.cols(); ++k) {
      REQUIRE_THAT(theta.col(k).sum(), WithinAbs(0.0, 1e-12));
    }
    REQUIRE((means >= 0.0).all());
    for (Eigen::Index i = 0; i < means.rows(); ++i) {
      REQUIRE_THAT(means.row(i).sum(), WithinAbs(1.0, 1e-12));
    }
    REQUIRE(
      std::isfinite(loss.dual(theta, y, Eigen::VectorXd::Ones(y.rows()))));
  }
}

TEST_CASE("Multinomial dual points satisfy the SLOPE dual constraint",
          "[multinomial][dual]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(6, 2);
  Eigen::MatrixXd eta(6, 3);
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(6, 3);
  x << 2.0, -1.0, -3.0, 0.5, 1.0, 4.0, -2.0, -3.0, 0.5, 2.0, 3.0, -2.0;
  eta << 8.0, -3.0, 1.0, -4.0, 7.0, -2.0, 2.0, -6.0, 5.0, -1.0, 1.0, 0.0, 6.0,
    2.0, -5.0, -7.0, -2.0, 9.0;
  y(0, 0) = 1.0;
  y(1, 1) = 1.0;
  y(2, 2) = 1.0;
  y(4, 0) = 1.0;

  slope::Multinomial loss;
  slope::SortedL1Norm norm;
  Eigen::MatrixXd theta = loss.dualPoint(eta, y, true);
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(x.cols() * y.cols());
  Eigen::ArrayXd lambda(gradient.size());
  lambda << 0.06, 0.05, 0.04, 0.03, 0.02, 0.01;
  slope::updateGradient(gradient,
                        x,
                        theta,
                        Eigen::VectorXd::Zero(x.cols()),
                        Eigen::VectorXd::Ones(x.cols()),
                        Eigen::VectorXd::Ones(x.rows()),
                        slope::JitNormalization::None);

  const double scale = std::max(1.0, norm.dualNorm(gradient, lambda));
  REQUIRE(scale > 1.0);
  theta.array() /= scale;
  slope::updateGradient(gradient,
                        x,
                        theta,
                        Eigen::VectorXd::Zero(x.cols()),
                        Eigen::VectorXd::Ones(x.cols()),
                        Eigen::VectorXd::Ones(x.rows()),
                        slope::JitNormalization::None);

  Eigen::ArrayXXd means = theta.array() + y.array();
  REQUIRE_THAT(norm.dualNorm(gradient, lambda), WithinAbs(1.0, 1e-12));
  for (Eigen::Index k = 0; k < theta.cols(); ++k) {
    REQUIRE_THAT(theta.col(k).sum(), WithinAbs(0.0, 1e-12));
  }
  REQUIRE((means >= 0.0).all());
  REQUIRE((means.rowwise().sum() <= 1.0).all());
  REQUIRE(std::isfinite(loss.dual(theta, y, Eigen::VectorXd::Ones(y.rows()))));
}

TEST_CASE("Multinomial dual rejects points outside the simplex",
          "[multinomial][dual]")
{
  slope::Multinomial loss;
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(1, 2);
  Eigen::MatrixXd negative(1, 2);
  Eigen::MatrixXd overfull(1, 2);
  negative << -0.1, 0.0;
  overfull << 0.8, 0.4;

  REQUIRE_THROWS_AS(loss.dual(negative, y, Eigen::VectorXd::Ones(1)),
                    std::domain_error);
  REQUIRE_THROWS_AS(loss.dual(overfull, y, Eigen::VectorXd::Ones(1)),
                    std::domain_error);
}

TEST_CASE("Multinomial alternative response types", "[multinomial][predict]")
{
  using namespace Catch::Matchers;
  using namespace slope;

  int n = 4;
  int p = 2;
  int m = 3;

  Eigen::MatrixXd x(n, p);
  Eigen::MatrixXd y_mat(n, m);
  Eigen::VectorXd y_vec(n);

  // clang-format off
  x <<  1.2, -0.3,
       -0.5,  0.7,
        0.8, -1.2,
        0.3, -0.8;
  y_mat << 1, 0, 0,
           0, 0, 1,
           0, 0, 1,
           1, 0, 0;
  y_vec << 0, 2, 2, 0;
  // clang-format on

  slope::Slope model;

  auto fit1 = model.fit(x, y_mat, 1.0);
  auto fit2 = model.fit(x, y_vec, 1.0);

  Eigen::MatrixXd coef1 = fit1.getCoefs();
  Eigen::MatrixXd coef2 = fit2.getCoefs();

  REQUIRE_THAT(coef1.reshaped(), VectorApproxEqual(coef2.reshaped()));
  REQUIRE(!slope::WarningLogger::hasWarnings());
}
