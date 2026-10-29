#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>
#include <slope/slope.h>

TEST_CASE("Path fitting", "[path][quadratic][alpha]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(3, 2);
  Eigen::Vector2d beta;
  Eigen::VectorXd y(3);

  // clang-format off
  x << 1, 2,
       0, 1,
       1, 0;
  // clang-format on
  beta << 1, 1;

  y = x * beta;

  SECTION("Single alpha")
  {
    Eigen::ArrayXd alpha(1);
    alpha << 0.41658754;
    Eigen::ArrayXd lambda(2);
    lambda << 1.959964, 1.644854;

    slope::Slope model;
    model.setLoss("quadratic");

    auto fit = model.path(x, y, alpha, lambda);

    REQUIRE(fit.getAlpha().size() == 1);
  }

  SECTION("Fixed alpha and lambda")
  {
    Eigen::ArrayXd alpha(7);
    alpha << 0.41658754, 0.25655469, 0.15799875, 0.09730325, 0.05992403,
      0.03690411, 0.02272733;
    Eigen::ArrayXd lambda(2);
    lambda << 1.959964, 1.644854;

    slope::Slope model;
    model.setLoss("quadratic");

    auto fit = model.path(x, y, alpha, lambda);

    Eigen::VectorXd coef = fit.getCoefs()[2];
    std::vector<double> coef_true = { 0.4487011, 0.6207310 };

    REQUIRE_THAT(coef, VectorApproxEqual(coef_true, 1e-4));
    REQUIRE(fit.getAlpha().size() == 7);
  }

  SECTION("Automatic alpha")
  {
    Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(0);
    Eigen::ArrayXd lambda(2);
    lambda << 1.959964, 1.644854;

    slope::Slope model;
    model.setPathLength(20);
    auto fit = model.path(x, y, alpha, lambda);

    auto coefs = fit.getCoefs();
    alpha = fit.getAlpha();

    // First step should be the null model
    std::vector<double> coef_true = { 0, 0 };
    Eigen::VectorXd coef = coefs[0];
    REQUIRE_THAT(coef, VectorApproxEqual(coef_true, 1e-5));
    REQUIRE_THAT(alpha(0), WithinAbs(0.41658754, 1e-5));
    REQUIRE_THAT(alpha(1), WithinAbs(0.25655469, 1e-5));

    coef_true = { 0.4487011, 0.6207310 };
    coef = coefs[2];
    REQUIRE_THAT(coef, VectorApproxEqual(coef_true, 1e-5));

    fit = model.path(x, y, alpha);

    // When alpha is supplied manually, the returned alpha should be unchanged
    REQUIRE(fit.getAlpha().size() == alpha.size());
    REQUIRE(fit.getLambda().size() == 2);

    Eigen::ArrayXd alpha_manual =
      Eigen::ArrayXd::LinSpaced(10, 0.01, 1.0).reverse();
    fit = model.path(x, y, alpha_manual);

    REQUIRE(fit.getAlpha().size() == 10);
  }

  SECTION("Early stopping")
  {
    slope::Slope model;
    model.setPathLength(100);

    auto data = generateData(100, 200);

    model.setDevRatioTol(0.99);

    auto path = model.path(data.x, data.y);

    auto null_deviance = path.getNullDeviance();
    auto deviances = path.getDeviance();
    auto path_length = deviances.size();

    REQUIRE(null_deviance >= 0);
    REQUIRE(deviances.size() > 0);
    REQUIRE(deviances.size() < 100);
    REQUIRE_THAT(deviances, VectorMonotonic(false, true));

    model.setDevRatioTol(0.9);

    auto fit = model.path(data.x, data.y);

    REQUIRE(fit.getDeviance().size() < path_length);

    path_length = fit.getDeviance().size();

    model.setDevChangeTol(0.1);
    fit = model.path(data.x, data.y);

    REQUIRE(fit.getDeviance().size() < path_length);
  }

  SECTION("Sanity checks")
  {
    slope::Slope model;
    model.setPathLength(100);

    auto data = generateData(100, 10);

    Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
    alpha << 1000000;
    Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(0);

    REQUIRE_NOTHROW(model.path(data.x, data.y, alpha, lambda));
  }

  SECTION("Return clusters")
  {
    slope::Slope model;
    model.setPathLength(84);
    model.setReturnClusters(true);

    auto data = generateData(100, 10);

    auto path = model.path(data.x, data.y);

    REQUIRE(path.getClusters().size() > 0);
  }
}

TEST_CASE("Relative alphas match absolute penalties", "[path][alpha_scale]")
{
  using Catch::Matchers::WithinAbs;

  for (const std::string family :
       { "quadratic", "logistic", "poisson", "multinomial" }) {
    DYNAMIC_SECTION(family)
    {
      auto data = generateData(40, 4, family, 3, 0.8, 0.5, 129);
      Eigen::SparseMatrix<double> sparse_x = data.x.sparseView();
      Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(40, 0, 3);
      const int n_coefs = family == "multinomial" ? 8 : 4;
      const Eigen::ArrayXd lambda = Eigen::ArrayXd::LinSpaced(n_coefs, 3, 1);
      slope::Slope model;
      model.setLoss(family);
      model.setTol(1e-9);
      model.setPathLength(1);
      const auto null_path = model.path(
        data.x, data.y, {}, lambda, slope::defaultInterruptChecker, weights);
      const double alpha_max = null_path(0).getAlpha();
      REQUIRE(alpha_max > 0);

      Eigen::ArrayXd fractions(4);
      fractions << 1.5, 1, 0.3, 0.1;
      const Eigen::ArrayXd alphas = fractions * alpha_max;
      const auto absolute = model.path(data.x,
                                       data.y,
                                       alphas,
                                       lambda,
                                       slope::defaultInterruptChecker,
                                       weights);
      model.setAlphaScale("relative");
      const auto relative = model.path(data.x,
                                       data.y,
                                       fractions,
                                       lambda,
                                       slope::defaultInterruptChecker,
                                       weights);
      const auto sparse = model.path(sparse_x,
                                     data.y,
                                     fractions,
                                     lambda,
                                     slope::defaultInterruptChecker,
                                     weights);
      REQUIRE(relative.size() == fractions.size());
      REQUIRE_THAT(relative.getAlpha(), VectorApproxEqual(alphas, 1e-12));
      REQUIRE_THAT(sparse.getAlpha(), VectorApproxEqual(alphas, 1e-12));
      REQUIRE(relative(0).getCoefs().norm() < 1e-8);
      REQUIRE(relative(1).getCoefs().norm() < 1e-8);
      REQUIRE(relative(2).getCoefs().norm() > 1e-6);
      for (int j = 0; j < fractions.size(); ++j) {
        REQUIRE((Eigen::MatrixXd(relative(j).getCoefs()) -
                 Eigen::MatrixXd(absolute(j).getCoefs()))
                  .norm() < 1e-7);
        REQUIRE(
          (relative(j).getIntercepts() - absolute(j).getIntercepts()).norm() <
          1e-7);
        REQUIRE((Eigen::MatrixXd(sparse(j).getCoefs()) -
                 Eigen::MatrixXd(relative(j).getCoefs()))
                  .norm() < 1e-5);
      }

      const auto fit = model.fit(
        data.x, data.y, 0.3, lambda, slope::defaultInterruptChecker, weights);
      REQUIRE_THAT(fit.getAlpha(), WithinAbs(0.3 * alpha_max, 1e-12));
      REQUIRE((Eigen::MatrixXd(fit.getCoefs()) -
               Eigen::MatrixXd(absolute(2).getCoefs()))
                .norm() < 1e-5);
    }
  }
}

TEST_CASE("Relative alpha scale preserves automatic paths",
          "[path][alpha_scale]")
{
  auto data = generateData(40, 4, "quadratic", 1, 0.8, 0.5, 129);
  slope::Slope model;
  model.setPathLength(6);
  const auto absolute = model.path(data.x, data.y);
  model.setAlphaScale("relative");
  const auto relative = model.path(data.x, data.y);
  REQUIRE_THAT(relative.getAlpha(),
               VectorApproxEqual(absolute.getAlpha(), 1e-12));
  for (int j = 0; j < absolute.size(); ++j) {
    REQUIRE((Eigen::MatrixXd(relative(j).getCoefs()) -
             Eigen::MatrixXd(absolute(j).getCoefs()))
              .norm() < 1e-10);
  }
  const auto null_fit = model.fit(data.x, data.y);
  REQUIRE_THAT(null_fit.getAlpha(),
               Catch::Matchers::WithinAbs(absolute(0).getAlpha(), 1e-12));
  REQUIRE(null_fit.getCoefs().norm() < 1e-8);

  model.setAlphaScale("absolute");
  const auto replay = model.path(data.x, data.y, relative.getAlpha());
  REQUIRE_THAT(replay.getAlpha(),
               VectorApproxEqual(absolute.getAlpha(), 1e-12));
}

TEST_CASE("Relative alpha handles zero penalties", "[path][alpha_scale]")
{
  using Catch::Matchers::WithinAbs;
  Eigen::MatrixXd x(4, 1);
  x << -3, -1, 1, 3;
  Eigen::VectorXd y = 2 * x.col(0).array() + 4;
  slope::Slope model;
  model.setAlphaScale("relative");
  model.setTol(1e-10);
  SECTION("Zero fraction fits the unpenalized model")
  {
    const auto fit = model.fit(x, y, 0);
    REQUIRE(fit.getAlpha() == 0);
    REQUIRE_THAT(fit.getCoefs().coeff(0, 0), WithinAbs(2, 1e-8));
    REQUIRE_THAT(fit.getIntercepts()(0), WithinAbs(4, 1e-8));
  }
  SECTION("Zero alpha max produces finite zero penalties")
  {
    y.setConstant(4);
    Eigen::ArrayXd fractions(3);
    fractions << 1, 0.5, 0;
    const auto path = model.path(x, y, fractions);
    REQUIRE(path.size() == fractions.size());
    REQUIRE(path.getAlpha().isZero());
    for (int j = 0; j < path.size(); ++j) {
      REQUIRE(path(j).getCoefs().norm() == 0);
      REQUIRE_THAT(path(j).getIntercepts()(0), WithinAbs(4, 1e-8));
    }
  }
  SECTION("Zero lambda leaves the model unpenalized")
  {
    const auto fit = model.fit(x, y, 1, Eigen::ArrayXd::Zero(1));
    REQUIRE(fit.getAlpha() == 0);
    REQUIRE_THAT(fit.getCoefs().coeff(0, 0), WithinAbs(2, 1e-8));
  }
}

TEST_CASE("Relative alpha validates inputs before scaling",
          "[path][alpha_scale][input_validation]")
{
  Eigen::MatrixXd x(4, 1);
  x << -3, -1, 1, 3;
  Eigen::VectorXd y = Eigen::VectorXd::Constant(4, 2);
  slope::Slope model;
  REQUIRE_THROWS_AS(model.setAlphaScale("fraction"), std::invalid_argument);
  model.setAlphaScale("relative");
  for (const double invalid : { -0.1,
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::infinity() }) {
    REQUIRE_THROWS_AS(model.fit(x, y, invalid), std::invalid_argument);
    Eigen::ArrayXd fractions(2);
    fractions << 1, invalid;
    REQUIRE_THROWS_AS(model.path(x, y, fractions), std::invalid_argument);
  }
  y = 10 * x.col(0);
  REQUIRE_THROWS_AS(model.fit(x, y, std::numeric_limits<double>::max()),
                    std::invalid_argument);
}
