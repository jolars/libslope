#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/losses/quadratic.h>
#include <slope/ols.h>
#include <slope/slope.h>

TEST_CASE("Relaxed quadratic fits", "[relax][quadratic]")
{
  using Catch::Matchers::WithinRel;

  slope::Slope model;
  slope::SlopeFit fit;

  // model.setNormalization("none");

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

  Eigen::ArrayXd lambda = Eigen::ArrayXd::Ones(2);

  SECTION("Relaxed fit is OLS on full set")
  {
    double alpha = 1e-6;

    fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coef0 = fit.getCoefs();

    // Assert that the coefficients are non-zero and different
    REQUIRE(coef0[0] > 0);
    REQUIRE(coef0[1] > 0);
    REQUIRE(coef0[1] != coef0[0]);

    Eigen::VectorXd bb(2);
    // bb << 0.37416793521666464, 1.7307684486543986;
    bb << 1, 2;
    double bb0 = -0.0000020240555609711093;

    Eigen::VectorXd eta = (x * bb).array() + bb0;

    Eigen::VectorXd grad = x.transpose() * (eta - y) / 3;
    // double deviance = 0.5 * (y - rr).squaredNorm() / 3;

    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-4));

    auto relaxed_fit = model.relax(fit, x, y);

    Eigen::VectorXd coef = relaxed_fit.getCoefs();

    std::vector<double> coef_target = { 1, 2 };

    REQUIRE_THAT(coef, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("Second predictor selected")
  {
    double alpha = 0.7;

    fit = model.fit(x, y, alpha, lambda);

    auto relaxed_fit = model.relax(fit, x, y);

    auto [beta0_ols, beta_ols] = fitOls(slope::subsetCols(x, { 1 }), y);

    Eigen::VectorXd coef = relaxed_fit.getCoefs();
    std::vector<double> coef_target = { 0, 2.1 };

    REQUIRE_THAT(coef[1], WithinRel(beta_ols[0], 1e-4));
  }

  SECTION("Scaling")
  {
    double alpha = 0.7;

    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coef0 = fit.getCoefs();

    double gamma = 0.25;

    auto relaxed_fit = model.relax(fit, x, y, gamma);
    Eigen::VectorXd coef = relaxed_fit.getCoefs();

    auto [beta0_ols, beta_ols] = fitOls(slope::subsetCols(x, { 1 }), y);

    Eigen::VectorXd full_coefs = model.relax(fit, x, y, 0).getCoefs();

    double coef_target = (1 - gamma) * beta_ols[0] + gamma * coef0[1];

    REQUIRE_THAT(full_coefs[1], WithinRel(beta_ols[0], 1e-4));

    REQUIRE(coef[0] == 0);
    REQUIRE_THAT(coef[1], WithinRel(coef_target, 1e-4));
  }

  SECTION("Clustered solution")
  {
    double alpha = 0.12;

    int n = 4;
    int p = 3;

    Eigen::MatrixXd x(n, p);
    Eigen::VectorXd beta(p);

    // model.setNormalization("none");

    // clang-format off
    x << 1.1, 0.3, 0.2,
         0.2, 0.9, 1.1,
         0.2, 2.5, 0.5,
         0.5, 0.0, 0.2;
    // clang-format on

    beta << 2.05, 0, 2;

    Eigen::VectorXd y = x * beta;

    fit = model.fit(x, y, alpha);
    Eigen::VectorXd coef_reg = fit.getCoefs();

    // Check that there is the expected cluster
    REQUIRE(coef_reg[0] > 0);
    REQUIRE(coef_reg[0] == coef_reg[2]);

    double gamma = 0;

    auto relaxed_fit = model.relax(fit, x, y, gamma);

    Eigen::SparseMatrix<double> U =
      fit.getClusters().patternMatrix().cast<double>();

    REQUIRE(U.rows() == p);
    REQUIRE(U.cols() == 2);

    Eigen::MatrixXd x_collapsed = x * U;

    auto [beta0_ols, beta_ols] = fitOls(x_collapsed, y);

    Eigen::VectorXd coef = relaxed_fit.getCoefs();

    REQUIRE_THAT(coef[0], WithinRel(beta_ols[0], 1e-4));
    REQUIRE_THAT(coef[2], WithinRel(beta_ols[0], 1e-4));
  }
}
