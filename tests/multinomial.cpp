#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>

TEST_CASE("Multinomial, unpenalized", "[multinomial]")
{
  Eigen::MatrixXd x(20, 2);

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

  Eigen::VectorXd y(20);
  y << 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1;

  Eigen::MatrixXd expected_coef(2, 3);
  Eigen::VectorXd expected_intercept(2);
  Eigen::VectorXd lambda(6);

  slope::Slope model;

  model.setLoss("multinomial");
  model.setSolver("pgd");
  model.setMaxIterations(1000);
  model.setTol(1e-6);

  SECTION("No intercept")
  {
    // clang-format off
    expected_coef << 0.1094486, 0.0000000, -0.2235200,
                     0.0000000, 0.3097892, -0.4369607;
    // clang-format on

    // Fit the model
    model.setIntercept(false);
    model.setNormalization("none");
    model.setDiagnostics(true);

    double alpha = 0;
    lambda << 6.0, 5.0, 4.0, 3.0, 2.0, 1.0;

    auto fit = model.fit(x, y, alpha, lambda);

    // Get coefficients
    Eigen::MatrixXd coef = fit.getCoefs();

    // Normalize hack to make comparison with glmnet output correct
    coef.row(0).array() -= coef(0, 1);
    coef.row(1).array() -= coef(1, 0);

    // Compare coefficients with expected values
    REQUIRE_THAT(coef.reshaped(),
                 VectorApproxEqual(expected_coef.reshaped(), 1e-4));

    REQUIRE(fit.getGaps().back() <= 1e-6);
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
  }
}
