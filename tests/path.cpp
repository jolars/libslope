#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>

TEST_CASE("Path fitting", "[path][gaussian]")
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

  SECTION("Fixed alpha and lambda")
  {
    Eigen::ArrayXd alpha(7);
    alpha << 0.41658754, 0.25655469, 0.15799875, 0.09730325, 0.05992403,
      0.03690411, 0.02272733;
    Eigen::ArrayXd lambda(2);
    lambda << 1.959964, 1.644854;

    slope::Slope model;
    model.setObjective("gaussian");

    model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coef = model.getCoefs()[2];
    std::vector<double> coef_true = { 0.4487011, 0.6207310 };

    REQUIRE_THAT(coef, VectorApproxEqual(coef_true, 1e-4));
  }

  SECTION("Automatic alpha")
  {
    Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(0);
    Eigen::ArrayXd lambda(2);
    lambda << 1.959964, 1.644854;

    slope::Slope model;
    model.setPathLength(20);
    model.fit(x, y, alpha, lambda);

    auto coefs = model.getCoefs();
    alpha = model.getAlpha();

    REQUIRE(alpha.rows() == 20);

    // First step should be the null model
    std::vector<double> coef_true = { 0, 0 };
    Eigen::VectorXd coef = coefs[0];
    REQUIRE_THAT(coef, VectorApproxEqual(coef_true, 1e-5));
    REQUIRE_THAT(alpha(0), WithinAbs(0.41658754, 1e-5));
    REQUIRE_THAT(alpha(1), WithinAbs(0.25655469, 1e-5));

    coef_true = { 0.4487011, 0.6207310 };
    coef = coefs[2];
    REQUIRE_THAT(coef, VectorApproxEqual(coef_true, 1e-5));
  }

  SECTION("Early stopping")
  {
    slope::Slope model;
    model.setPathLength(100);
    model.setPrintLevel(1);

    auto data = generateData(100, 200);

    model.fit(data.x, data.y);

    auto null_deviance = model.getNullDeviance();
    auto deviances = model.getDeviances();

    int path_length = deviances.size();

    REQUIRE(null_deviance >= 0);
    REQUIRE(deviances.size() > 0);
    REQUIRE(deviances.size() < 100);
    REQUIRE_THAT(deviances, VectorMonotonic(false, true));

    model.setDevRatioTol(0.99);
    model.fit(data.x, data.y);

    REQUIRE(model.getDeviances().size() < path_length);

    path_length = model.getDeviances().size();

    model.setDevChangeTol(0.1);
    model.fit(data.x, data.y);

    REQUIRE(model.getDeviances().size() < path_length);
  }
}
