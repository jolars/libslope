#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>

TEST_CASE("Path fitting", "[path][gaussian]")
{
  using namespace Catch::Matchers;

  Eigen::Matrix<double, 3, 2> x;
  Eigen::Vector2d beta;
  Eigen::Vector3d y;

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

    slope::SlopeParameters params;
    params.objective = "gaussian";

    auto res = slope::slope(x, y, alpha, lambda, params);

    Eigen::VectorXd coef = res.betas.col(2);
    std::vector<double> coef_true = { 0.4487011, 0.6207310 };

    REQUIRE_THAT(coef, VectorApproxEqual(coef_true, 1e-4));
  }

  SECTION("Automatic alpha")
  {
    Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(0);
    Eigen::ArrayXd lambda(2);
    lambda << 1.959964, 1.644854;

    slope::SlopeParameters params;
    params.objective = "gaussian";
    params.path_length = 20;

    auto fit = slope::slope(x, y, alpha, lambda, params);

    Eigen::MatrixXd coefs = fit.betas;
    alpha = fit.alpha;

    REQUIRE(alpha.rows() == 20);

    // First step should be the null model
    std::vector<double> coef_true = { 0, 0 };
    REQUIRE_THAT(coefs.col(0), VectorApproxEqual(coef_true, 1e-5));
    REQUIRE_THAT(alpha(0), WithinAbs(0.41658754, 1e-5));
    REQUIRE_THAT(alpha(1), WithinAbs(0.25655469, 1e-5));

    coef_true = { 0.4487011, 0.6207310 };
    REQUIRE_THAT(coefs.col(2), VectorApproxEqual(coef_true, 1e-5));
  }
}
