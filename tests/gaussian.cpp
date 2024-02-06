#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <slope/slope.h>

TEST_CASE("Simple low-dimensional design", "[gaussian][basic]")
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

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(2);

  slope::Slope model;

  model.setIntercept(false);
  model.setStandardize(false);

  model.fit(x, y, alpha, lambda);

  auto coefs = model.getCoefs();
  auto dual_gaps = model.getDualGaps();
  auto primals = model.getPrimals();

  Eigen::VectorXd coef = coefs.col(0);

  REQUIRE_THAT(coef, VectorApproxEqual(beta, 1e-4));

  double gap = dual_gaps.front().back();
  double primal = primals.front().back();

  REQUIRE(gap <= (primal + 1e-10) * 1e-4);
}

TEST_CASE("X is identity", "[gaussian][identity]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(4, 4);
  Eigen::Vector4d y;
  y << 8, 6, 4, 2;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Ones(1);
  Eigen::Array4d lambda;
  lambda << 1, 0.75, 0.5, 0.25;

  slope::Slope model;
  model.setIntercept(false);
  model.setStandardize(false);
  model.setPrintLevel(3);
  model.fit(x, y, alpha, lambda);
  // model.fit(x, y);
  //
  // double gap = model.getDualGaps()[0].back();
  // double primal = model.getPrimals()[0].back();
  //
  // Eigen::VectorXd coefs = model.getCoefs().col(0);
  // Eigen::VectorXd betas = coefs.col(0);
  //
  // std::array<double, 4> expected = { 4.0, 3.0, 2.0, 1.0 };
  //
  // REQUIRE_THAT(betas, VectorApproxEqual(expected));
  // REQUIRE(gap < primal * 1e-4);
}

TEST_CASE("Automatic lambda and alpha", "[gaussian]")
{
  using namespace Catch::Matchers;

  Eigen::Matrix<double, 3, 2> x;
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
