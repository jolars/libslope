#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>

TEST_CASE("Printing", "[path][gaussian]")
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

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(2);

  slope::Slope model;

  model.setPrintLevel(3);

  REQUIRE_NOTHROW(model.fit(x, y));
}
