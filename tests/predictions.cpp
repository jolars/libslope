#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>

TEST_CASE("Predictions", "[predict]")
{
  using namespace Catch::Matchers;

  auto data = generateData(100, 10);

  slope::Slope model;

  auto path = model.path(data.x, data.y);

  auto fit = path(5);

  auto new_data = generateData(20, 10);

  auto pred = fit.predict(new_data.x);

  REQUIRE(pred.rows() == 20);
}
