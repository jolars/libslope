
#include "load_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>
#include <unistd.h>

TEST_CASE("Abalone dataset", "[realdata][poisson][quadratic]")
{
  auto [x, y] = loadData("tests/data/abalone.csv");

  slope::Slope model;

  for (const std::string loss : { "poisson", "quadratic" }) {
    DYNAMIC_SECTION("Loss: " << loss)
    {
      model.setLoss(loss);

      auto solver = loss == "poisson" ? "fista" : "hybrid";

      model.setSolver(solver);

      REQUIRE_NOTHROW(model.path(x, y));
    }
  }
}
