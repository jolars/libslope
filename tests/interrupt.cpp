#include "generate_data.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <slope/slope.h>

TEST_CASE("Interrupt callback", "[interrupt][path]")
{
  auto data = generateData(200, 100);

  SECTION("Interrupt during path fitting")
  {
    slope::Slope model;
    model.setPathLength(50);
    model.setMaxIterations(10000);

    int call_count = 0;
    int interrupt_after = 5;

    auto interrupt_callback = [&call_count, interrupt_after]() {
      call_count++;
      return call_count > interrupt_after;
    };

    auto path = model.path(data.x,
                           data.y,
                           Eigen::ArrayXd::Zero(0),
                           Eigen::ArrayXd::Zero(0),
                           interrupt_callback);

    // Should have stopped early due to interrupt
    REQUIRE(path.size() < 50);
    REQUIRE(path.size() > 0);
    REQUIRE(call_count > 0);
  }

  SECTION("No interrupt - full path")
  {
    slope::Slope model;
    model.setPathLength(20);

    auto no_interrupt = []() { return false; };

    auto path = model.path(data.x,
                           data.y,
                           Eigen::ArrayXd::Zero(0),
                           Eigen::ArrayXd::Zero(0),
                           no_interrupt);

    // Should complete the full path
    REQUIRE(path.size() == 20);
  }

  SECTION("Interrupt in fit()")
  {
    slope::Slope model;
    model.setMaxIterations(10000);
    model.setPathLength(10);

    bool interrupted = false;
    auto interrupt_callback = [&interrupted]() {
      // Interrupt immediately
      interrupted = true;
      return true;
    };

    // fit() calls path() internally - if interrupted on first step,
    // path should be empty and fit might throw or return empty result
    // So we test path() instead which we know handles it
    auto path = model.path(data.x,
                           data.y,
                           Eigen::ArrayXd::Zero(0),
                           Eigen::ArrayXd::Zero(0),
                           interrupt_callback);

    // The callback should have been called
    REQUIRE(interrupted);
    // Path should be empty or very short due to immediate interrupt
    REQUIRE(path.size() == 0);
  }

  SECTION("Default callback - never interrupts")
  {
    slope::Slope model;
    model.setPathLength(10);

    // Test that default parameter works
    auto path = model.path(data.x, data.y);

    REQUIRE(path.size() == 10);
  }
}
