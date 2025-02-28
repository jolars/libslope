#include <catch2/catch_test_macros.hpp>
#include <slope/timer.h>
#include <thread>

TEST_CASE("Timer", "[utils]")
{
  using namespace slope;

  Timer timer;

  timer.start();

  double t0 = timer.elapsed();

  std::this_thread::sleep_for(std::chrono::microseconds(10));

  timer.pause();

  double t1 = timer.elapsed();
  double t1b = timer.elapsed();

  REQUIRE(t1 == t1b);

  timer.resume();

  std::this_thread::sleep_for(std::chrono::microseconds(10));

  double t2 = timer.elapsed();

  REQUIRE(t1 > t0);
  REQUIRE(t2 > t1);
}
