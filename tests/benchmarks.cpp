#include "../src/slope/math.h"
#include "../src/slope/threads.h"
#include "generate_data.hpp"
#include "slope/slope.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>

TEST_CASE("Parallelized gradient computations", "[!benchmark]")
{
  int n = 1000;
  int p = 1000;
  int m = 1;

  Eigen::MatrixXd gradient(p, m);
  std::vector<int> active_set(p);
  Eigen::VectorXd x_centers(p);
  Eigen::VectorXd x_scales(p);
  Eigen::VectorXd w(n);
  slope::JitNormalization jit_normalization = slope::JitNormalization::Both;

  auto data = generateData(n, p);

  auto x = data.x;
  auto residual = data.y;

  BENCHMARK("Gradient sequential")
  {
    slope::Threads::set(1);
    slope::updateGradient(gradient,
                          x,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          jit_normalization);
  };

  BENCHMARK("Gradient parallel")
  {
    slope::Threads::set(4);
    slope::updateGradient(gradient,
                          x,
                          residual,
                          active_set,
                          x_centers,
                          x_scales,
                          w,
                          jit_normalization);
  };
}

TEST_CASE("Screening benchmarks", "[!benchmark]")
{
  auto data = generateData(100, 10000, "quadratic", 1, 1, 0.01, 315);

  slope::Slope model;

  model.setIntercept(false);
  auto fit = model.path(data.x, data.y);
  Eigen::VectorXd coefs = fit.getCoefs().back();

  BENCHMARK("No screening")
  {
    model.setScreening("none");
    model.fit(data.x, data.y);
  };

  BENCHMARK("Strong rule screening")
  {
    model.setScreening("strong");
    model.fit(data.x, data.y);
  };
}
