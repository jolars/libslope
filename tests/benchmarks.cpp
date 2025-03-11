#include "generate_data.hpp"
#include <Eigen/Core>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <slope/cv.h>
#include <slope/math.h>
#include <slope/slope.h>
#include <slope/threads.h>

TEST_CASE("Parallelized gradient computations", "[!benchmark][parallelization]")
{
  int n = 1000;
  int p = 1000;

  Eigen::VectorXd gradient(p);
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

TEST_CASE("Path screening benchmarks", "[!benchmark][screening]")
{
  const int p = 1000;
  const int n = 100;

  auto data = generateData(n, p, "quadratic", 1, 1, 0.01);

  slope::Slope model;

  model.setSolver("fista");

  BENCHMARK("Strong rule screening")
  {
    model.setScreening("strong");
    model.path(data.x, data.y);
  };

  BENCHMARK("No screening")
  {
    model.setScreening("none");
    model.path(data.x, data.y);
  };
}

TEST_CASE("One lambda screening benchmarks", "[!benchmark][screening]")
{
  const int p = 1000;
  const int n = 100;

  auto data = generateData(n, p, "quadratic", 1, 1, 0.01);

  slope::Slope model;

  model.setSolver("fista");

  double alpha = 0.1;

  BENCHMARK("Strong rule screening")
  {
    model.setScreening("strong");
    model.fit(data.x, data.y, alpha);
  };

  BENCHMARK("No screening")
  {
    model.setScreening("none");
    model.fit(data.x, data.y, alpha);
  };
}

TEST_CASE("Parallel cross-validation", "[!benchmark][cv][parallelization]")
{
  const int p = 100;
  const int n = 1000;

  auto data = generateData(n, p, "quadratic");

  slope::Slope model;

  BENCHMARK("Sequential")
  {
    slope::Threads::set(1);
    crossValidate(model, data.x, data.y);
  };

  BENCHMARK("Parallel")
  {
    slope::Threads::set(4);
    crossValidate(model, data.x, data.y);
  };
}

TEST_CASE("Cluster updating", "[!benchmark][clusters]")
{
  const int p = 10000;
  const int n = 100;

  auto data = generateData(n, p, "quadratic");

  slope::Slope model;

  model.setUpdateClusters(true);

  BENCHMARK("With cluster updates")
  {
    model.path(data.x, data.y);
  };

  model.setUpdateClusters(false);

  BENCHMARK("Without cluster updates")
  {
    model.path(data.x, data.y);
  };
}
