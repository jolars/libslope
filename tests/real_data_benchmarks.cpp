#include "load_libsvm.hpp"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <slope/slope.h>
#include <slope/threads.h>
#include <stdexcept>
#include <string>

TEST_CASE("High-dimensional solver paths", "[!benchmark][high_dimensional]")
{
  const char* dataset_value = std::getenv("SLOPE_BENCHMARK_DATASET");
  const char* path_value = std::getenv("SLOPE_BENCHMARK_DATA_PATH");

  if (dataset_value == nullptr || path_value == nullptr) {
    SKIP("Set SLOPE_BENCHMARK_DATASET and SLOPE_BENCHMARK_DATA_PATH");
  }

  const std::string dataset(dataset_value);
  const bool is_rcv1 = dataset == "rcv1";
  const bool is_e2006 = dataset == "e2006";

  if (!is_rcv1 && !is_e2006) {
    throw std::invalid_argument(
      "SLOPE_BENCHMARK_DATASET must be 'rcv1' or 'e2006'");
  }

  const int n_features = is_rcv1 ? 47'236 : 150'360;
  SparseData data = loadLibsvm(path_value, n_features, is_rcv1);

  slope::Threads::set(1);

  BENCHMARK("Hybrid path on " + dataset)
  {
    slope::Slope model;
    model.setLoss(is_rcv1 ? "logistic" : "quadratic");
    model.setSolver("hybrid");
    model.setScreening("strong");
    model.setPathLength(10);
    model.setHybridCdIterations(10);
    model.setRandomSeed(0);
    model.setTol(1e-4);

    return model.path(data.x, data.y);
  };
}
