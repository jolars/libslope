#include "generate_data.hpp"
#include <Eigen/SparseCore>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <slope/clusters.h>
#include <slope/cv.h>
#include <slope/math.h>
#include <slope/regularization_sequence.h>
#include <slope/slope.h>
#include <slope/solvers/slope_threshold.h>
#include <slope/threads.h>

TEST_CASE("Parallelized gradient computations", "[!benchmark]")
{
  int n = 10;
  int p = 10000;

  Eigen::VectorXd gradient(p);
  std::vector<int> active_set(p);
  Eigen::VectorXd x_centers(p);
  Eigen::VectorXd x_scales(p);
  Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
  slope::JitNormalization jit_normalization = slope::JitNormalization::Both;

  std::iota(active_set.begin(), active_set.end(), 0);

  auto data = generateData(n, p);

  auto x = data.x;
  auto residual = data.y;

  slope::computeCenters(x_centers, x, "mean");
  slope::computeScales(x_scales, x, "sd");

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

TEST_CASE("Linear predictor parallelization", "[!benchmark]")
{
  int n = 1000;
  int p = 10000;

  Eigen::VectorXd gradient(p);
  std::vector<int> active_set(p);
  Eigen::VectorXd x_centers(p);
  Eigen::VectorXd x_scales(p);
  Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
  slope::JitNormalization jit_normalization = slope::JitNormalization::Both;

  Eigen::VectorXd beta0 = Eigen::VectorXd::Random(1);
  Eigen::VectorXd beta = Eigen::VectorXd::Random(p);
  bool intercept = true;

  std::iota(active_set.begin(), active_set.end(), 0);

  auto data = generateData(n, p);

  auto x = data.x;
  auto residual = data.y;

  slope::computeCenters(x_centers, x, "mean");
  slope::computeScales(x_scales, x, "sd");

  BENCHMARK("Linear predictor sequential")
  {
    slope::Threads::set(1);
    linearPredictor(x,
                    active_set,
                    beta0,
                    beta,
                    x_centers,
                    x_scales,
                    jit_normalization,
                    intercept);
  };

  BENCHMARK("Linear predictor parallel")
  {
    slope::Threads::set(4);
    linearPredictor(x,
                    active_set,
                    beta0,
                    beta,
                    x_centers,
                    x_scales,
                    jit_normalization,
                    intercept);
  };
}

TEST_CASE("Path screening benchmarks", "[!benchmark]")
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

TEST_CASE("One lambda screening benchmarks", "[!benchmark]")
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

TEST_CASE("Parallel cross-validation", "[!benchmark]")
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

TEST_CASE("Benchmark cluster updating", "[!benchmark]")
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

TEST_CASE("Cluster comparison", "[!benchmark]")
{
  const int p = 100000;
  const int n = 100;

  auto data = generateData(n, p, "quadratic");
  auto beta = data.beta;

  slope::Slope model;

  // Create a more challenging beta vector with some clusters
  Eigen::VectorXd beta_clustered = Eigen::VectorXd::Random(p);
  // Create some clusters by setting coefficients equal
  for (int i = 0; i < p; i += 3) {
    double value = beta_clustered(i);
    int cluster_size = std::min(3, p - i);
    for (int j = 0; j < cluster_size; j++) {
      beta_clustered(i + j) = value;
    }
  }

  BENCHMARK("Cluster initialization")
  {
    slope::Clusters clusters(beta_clustered);
  };

  // Create instances for update benchmarks
  slope::Clusters clusters(beta_clustered);

  BENCHMARK("Clusters accessing")
  {
    // Clone to avoid modifying the original
    // Random updates (use the old API with three parameters)
    for (int j = 0; j < clusters.n_clusters(); ++j) {
      double c_old = clusters.coeff(j);

      std::vector<int> s;
      int cluster_size = clusters.cluster_size(j);
      s.reserve(cluster_size);

      for (auto c_it = clusters.cbegin(j); c_it != clusters.cend(j); ++c_it) {
        int k = *c_it;
        double s_k = beta(k) * c_old;
        s.emplace_back(s_k);
      }
    };
  };

  BENCHMARK("Cluster reordering")
  {
    clusters.update(5, clusters.n_clusters() - 1, 0.912);
  };
}

TEST_CASE("Thresholding", "[!benchmark]")
{

  double a = 0.1;
  int p = 100000;
  int j = 0;

  Eigen::VectorXd beta = Eigen::VectorXd::Random(p);

  for (int i = 0; i < p; i += 3) {
    double value = beta(i);
    int cluster_size = std::min(3, p - i);
    for (int j = 0; j < cluster_size; j++) {
      beta(i + j) = value;
    }
  }

  for (int i = 0; i < 1000; ++i) {
    beta(i) = 1.1;
  }

  Eigen::ArrayXd lambdas = slope::lambdaSequence(p, 0.2, "bh");

  slope::Clusters clusters(beta);

  BENCHMARK("Thresholding")
  {
    slope::slopeThreshold(a, j, lambdas, clusters);
  };
}

TEST_CASE("Normalization", "[!benchmark]")
{
  auto data = generateData(100, 500, "quadratic", 1, 0.01, 0.01);

  Eigen::SparseMatrix<double> x_sparse = data.x.sparseView();

  slope::Slope model;

  BENCHMARK("Dense: JIT")
  {
    model.setModifyX(false);
    model.path(data.x, data.y);
  };

  BENCHMARK("Dense: Modify X")
  {
    model.setModifyX(true);
    model.path(data.x, data.y);
  };

  BENCHMARK("Sparse: JIT")
  {
    model.setModifyX(false);
    model.path(x_sparse, data.y);
  };

  // Should currently be just as fast as the JIT version since
  // we actually do not modify X when it is sparse
  BENCHMARK("Sparse: Modify X")
  {
    model.setModifyX(true);
    model.path(x_sparse, data.y);
  };
}
