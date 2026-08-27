#include "generate_data.hpp"
#include <Eigen/SparseCore>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <iostream>
#include <slope/clusters.h>
#include <slope/cv.h>
#include <slope/math.h>
#include <slope/regularization_sequence.h>
#include <slope/slope.h>
#include <slope/solvers/hybrid_cd.h>
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

TEST_CASE("Full-set gradient materialization", "[!benchmark][full_set]")
{
  constexpr int n = 4;
  constexpr int p = 1'000'000;

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(n, p);
  Eigen::MatrixXd residual = Eigen::MatrixXd::Random(n, 1);
  Eigen::VectorXd gradient(p);
  Eigen::VectorXd x_centers = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd x_scales = Eigen::VectorXd::Ones(p);
  Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
  std::vector<int> full_set(p);
  std::iota(full_set.begin(), full_set.end(), 0);

  std::cout << "Materialized full-set index storage: " << p * sizeof(int)
            << " bytes\n";
  std::cout << "Allocation-free full-set index storage: 0 bytes\n";

  BENCHMARK("Materialized full set")
  {
    std::vector<int> materialized_indices(p);
    std::iota(materialized_indices.begin(), materialized_indices.end(), 0);
    slope::updateGradient(gradient,
                          x,
                          residual,
                          materialized_indices,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::None);
    return gradient.sum();
  };

  BENCHMARK("Pre-materialized full set")
  {
    slope::updateGradient(gradient,
                          x,
                          residual,
                          full_set,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::None);
    return gradient.sum();
  };

  BENCHMARK("Allocation-free full set")
  {
    slope::updateGradient(gradient,
                          x,
                          residual,
                          x_centers,
                          x_scales,
                          w,
                          slope::JitNormalization::None);
    return gradient.sum();
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

TEST_CASE("Wide strong-screening path", "[!benchmark][full_set]")
{
  constexpr int n = 10;
  constexpr int p = 100'000;

  auto data = generateData(n, p, "quadratic", 1, 1, 0.001);

  slope::Slope model;
  model.setPathLength(10);
  model.setScreening("strong");
  model.setSolver("fista");

  BENCHMARK("Strong screening with many coefficients")
  {
    return model.path(data.x, data.y);
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
    for (int j = 0; j < clusters.size(); ++j) {
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
    clusters.update(5, clusters.size() - 1, 0.912);
  };
}

TEST_CASE("Thresholding", "[!benchmark]")
{
  constexpr int p = 100000;
  constexpr int cluster_size = 100;
  constexpr int n_clusters = p / cluster_size;
  constexpr int j = n_clusters / 2;
  constexpr double hess = 1.7;

  Eigen::VectorXd beta(p);
  for (int cluster = 0; cluster < n_clusters; ++cluster) {
    beta.segment(cluster * cluster_size, cluster_size)
      .setConstant(n_clusters - cluster);
  }

  const Eigen::ArrayXd lambdas = slope::lambdaSequence(p, 0.2, "bh");
  const Eigen::ArrayXd lambda_cumsum = slope::cumSum(lambdas, true);
  slope::Clusters clusters(beta);
  const int start = clusters.pointer(j);
  const double lambda_sum =
    lambda_cumsum(start + cluster_size) - lambda_cumsum(start);
  const double x = clusters.coeff(j) + lambda_sum / hess;
  const double gamma = hess * x;

  BENCHMARK("Thresholding with materialized scaled cumsum")
  {
    return slope::slopeThreshold(x, j, lambda_cumsum / hess, clusters);
  };

  BENCHMARK("Thresholding with unscaled cumsum")
  {
    return slope::slopeThreshold(gamma, hess, j, lambda_cumsum, clusters);
  };
}

TEST_CASE("Sparse cluster gradient and Hessian benchmark",
          "[!benchmark][sparse_cluster]")
{
  constexpr int n = 20'000;
  constexpr int p = 5'000;
  constexpr int cluster_size = 256;
  constexpr int nonzeros_per_column = 20;

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(p * nonzeros_per_column);

  for (int j = 0; j < p; ++j) {
    for (int k = 0; k < nonzeros_per_column; ++k) {
      const int i = (37 * j + 997 * k) % n;
      const double value = 0.25 + (j + k) % 11;
      triplets.emplace_back(i, j, value);
    }
  }

  Eigen::SparseMatrix<double> x(n, p);
  x.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
  for (int j = 0; j < cluster_size; ++j) {
    beta(j) = j % 2 == 0 ? 1.0 : -1.0;
  }

  slope::Clusters clusters(beta);
  std::vector<int> signs;
  signs.reserve(cluster_size);
  for (auto it = clusters.cbegin(0); it != clusters.cend(0); ++it) {
    signs.emplace_back(slope::sign(beta(*it)));
  }

  const Eigen::MatrixXd weights = Eigen::MatrixXd::Ones(n, 1);
  const Eigen::MatrixXd residual = Eigen::VectorXd::Random(n);
  const Eigen::VectorXd x_centers = Eigen::VectorXd::LinSpaced(p, -0.01, 0.01);
  const Eigen::VectorXd x_scales = Eigen::VectorXd::LinSpaced(p, 0.5, 1.5);

  BENCHMARK("Sparse multi-feature cluster")
  {
    return slope::computeClusterGradientAndHessian(
      x,
      0,
      signs,
      clusters,
      weights,
      residual,
      x_centers,
      x_scales,
      slope::JitNormalization::Both);
  };
}

TEST_CASE("Sparse singleton gradient and Hessian benchmark",
          "[!benchmark][sparse_singleton]")
{
  constexpr int n = 20'242;
  constexpr int p = 47'236;
  constexpr int nonzeros = 32;
  constexpr int j = p / 2;

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(nonzeros);

  for (int k = 0; k < nonzeros; ++k) {
    const int i = (997 * k) % n;
    const double value = 0.25 + k % 11;
    triplets.emplace_back(i, j, value);
  }

  Eigen::SparseMatrix<double> x(n, p);
  x.setFromTriplets(triplets.begin(), triplets.end());

  const Eigen::MatrixXd weights = Eigen::VectorXd::LinSpaced(n, 0.25, 0.75);
  const Eigen::MatrixXd residual = Eigen::VectorXd::Random(n);
  const Eigen::VectorXd x_centers = Eigen::VectorXd::Constant(p, 0.01);
  const Eigen::VectorXd x_scales = Eigen::VectorXd::Constant(p, 1.25);

  BENCHMARK("Sparse singleton coordinate")
  {
    return slope::computeGradientAndHessian(x,
                                            j,
                                            weights,
                                            residual,
                                            x_centers,
                                            x_scales,
                                            1.0,
                                            slope::JitNormalization::Both,
                                            n);
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
