#include "generate_data.hpp"
#include "load_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/cv.h>
#include <slope/losses/poisson.h>
#include <slope/slope.h>
#include <slope/threads.h>

TEST_CASE("Poisson models", "[poisson]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(10, p);
  Eigen::VectorXd y(n);

  // clang-format off
  x << 0.288,  -0.0452,  0.880,
       0.788,   0.576,  -0.305,
       1.510,   0.390,  -0.621,
      -2.210,  -1.120,  -0.0449,
      -0.0162,  0.944,   0.821,
       0.594,   0.919,   0.782,
       0.0746, -1.990,   0.620,
      -0.0561, -0.156,  -1.470,
      -0.478,   0.418,   1.360,
      -0.103,   0.388,  -0.0538;
  // clang-format on

  y << 2, 0, 1, 0, 0, 0, 1, 0, 1, 2;

  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  double alpha = 0.01;
  lambda << 2.0, 1.8, 1.0;

  slope::Slope model;

  model.setTol(1e-8);
  model.setLoss("poisson");
  model.setDiagnostics(true);

  Eigen::Vector3d coefs_ref;

  SECTION("No intercept, no standardization")
  {
    model.setNormalization("none");
    model.setIntercept(false);

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);

    auto dual_gaps = fit.getGaps();

    REQUIRE(dual_gaps.front() >= 0);
    REQUIRE(dual_gaps.back() <= 1e-6);

    Eigen::VectorXd coefs_hybrid = fit.getCoefs();

    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();

    auto dual_gaps_pgd = fit.getGaps();

    REQUIRE(dual_gaps_pgd.front() >= 0);
    REQUIRE(dual_gaps_pgd.back() <= 1e-6);

    coefs_ref << 0.1957634, -0.1612890, 0.1612890;

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
  }

  SECTION("With intercept, no standardization")
  {
    model.setNormalization("none");
    model.setIntercept(true);

    coefs_ref << 0.3925911, -0.2360691, 0.4464808;

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_hybrid = fit.getCoefs();
    double intercept_hybrid = fit.getIntercepts()[0];

    auto dual_gaps_hybrid = fit.getGaps();

    REQUIRE(dual_gaps_hybrid.front() >= 0);
    REQUIRE(dual_gaps_hybrid.back() <= 1e-6);

    model.setMaxIterations(1e4);
    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();
    double intercept_pgd = fit.getIntercepts()[0];

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept_hybrid, WithinRel(-0.5408344, 1e-4));
    REQUIRE_THAT(intercept_pgd, WithinRel(-0.5408344, 1e-4));
  }

  SECTION("With intercept, with standardization")
  {
    model.setNormalization("standardization");
    model.setIntercept(true);

    coefs_ref << 0.4017805, -0.2396130, 0.4600816;

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs = fit.getCoefs();
    double intercept = fit.getIntercepts()[0];

    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();
    double intercept_pgd = fit.getIntercepts()[0];

    REQUIRE_THAT(coefs, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept, WithinRel(-0.5482493, 1e-3));
    REQUIRE_THAT(intercept_pgd, WithinRel(-0.5482493, 1e-4));
  }

  SECTION("Lasso penalty, no intercept")
  {

    double alpha = 0.1;
    Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

    lambda << 1.0, 1.0, 1.0;

    model.setNormalization("none");
    model.setIntercept(false);

    coefs_ref << 0.010928758, 0.0, 0.007616257;

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_hybrid = fit.getCoefs();

    auto dual_gaps_hybrid = fit.getGaps();

    REQUIRE(dual_gaps_hybrid.front() >= 0);
    REQUIRE(dual_gaps_hybrid.back() <= 1e-6);

    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
  }

  SECTION("Lasso penalty, with intercept")
  {

    Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

    lambda << 1.0, 1.0, 1.0;
    double alpha = 0.1;

    model.setNormalization("none");
    model.setIntercept(true);

    coefs_ref << 0.05533582, 0.0, 0.15185182;

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_hybrid = fit.getCoefs();
    double intercept_hybrid = fit.getIntercepts()[0];

    auto dual_gaps_hybrid = fit.getGaps();

    REQUIRE(dual_gaps_hybrid.front() >= 0);
    REQUIRE(dual_gaps_hybrid.back() <= 1e-6);

    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();
    double intercept_pgd = fit.getIntercepts()[0];

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept_hybrid, WithinRel(-0.39652440, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept_pgd, WithinRel(-0.39652440, 1e-4));
  }
}

TEST_CASE("Poisson abalone data", "[poisson][realdata]")
{
  auto [x, y] = loadData("tests/data/abalone.csv");

  slope::Slope model;

  model.setLoss("poisson");
  model.setSolver("hybrid");

  auto path = model.path(x, y);

  REQUIRE(path.getDeviance().back() > 0);
  REQUIRE(path.getDeviance().size() > 1);
}

TEST_CASE("Poisson predictions", "[poisson][predict]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(3, 2);
  Eigen::VectorXd beta(2);
  Eigen::VectorXd eta(3);

  // clang-format off
  x << 1.1, 2.3,
       0.2, 1.5,
       0.5, 0.2;
  // clang-format on
  beta << 1, 2;

  eta = x * beta;

  slope::Poisson loss;

  auto pred = loss.predict(eta);
  std::array<double, 3> expected = { 298.867, 24.5325, 2.4596 };

  REQUIRE_THAT(pred.reshaped(), VectorApproxEqual(expected, 1e-3));
}

TEST_CASE("Poisson low regularization", "[poisson]")
{
  auto data = generateData(100, 5, "poisson", 1, 1, 1);

  slope::Slope model;

  model.setLoss("poisson");
  model.setSolver("pgd");

  double alpha = 1e-5;

  auto fit = model.fit(data.x, data.y, alpha);
}

TEST_CASE("Poisson sparse and dense methods agree", "[poisson][sparse]")
{
  using namespace Catch::Matchers;

  const int n = 100;
  const int p = 2;

  Eigen::MatrixXd x_dense(n, p);
  Eigen::VectorXd y(n);

  x_dense << -0.240, 0.210, -0.890, 0.000, 0.000, -0.590, -0.710, 2.000, 0.000,
    0.000, 0.680, -1.100, 0.070, 0.000, 0.280, 0.000, 0.000, 0.000, 0.860,
    -1.600, -0.590, 0.000, 0.000, 0.900, 0.000, 0.000, 0.000, 0.420, 0.000,
    0.000, 0.000, -0.290, -1.700, 1.200, 0.000, 0.000, 0.430, 0.950, 0.000,
    -1.500, 0.000, -0.940, -0.540, 1.200, 0.000, 1.700, -0.660, 0.150, 0.000,
    1.300, -1.800, -0.026, 0.740, 0.000, -0.960, 0.000, -0.470, 0.000, 0.000,
    0.000, -0.440, 0.000, -0.600, -0.560, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.760, 0.046, 0.350, 0.000, 0.000, 0.140,
    0.000, 0.000, -0.015, -2.000, 0.000, 0.000, -1.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.230, 1.200, 0.000, -0.470, 0.000, 2.100, 0.000, 0.180,
    0.000, 0.770, 0.000, -0.310, -0.096, 0.000, 0.000, 0.000, 0.000, -0.590,
    1.300, 0.000, -0.360, 0.000, 0.000, -0.410, 0.270, -1.700, 0.000, 0.000,
    0.240, 0.000, 0.000, 0.000, 0.091, -0.770, 1.200, 0.990, 1.500, 0.000,
    0.000, -0.800, 0.000, 0.000, -1.300, 0.000, -1.000, 0.089, 1.500, 0.000,
    -0.810, -0.850, 0.000, -0.480, 1.700, 1.300, 1.100, 0.000, 0.000, 2.300,
    0.000, 0.000, -0.610, 0.630, 0.000, 0.120, 0.780, 0.000, 0.000, 0.000,
    0.000, -0.460, 0.120, -0.840, -0.720, 0.000, 1.300, 0.140, 0.000, -2.300,
    -1.500, -0.760, 0.000, 0.360, -1.100, -1.200, 0.000, 0.630, 0.550, 0.370,
    0.250, 0.000, 0.000, 0.000, -0.980, -0.450, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, -0.860, -1.500, 0.000, 0.000, 0.000, 0.000, 0.000, -0.200;

  y << 0, 3, 0, 1, 0, 1, 2, 3, 2, 0, 1, 0, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0, 4, 2,
    2, 4, 0, 2, 1, 1, 1, 2, 1, 2, 1, 0, 0, 0, 2, 3, 1, 1, 1, 1, 0, 3, 1, 2, 1,
    1, 0, 0, 0, 0, 0, 0, 1, 3, 1, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 0, 2, 1, 5, 1,
    0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 1, 1, 0, 0,
    1;

  Eigen::SparseMatrix<double> x_sparse = x_dense.sparseView();

  Eigen::VectorXd x_centers_sparse(p);
  Eigen::VectorXd x_scales_sparse(p);
  Eigen::VectorXd x_centers_dense(p);
  Eigen::VectorXd x_scales_dense(p);

  slope::computeCenters(x_centers_sparse, x_sparse, "mean");
  slope::computeScales(x_scales_sparse, x_sparse, "sd");
  slope::computeCenters(x_centers_dense, x_dense, "mean");
  slope::computeScales(x_scales_dense, x_dense, "sd");

  REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_sparse, 1e-5));
  REQUIRE_THAT(x_scales_dense, VectorApproxEqual(x_scales_sparse, 1e-5));

  slope::Slope model;

  model.setCentering("none");
  model.setScaling("sd");
  model.setLoss("poisson");
  model.setSolver("pgd");
  model.setTol(1e-8);

  slope::Threads::set(1);

  auto res_dense = model.path(x_dense, y);
  auto res_sparse = model.path(x_sparse, y);

  auto dense_coefs = res_dense.getCoefs();
  auto sparse_coefs = res_sparse.getCoefs();

  Eigen::VectorXd dense_ref = dense_coefs[3];
  Eigen::VectorXd sparse_ref = sparse_coefs[3];

  REQUIRE_THAT(dense_ref, VectorApproxEqual(sparse_ref, 1e-4));
}
