#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/slope.h>
#include <slope/solvers/pgd.h>

TEST_CASE("Poisson models", "[models][poisson]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  const int p = 3;

  Eigen::Matrix<double, n, p> x;
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

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  alpha[0] = 0.01;
  lambda << 2.0, 1.8, 1.0;

  slope::Slope model;

  model.setTol(1e-8);
  model.setMaxIt(1e4);
  model.setObjective("poisson");

  Eigen::Vector3d coefs_ref;

  SECTION("No intercept, no standardization")
  {
    model.setStandardize(false);
    model.setIntercept(false);

    model.setMaxIt(25);
    model.fit<slope::solvers::Hybrid>(x, y, alpha, lambda);

    auto dual_gaps = model.getDualGaps().front();

    REQUIRE(dual_gaps.front() >= 0);
    REQUIRE(dual_gaps.back() <= 1e-6);

    Eigen::VectorXd coefs_hybrid = model.getCoefs().front();

    model.setMaxIt(2000);
    model.fit<slope::solvers::PGD>(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();

    auto dual_gaps_pgd = model.getDualGaps().front();

    REQUIRE(dual_gaps_pgd.front() >= 0);
    REQUIRE(dual_gaps_pgd.back() <= 1e-6);

    coefs_ref << 0.1957634, -0.1612890, 0.1612890;

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coefs_ref, 1e-6));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-6));
  }

  SECTION("With intercept, no standardization")
  {
    model.setStandardize(false);
    model.setIntercept(true);

    coefs_ref << 0.3925911, -0.2360691, 0.4464808;

    model.setMaxIt(25);
    model.fit<slope::solvers::Hybrid>(x, y, alpha, lambda);

    Eigen::VectorXd coefs_hybrid = model.getCoefs().front();
    double intercept_hybrid = model.getIntercepts()[0][0];

    auto dual_gaps_hybrid = model.getDualGaps().front();

    REQUIRE(dual_gaps_hybrid.front() >= 0);
    REQUIRE(dual_gaps_hybrid.back() <= 1e-6);

    model.setMaxIt(1e4);
    model.fit<slope::solvers::PGD>(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();
    double intercept_pgd = model.getIntercepts()[0][0];

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept_hybrid, WithinRel(-0.5408344, 1e-4));
    REQUIRE_THAT(intercept_pgd, WithinRel(-0.5408344, 1e-4));
  }

  SECTION("With intercept, with standardization")
  {
    model.setStandardize(true);
    model.setIntercept(true);

    coefs_ref << 0.4017805, -0.2396130, 0.4600816;

    model.setMaxIt(25);
    model.fit<slope::solvers::Hybrid>(x, y, alpha, lambda);

    Eigen::VectorXd coefs = model.getCoefs().front();
    double intercept = model.getIntercepts()[0][0];

    model.setMaxIt(1e4);
    model.fit<slope::solvers::PGD>(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();
    double intercept_pgd = model.getIntercepts()[0][0];

    REQUIRE_THAT(coefs, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept, WithinRel(-0.5482493, 1e-4));
    REQUIRE_THAT(intercept_pgd, WithinRel(-0.5482493, 1e-4));
  }

  SECTION("Lasso penalty, no intercept")
  {

    Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
    Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

    lambda << 1.0, 1.0, 1.0;
    alpha[0] = 0.1;

    model.setStandardize(false);
    model.setIntercept(false);

    coefs_ref << 0.010928758, 0.0, 0.007616257;

    model.setMaxIt(25);
    model.fit<slope::solvers::Hybrid>(x, y, alpha, lambda);

    Eigen::VectorXd coefs_hybrid = model.getCoefs().front();

    auto dual_gaps_hybrid = model.getDualGaps().front();

    REQUIRE(dual_gaps_hybrid.front() >= 0);
    REQUIRE(dual_gaps_hybrid.back() <= 1e-6);

    model.setMaxIt(20000);
    model.fit<slope::solvers::PGD>(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
  }

  SECTION("Lasso penalty, with intercept")
  {

    Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
    Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

    lambda << 1.0, 1.0, 1.0;
    alpha[0] = 0.1;

    model.setStandardize(false);
    model.setIntercept(true);

    coefs_ref << 0.05533582, 0.0, 0.15185182;

    model.setMaxIt(25);
    model.fit<slope::solvers::Hybrid>(x, y, alpha, lambda);

    Eigen::VectorXd coefs_hybrid = model.getCoefs().front();
    double intercept_hybrid = model.getIntercepts()[0][0];

    auto dual_gaps_hybrid = model.getDualGaps().front();

    REQUIRE(dual_gaps_hybrid.front() >= 0);
    REQUIRE(dual_gaps_hybrid.back() <= 1e-6);

    model.setMaxIt(20000);
    model.fit<slope::solvers::PGD>(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = model.getCoefs().front();
    double intercept_pgd = model.getIntercepts()[0][0];

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept_hybrid, WithinRel(-0.39652440, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coefs_ref, 1e-4));
    REQUIRE_THAT(intercept_pgd, WithinRel(-0.39652440, 1e-4));
  }
}
