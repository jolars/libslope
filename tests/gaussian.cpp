#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <slope/slope.h>

TEST_CASE("Simple low-dimensional design", "[gaussian][basic]")
{
  using namespace Catch::Matchers;

  Eigen::Matrix<double, 3, 2> x;
  Eigen::Vector2d beta;
  Eigen::Vector3d y;

  // clang-format off
  x << 1, 2,
       0, 1,
       1, 0;
  // clang-format on
  beta << 1, 1;

  y = x * beta;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(2);

  slope::Slope model;

  model.setIntercept(false);
  model.setStandardize(false);

  model.fit(x, y, alpha, lambda);

  auto coefs = model.getCoefs();
  auto dual_gaps = model.getDualGaps();
  auto primals = model.getPrimals();

  Eigen::VectorXd coef = coefs.col(0);

  REQUIRE_THAT(coef, VectorApproxEqual(beta, 1e-4));

  double gap = dual_gaps.front().back();
  double primal = primals.front().back();

  REQUIRE(gap <= (primal + 1e-10) * 1e-4);
}

TEST_CASE("X is identity", "[gaussian][identity]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(4, 4);
  Eigen::Vector4d y;
  y << 8, 6, 4, 2;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Ones(1);
  Eigen::Array4d lambda;
  lambda << 1, 0.75, 0.5, 0.25;

  slope::Slope model;
  model.setIntercept(false);
  model.setStandardize(false);
  model.setPrintLevel(3);
  model.fit(x, y, alpha, lambda);
  // model.fit(x, y);
  //
  // double gap = model.getDualGaps()[0].back();
  // double primal = model.getPrimals()[0].back();
  //
  // Eigen::VectorXd coefs = model.getCoefs().col(0);
  // Eigen::VectorXd betas = coefs.col(0);
  //
  // std::array<double, 4> expected = { 4.0, 3.0, 2.0, 1.0 };
  //
  // REQUIRE_THAT(betas, VectorApproxEqual(expected));
  // REQUIRE(gap < primal * 1e-4);
}

TEST_CASE("Automatic lambda and alpha", "[gaussian]")
{
  using namespace Catch::Matchers;

  Eigen::Matrix<double, 3, 2> x;
  // clang-format off
  x << 1, 2,
       0, 1,
       1, 0;
  // clang-format on
  Eigen::Vector3d y;
  y << -1, 6, 2;

  slope::Slope model;

  REQUIRE_NOTHROW(model.fit(x, y));
}

TEST_CASE("Gaussian models", "[gaussian]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  const int p = 3;

  Eigen::Matrix<double, n, p> x;
  Eigen::VectorXd beta(p);

  // clang-format off
    x <<  0.288,  -0.0452,  0.880,
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

  // Fixed coefficients beta
  beta << 1, -1, 0.2;

  // Compute linear predictor
  Eigen::VectorXd y = x * beta;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  alpha[0] = 0.05;

  lambda << 3.0, 2.0, 2.0;

  slope::Slope model;

  model.setTol(1e-9);
  model.setObjective("gaussian");

  SECTION("No intercept, no standardization")
  {
    model.setStandardize(false);
    model.setIntercept(false);

    Eigen::Vector3d coef_target;
    coef_target << 0.6864545, -0.6864545, 0.0000000;

    // PGD
    model.setPgdFreq(1);
    model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_pgd = model.getCoefs().col(0);

    // Hybrid
    model.setPgdFreq(10);
    model.fit(x, y, alpha, lambda);
    // model.setUpdateClusters(true);

    Eigen::VectorXd coefs_hybrid = model.getCoefs().col(0);

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-6));
  }
}
