#include "slope/losses/logistic.h"
#include "generate_data.hpp"
#include "slope/slope.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Logistic, simple fixed design", "[logistic]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
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
  beta << 0.5, -0.1, 0.2;

  // Compute linear predictor
  Eigen::VectorXd linear_predictor = x * beta;

  // Compute probabilities using logistic function
  Eigen::VectorXd prob = linear_predictor.unaryExpr(
    [](double x) { return 1.0 / (1.0 + std::exp(-x)); });

  // Generate deterministic response variable y
  Eigen::VectorXd y =
    prob.unaryExpr([](double p) { return p > 0.5 ? 1.0 : 0.0; });

  double alpha = 0.05;
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  lambda << 2.128045, 1.833915, 1.644854;

  slope::Slope model;

  model.setTol(1e-7);
  model.setLoss("logistic");
  model.setDiagnostics(true);

  slope::SlopeFit fit;

  SECTION("No intercept, no standardization")
  {
    model.setNormalization("none");
    model.setIntercept(false);

    Eigen::Vector3d coef_target;
    coef_target << 1.3808558, 0.0000000, 0.3205496;

    model.setSolver("pgd");

    fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_pgd = fit.getCoefs();

    auto dual_gaps = fit.getGaps();

    REQUIRE(dual_gaps.front() >= 0);
    REQUIRE(dual_gaps.back() >= 0);
    REQUIRE(dual_gaps.back() <= 1e-4);

    model.setSolver("hybrid");

    fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coefs_hybrid = fit.getCoefs();

    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("Intercept, no standardization")
  {
    model.setIntercept(true);
    model.setNormalization("none");

    std::vector<double> coef_target = { 1.2748806, 0.0, 0.2062611 };
    double intercept_target = 0.3184528;

    model.setSolver("pgd");
    model.setMaxIterations(1e7);
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coef_pgd = fit.getCoefs();
    double intercept_pgd = fit.getIntercepts()[0];

    REQUIRE_THAT(coef_pgd, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(intercept_pgd, WithinAbs(intercept_target, 1e-4));

    model.setSolver("hybrid");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_hybrid = fit.getCoefs();
    double intercept_hybrid = fit.getIntercepts()[0];

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(intercept_hybrid, WithinAbs(intercept_target, 1e-4));
  }
}

TEST_CASE("Logistic path", "[logistic]")
{

  slope::Slope model;
  model.setLoss("logistic");
  model.setDiagnostics(true);

  auto data = generateData(1000, 100, "logistic", 1, 0.4, 0.5, 93);

  auto fit = model.path(data.x, data.y);

  auto null_deviance = fit.getNullDeviance();
  auto deviances = fit.getDeviance();
  auto gaps = fit.getGaps();

  for (auto& gap : gaps) {
    REQUIRE(gap.back() >= -1e-12); // Allow for minimal floating point error
  }

  REQUIRE(null_deviance >= 0);
  REQUIRE(deviances.size() > 10);
  REQUIRE(deviances.size() < 100);
  REQUIRE(deviances.back() > 0);
  REQUIRE_THAT(deviances, VectorMonotonic(false, true));
}

TEST_CASE("Logistic predictions", "[logistic][predict]")
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

  slope::Logistic loss;

  auto pred = loss.predict(eta);

  std::array<double, 3> expected = { 1, 1, 1 };

  REQUIRE_THAT(pred.reshaped(), VectorApproxEqual(expected));
}

TEST_CASE("Failing example (at one point)", "[logistic]")
{
  int n = 30;
  int p = 5;
  Eigen::MatrixXd x(n, p);
  Eigen::VectorXd y(n);

  // clang-format off
  x << -0.62645381,  1.35867955,  2.401617761, -0.54252003, -0.50595746, 
        0.18364332, -0.10278773, -0.039240003,  1.20786781,  1.34303883, 
       -0.83562861,  0.38767161,  0.689739362,  1.16040262, -0.21457941, 
        1.59528080, -0.05380504,  0.028002159,  0.70021365, -0.17955653, 
        0.32950777, -1.37705956, -0.743273209,  1.58683345, -0.10019074, 
       -0.82046838, -0.41499456,  0.188792300,  0.55848643,  0.71266631, 
        0.48742905, -0.39428995, -1.804958629, -1.27659221, -0.07356440, 
        0.73832471, -0.05931340,  1.465554862, -0.57326541, -0.03763417, 
        0.57578135,  1.10002537,  0.153253338, -1.22461261, -0.68166048, 
       -0.30538839,  0.76317575,  2.172611670, -0.47340064, -0.32427027, 
        1.51178117, -0.16452360,  0.475509529, -0.62036668,  0.06016044, 
        0.38984324, -0.25336168, -0.709946431,  0.04211587, -0.58889449, 
       -0.62124058,  0.69696338,  0.610726353, -0.91092165,  0.53149619, 
       -2.21469989,  0.55666320, -0.934097632,  0.15802877, -1.51839408, 
        1.12493092, -0.68875569, -1.253633400, -0.65458464,  0.30655786, 
       -0.04493361, -0.70749516,  0.291446236,  1.76728727, -1.53644982, 
       -0.01619026,  0.36458196, -0.443291873,  0.71670748, -0.30097613, 
        0.94383621,  0.76853292,  0.001105352,  0.91017423, -0.52827990, 
        0.82122120, -0.11234621,  0.074341324,  0.38418536, -0.65209478, 
        0.59390132,  0.88110773, -0.589520946,  1.68217608, -0.05689678, 
        0.91897737,  0.39810588, -0.568668733, -0.63573645, -1.91435943, 
        0.78213630, -0.61202639, -0.135178615, -0.46164473,  1.17658331, 
        0.07456498,  0.34111969,  1.178086997,  1.43228224, -1.66497244, 
       -1.98935170, -1.12936310, -1.523566800, -0.65069635, -0.46353040, 
        0.61982575,  1.43302370,  0.593946188, -0.20738074, -1.11592011, 
       -0.05612874,  1.98039990,  0.332950371, -0.39280793, -0.75081900, 
       -0.15579551, -0.36722148,  1.063099837, -0.31999287,  2.08716655, 
       -1.47075238, -1.04413463, -0.304183924, -0.27911330,  0.01739562, 
       -0.47815006,  0.56971963,  0.370018810,  0.49418833, -1.28630053, 
        0.41794156, -0.13505460,  0.267098791, -0.17733048, -1.64060553;
  // clang-format on

  y << 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 0;

  slope::Slope model;

  // model.setTol(1e-7);
  model.setLoss("logistic");
  model.setDiagnostics(true);
  model.setPathLength(100);

  auto res = model.path(x, y);
}
