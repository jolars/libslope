#include "generate_data.hpp"
#include "slope/estimate_alpha.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include <slope/ols.h>

TEST_CASE("OLS on small dense matrix without intercept", "[ols]")
{
  // Simple system: y = 2*x + 3*z
  Eigen::MatrixXd X(3, 2);
  Eigen::VectorXd y(3);

  // clang-format off
  X << 1.0, 2.0,
       2.0, 1.0,
       3.0, 4.0;
  y << 7.0, 5.0, 11.0;
  // clang-format on

  auto [intercept, coeffs] = fitOls(X, y, /*fit_intercept=*/false);
  REQUIRE(intercept == 0.0);

  // Expected: [2, 3]
  Eigen::VectorXd expected(2);
  expected << 1.105263, 2.157895;

  REQUIRE_THAT(coeffs, VectorApproxEqual(expected, 1e-6));
}

TEST_CASE("OLS on small dense matrix with intercept", "[ols]")
{
  // y = 10 + 1*x + 2*z
  Eigen::MatrixXd X(3, 2);
  Eigen::VectorXd y(3);

  // clang-format off
    X << 1.0, 1.0,
         2.0, 1.0,
         5.0, 2.0;
    y << 10 + 1*1.0 + 2*1.0,   // = 13
         10 + 1*2.0 + 2*1.0,   // = 14
         10 + 1*5.0 + 2*2.0;   // = 19
  // clang-format on

  auto [intercept, coeffs] = fitOls(X, y, /*fit_intercept=*/true);

  REQUIRE_THAT(intercept, Catch::Matchers::WithinAbs(10.0, 1e-8));

  // Expected: [1, 2]
  Eigen::VectorXd expected(2);
  expected << 1.0, 2.0;

  REQUIRE_THAT(coeffs, VectorApproxEqual(expected, 1e-6));
}

TEST_CASE("OLS on small sparse matrix with intercept", "[ols][sparse]")
{
  // Constructing a sparse matrix with the same relationship:
  // y = 5 + 2*x + 3*z
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(6);

  // Non-zero entries for a 3x2 matrix
  // Row 0: X(0,0)=1, X(0,1)=1
  // Row 1: X(1,0)=2, X(1,1)=1
  // Row 2: X(2,0)=0, X(2,1)=3
  tripletList.push_back(T(0, 0, 1.0));
  tripletList.push_back(T(0, 1, 1.0));
  tripletList.push_back(T(1, 0, 2.0));
  tripletList.push_back(T(1, 1, 1.0));
  tripletList.push_back(T(2, 1, 3.0));

  Eigen::SparseMatrix<double> X(3, 2);
  X.setFromTriplets(tripletList.begin(), tripletList.end());

  Eigen::VectorXd y(3);
  y << 5 + 2 * 1.0 + 3 * 1.0, // = 10
    5 + 2 * 2.0 + 3 * 1.0,    // = 12
    5 + 2 * 0.0 + 3 * 3.0;    // = 14

  auto [intercept, coeffs] = fitOls(X, y, /*fit_intercept=*/true);

  REQUIRE_THAT(intercept, Catch::Matchers::WithinAbs(5.0, 1e-8));

  // Expected: [2, 3]
  Eigen::VectorXd expected(2);
  expected << 2.0, 3.0;
  REQUIRE_THAT(coeffs, VectorApproxEqual(expected, 1e-6));
}

TEST_CASE("estimateNoise basic functionality", "[estimate_alpha]")
{
  // Create a simple regression problem with known noise level
  int n = 100; // observations
  int p = 5;   // predictors

  // Create design matrix and response with known coefficients
  Eigen::MatrixXd x = Eigen::MatrixXd::Random(n, p);
  Eigen::VectorXd true_coefs = Eigen::VectorXd::Random(p);
  double true_intercept = 2.5;

  // Set noise level
  double sigma = 1.0;
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::normal_distribution<double> noise_dist(0, sigma);

  // Generate response with noise
  Eigen::MatrixXd y =
    x * true_coefs + Eigen::VectorXd::NullaryExpr(n, [&](int) {
      return noise_dist(gen) + true_intercept;
    });

  SECTION("With intercept")
  {
    double estimated_sigma = slope::estimateNoise(x, y, true);
    // Estimated sigma should be close to true sigma
    REQUIRE(estimated_sigma > 0.0);
    REQUIRE_THAT(estimated_sigma, Catch::Matchers::WithinAbs(sigma, 0.3));
  }

  SECTION("Without intercept - when true model has intercept")
  {
    // Without intercept, the estimate should be biased upward
    double estimated_sigma = slope::estimateNoise(x, y, false);
    REQUIRE(estimated_sigma > 0.0);
    REQUIRE(estimated_sigma > sigma);
  }
}

TEST_CASE("estimateAlpha for n >= p + 30", "[estimate_alpha]")
{
  // Create a case where n is much larger than p
  int n = 100; // observations
  int p = 10;  // predictors

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(n, p);
  Eigen::VectorXd true_coefs = Eigen::VectorXd::Random(p);
  // Make some coefficients exactly zero to simulate sparsity
  true_coefs(2) = 0.0;
  true_coefs(5) = 0.0;
  true_coefs(8) = 0.0;

  double true_intercept = 1.5;
  double sigma = 0.5;

  std::mt19937 gen(123);
  std::normal_distribution<double> noise_dist(0, sigma);

  Eigen::MatrixXd y =
    x * true_coefs + Eigen::VectorXd::NullaryExpr(n, [&](int) {
      return noise_dist(gen) + true_intercept;
    });

  // Setup slope model
  slope::Slope model;
  model.setIntercept(true);

  // Run alpha estimation
  auto path = slope::estimateAlpha(x, y, model);

  // Check that result is valid
  REQUIRE(path.getAlpha().size() == 1);
  REQUIRE(path.getCoefs().size() > 0);

  // Some basic sanity checks
  auto coefs = path.getCoefs().back();
  REQUIRE(coefs.nonZeros() > 0);  // Should select at least some variables
  REQUIRE(coefs.nonZeros() <= p); // Should not select more than p variables
}

TEST_CASE("estimateAlpha for n < p + 30", "[estimate_alpha]")
{
  // Test the iterative procedure case
  int n = 25; // observations
  int p = 20; // predictors - making n < p + 30

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(n, p);
  // Creating a sparse true coefficient vector
  Eigen::VectorXd true_coefs = Eigen::VectorXd::Zero(p);
  // Only 5 non-zero coefficients
  true_coefs(0) = 2.0;
  true_coefs(5) = -1.5;
  true_coefs(10) = 3.0;
  true_coefs(15) = -2.5;
  true_coefs(19) = 1.8;

  double sigma = 0.3;
  std::mt19937 gen(456);
  std::normal_distribution<double> noise_dist(0, sigma);

  // No intercept in this test case
  Eigen::MatrixXd y =
    x * true_coefs +
    Eigen::VectorXd::NullaryExpr(n, [&](int) { return noise_dist(gen); });

  // Setup slope model
  slope::Slope model;
  model.setIntercept(false);
  model.setAlphaEstimationMaxIterations(20); // Ensure enough iterations

  // Run alpha estimation
  auto path = slope::estimateAlpha(x, y, model);

  // Check that result is valid
  REQUIRE(path.getAlpha().size() > 0);
  REQUIRE(path.getCoefs().size() > 0);

  // Sanity checks
  auto coefs = path.getCoefs().back();
  REQUIRE(coefs.nonZeros() > 0); // Should select some variables
  REQUIRE(coefs.nonZeros() < n); // Should select fewer than n variables

  // Check if at least some of the true non-zero coefficients are selected
  int true_nonzeros_selected = 0;
  for (Eigen::SparseMatrix<double>::InnerIterator it(coefs, 0); it; ++it) {
    if (true_coefs(it.row()) != 0) {
      true_nonzeros_selected++;
    }
  }
  // We should identify at least some of the true non-zero coefficients
  REQUIRE(true_nonzeros_selected > 0);
}

TEST_CASE("estimateAlpha error handling", "[estimate_alpha]")
{
  // Test error cases
  int n = 15; // Small number of observations
  int p = 14; // Almost as many predictors as observations

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(n, p);
  Eigen::MatrixXd y = Eigen::VectorXd::Random(n);

  slope::Slope model;
  model.setIntercept(false);

  SECTION("When max iterations is reached")
  {
    // Set a very low max iteration count
    model.setAlphaEstimationMaxIterations(1);

    // This should throw due to max iterations
    REQUIRE_THROWS_AS(slope::estimateAlpha(x, y, model), std::runtime_error);
  }
}

TEST_CASE("Full fit with estimate alpha", "[estimate_alpha][fail]")
{
  int n = 100;
  int p = 20;

  auto data = generateData(n, p);

  slope::Slope model;

  model.setAlphaType("estimate");

  REQUIRE_NOTHROW(model.fit(data.x, data.y));

  model.setLoss("logistic");

  REQUIRE_THROWS_AS(model.fit(data.x, data.y), std::invalid_argument);
}
