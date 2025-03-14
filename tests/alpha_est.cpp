#include "test_helpers.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
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
